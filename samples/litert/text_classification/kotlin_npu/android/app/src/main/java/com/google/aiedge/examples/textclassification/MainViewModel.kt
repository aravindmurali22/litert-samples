/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.textclassification

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainViewModel(private val textClassificationHelper: TextClassificationHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val textClassificationHelper = TextClassificationHelper(context)
                return MainViewModel(textClassificationHelper) as T
            }
        }
    }

    private val currentModel = MutableStateFlow(TextClassificationHelper.Model.MobileBERT)
    private val currentDelegate = MutableStateFlow(TextClassificationHelper.AcceleratorEnum.CPU)
    private val statusMessage = MutableStateFlow<String?>("Ready on CPU (MobileBERT)")

    init {
        viewModelScope.launch(Dispatchers.IO) {
            textClassificationHelper.initClassifier(currentModel.value, currentDelegate.value)
        }
    }

    private var classifyJob: Job? = null

    private val percentages = textClassificationHelper.percentages.distinctUntilChanged().stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), Pair(FloatArray(0), 0L)
    )

    private val errorMessage = MutableStateFlow<Throwable?>(null).also { state ->
        viewModelScope.launch {
            textClassificationHelper.error.collect { throwable ->
                state.value = throwable
                if (throwable != null) {
                    statusMessage.value = throwable.message
                } else {
                    statusMessage.value = "Ready on ${currentDelegate.value} (${currentModel.value})"
                }
            }
        }
    }

    val uiState: StateFlow<UiState> = combine(
        percentages,
        currentModel,
        currentDelegate,
        errorMessage,
        statusMessage,
    ) { percentages, model, delegate, throwable, status ->
        textClassificationHelper.completableDeferred?.complete(Unit)
        val percents = percentages.first
        UiState(
            negativePercentage = if (percents.isNotEmpty()) percents.first() else 0f,
            positivePercentage = if (percents.isNotEmpty()) percents.last() else 0f,
            inferenceTime = percentages.second,
            currentModel = model,
            currentDelegate = delegate,
            errorMessage = throwable?.message,
            statusMessage = status,
        )
    }.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState()
    )

    /**
     * Cancel current job and create a new coroutine to classify the new input text.
     */
    fun runClassification(inputText: String) {
        classifyJob?.cancel()
        classifyJob = viewModelScope.launch {
            textClassificationHelper.classify(inputText)
        }
    }

    /**
     * Stop classification and setup CompiledModel with new model.
     */
    fun setModel(model: TextClassificationHelper.Model) {
        currentModel.value = model
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                textClassificationHelper.initClassifier(model, currentDelegate.value)
            }
        }
    }

    /**
     * Stop classification and setup CompiledModel with new accelerator delegate (CPU/GPU/NPU).
     */
    fun setDelegate(delegate: TextClassificationHelper.AcceleratorEnum) {
        currentDelegate.value = delegate
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                textClassificationHelper.initClassifier(currentModel.value, delegate)
            }
        }
    }

    /** Clear error message after it has been consumed. */
    fun errorMessageShown() {
        errorMessage.update { null }
    }

    override fun onCleared() {
        super.onCleared()
        textClassificationHelper.stopClassify()
    }
}