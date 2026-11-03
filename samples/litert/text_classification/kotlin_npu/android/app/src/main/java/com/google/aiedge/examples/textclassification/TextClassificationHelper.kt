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
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.support.common.FileUtil
import com.google.ai.edge.litert.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext

class TextClassificationHelper(private val context: Context) {
    /** The LiteRT CompiledModel instance. */
    private var model: CompiledModel? = null
    private var environment: Environment? = null
    private val vocabularyMap = mutableMapOf<String, Int>()

    var currentModel: Model = Model.MobileBERT
        private set

    var currentDelegate: AcceleratorEnum = AcceleratorEnum.CPU
        private set

    /** Emits pair of (probability array, inference time in ms). */
    val percentages: SharedFlow<Pair<FloatArray, Long>>
        get() = _percentages
    private val _percentages = MutableSharedFlow<Pair<FloatArray, Long>>(
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>(
        replay = 1,
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    var completableDeferred: CompletableDeferred<Unit>? = null

    /** Init a CompiledModel from [modelEnum] and [delegate]. */
    fun initClassifier(
        modelEnum: Model = currentModel,
        delegate: AcceleratorEnum = currentDelegate,
    ) {
        stopClassify()
        currentModel = modelEnum
        currentDelegate = delegate
        try {
            // Load vocabulary using MetadataExtractor
            val litertBuffer = FileUtil.loadMappedFile(context, modelEnum.fileName)
            loadModelMetadata(litertBuffer)

            val appContext = context.applicationContext
            val accelerator = toAccelerator(delegate)
            val createdEnv: Environment = if (delegate == AcceleratorEnum.NPU) {
                runCatching {
                    System.loadLibrary("QnnSystem")
                    System.loadLibrary("QnnHtp")
                }
                val npuEnv = Environment.create(appContext, BuiltinNpuAcceleratorProvider(appContext))
                val available = npuEnv.getAvailableAccelerators()
                if (Accelerator.NPU !in available) {
                    npuEnv.close()
                    throw IllegalStateException(
                        "NPU accelerator unavailable on this device (available: ${available.joinToString()}). " +
                            "Bundle QNN libraries or run on a device with supported NPU hardware."
                    )
                }
                npuEnv
            } else {
                Environment.create(appContext)
            }

            val modelOptions = CompiledModel.Options(accelerator).apply {
                if (delegate == AcceleratorEnum.NPU) {
                    qualcommOptions = CompiledModel.QualcommOptions(
                        htpPerformanceMode =
                            CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
                    )
                }
            }

            environment = createdEnv
            model = CompiledModel.create(
                context.assets,
                modelEnum.fileName,
                modelOptions,
                createdEnv
            )
            _error.tryEmit(null)
            Log.i(TAG, "Done creating CompiledModel from ${modelEnum.fileName} on $delegate")
        } catch (e: Exception) {
            Log.e(
                TAG,
                "Create CompiledModel from ${modelEnum.fileName} on $delegate failed: ${e.message}",
                e
            )
            _error.tryEmit(e)
        }
    }

    /** Stop current model and release resources. */
    fun stopClassify() {
        runCatching { model?.close() }
        model = null
        runCatching { environment?.close() }
        environment = null
    }

    /** Run classify [inputText] using LiteRT CompiledModel API. */
    suspend fun classify(inputText: String) {
        withContext(Dispatchers.IO) {
            val localModel = model ?: run {
                _error.emit(
                    Throwable(
                        "Classifier is not initialized on $currentDelegate. Select an available accelerator."
                    )
                )
                return@withContext
            }

            try {
                // 1. Prepare buffers
                val inputBuffers = localModel.createInputBuffers()
                val outputBuffers = localModel.createOutputBuffers()

                // MobileBERT expects 128 tokens (512 bytes buffer),
                // AverageWordVec expects 256 tokens (1024 bytes buffer).
                val expectedInputSize = expectedTokenCount(currentModel)

                val tokenizerText = tokenizeText(inputText)
                if (tokenizerText.size > expectedInputSize) {
                    Log.e(TAG, "The number of word exceeds the limit: $expectedInputSize")
                    _error.emit(
                        Throwable(
                            "The number of word exceeds the limit, please input the number of word <= $expectedInputSize"
                        )
                    )

                    for (buffer in inputBuffers) { buffer.close() }
                    for (buffer in outputBuffers) { buffer.close() }
                    return@withContext
                }

                // Write input_ids
                val paddedIntArray = IntArray(expectedInputSize) { index ->
                    if (index < tokenizerText.size) tokenizerText[index] else 0
                }
                inputBuffers[0].writeInt(paddedIntArray)

                // If the model expects multiple inputs (like MobileBERT), we must initialize them
                // to prevent raw native uninitialized memory garbage from crashing the ML operations.
                if (inputBuffers.size > 1) {
                    // input_mask -> 1s for valid tokens, 0 for padded space
                    val inputMaskArray = IntArray(expectedInputSize) { index ->
                        if (index < tokenizerText.size) 1 else 0
                    }
                    inputBuffers[1].writeInt(inputMaskArray)
                }

                if (inputBuffers.size > 2) {
                    // segment_ids -> 0s for all tokens
                    val segmentIdsArray = IntArray(expectedInputSize) { 0 }
                    inputBuffers[2].writeInt(segmentIdsArray)
                }

                val startTime = SystemClock.uptimeMillis()
                localModel.run(inputBuffers, outputBuffers)
                val inferenceTime = SystemClock.uptimeMillis() - startTime

                val outputFloatArray = outputBuffers[0].readFloat()

                // Close buffers to free memory
                for (buffer in inputBuffers) { buffer.close() }
                for (buffer in outputBuffers) { buffer.close() }

                /*
                 * MobileBERT labels: negative & positive
                 * AverageWordVec: 0 & 1
                 */
                _percentages.tryEmit(Pair(outputFloatArray, inferenceTime))
            } catch (e: Exception) {
                Log.e(TAG, "Error during classification: ${e.message}")
                _error.emit(Throwable("Error running inference: ${e.message}"))
            }
        }
    }

    /** Load metadata from model. */
    private fun loadModelMetadata(litertBuffer: ByteBuffer) {
        val metadataExtractor = MetadataExtractor(litertBuffer)
        if (metadataExtractor.hasMetadata()) {
            val vocalBuffer = metadataExtractor.getAssociatedFile("vocab.txt")
            vocabularyMap.clear()
            vocabularyMap.putAll(getVocabulary(vocalBuffer))
            Log.i(TAG, "Successfully loaded model metadata")
        }
    }

    /** Tokenize the text from String to int[], based on the index of words in `vocab.txt`. */
    private fun tokenizeText(inputText: String): List<Int> {
        return try {
            val ids = tokenizeWithVocabulary(inputText, vocabularyMap)
            Log.i(TAG, "tokenizeText: $ids")
            ids
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read vocabulary.txt: ${e.message}")
            emptyList()
        }
    }

    /** Retrieve vocabularies from "vocab.txt" file metadata. */
    private fun getVocabulary(inputStream: InputStream): Map<String, Int> {
        val reader = BufferedReader(InputStreamReader(inputStream))

        val map = mutableMapOf<String, Int>()
        var index = 0
        var line = ""
        while (reader.readLine().also { if (it != null) line = it } != null) {
            map[line] = index
            index++
        }

        reader.close()
        Log.d(TAG, "loadVocabulary: ${map.size}")
        return map
    }

    companion object {
        private const val TAG = "TextClassifier"

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
                AcceleratorEnum.NPU -> Accelerator.NPU
            }
        }

        fun expectedTokenCount(model: Model): Int {
            return if (model == Model.MobileBERT) 128 else 256
        }

        /** Remove punctuation to reduce unnecessary inputs. */
        fun removePunctuation(text: String): String {
            return text.replace("[^a-zA-Z0-9 ]".toRegex(), "")
        }

        fun tokenizeWithVocabulary(inputText: String, vocabulary: Map<String, Int>): List<Int> {
            val clean = removePunctuation(inputText)
            val words = clean.split(" ")
            return words.map { word -> vocabulary[word] ?: 0 }
        }
    }

    enum class AcceleratorEnum {
        CPU,
        GPU,
        NPU,
    }

    enum class Model(val fileName: String) {
        MobileBERT("mobile_bert.tflite"),
        AverageWordVec("word_vec.tflite"),
    }
}