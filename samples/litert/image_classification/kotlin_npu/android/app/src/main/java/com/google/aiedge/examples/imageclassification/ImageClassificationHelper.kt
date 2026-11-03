/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.aiedge.examples.imageclassification

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.support.metadata.MetadataExtractor
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer

class ImageClassificationHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** The enum contains the model file name, relative to the assets/ directory */
        var model: Model = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations*/
        var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
        /** Number of output classes of the TFLite model.  */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        var probabilityThreshold: Float = DEFAULT_THRESHOLD,
        /** Number of threads to be used for ops that support multi-threading.
         * threadCount>= -1. Setting numThreads to 0 has the effect of disabling multithreading,
         * which is equivalent to setting numThreads to 1. If unspecified, or set to the value -1,
         * the number of threads used will be implementation-defined and platform-dependent.
         * */
        var threadCount: Int = DEFAULT_THREAD_COUNT
    )

    companion object {
        private const val TAG = "ImageClassification"

        val DEFAULT_MODEL = Model.EfficientNetLite0
        val DEFAULT_DELEGATE = AcceleratorEnum.CPU
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_THRESHOLD = 0.3f
        const val DEFAULT_THREAD_COUNT = 2

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
                AcceleratorEnum.NPU -> Accelerator.NPU
            }
        }

        fun topCategories(
            labels: List<String>,
            output: FloatArray,
            probabilityThreshold: Float,
            resultCount: Int,
        ): List<Category> {
            val outputList = output.map {
                if (it < probabilityThreshold) 0f else it
            }
            return labels.zip(outputList).map {
                Category(label = it.first, score = it.second)
            }.sortedByDescending { it.score }.take(resultCount)
        }
    }

    /** As the result of sound classification, this value emits map of probabilities */
    val classification: SharedFlow<ClassificationResult>
        get() = _classification
    private val _classification = MutableSharedFlow<ClassificationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var model: CompiledModel? = null
    private var environment: Environment? = null
    private lateinit var labels: List<String>
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    /** Init a CompiledModel from [Model] with [AcceleratorEnum]*/
    suspend fun initClassifier() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                // Load metadata first using a ByteBuffer
                val assetManager = context.assets
                val modelStream = assetManager.open(options.model.fileName)
                val modelBytes = modelStream.readBytes()
                val modelBuffer = ByteBuffer.wrap(modelBytes)
                labels = getModelMetadata(modelBuffer)
                modelStream.close()

                val appContext = context.applicationContext
                val accelerator = toAccelerator(options.delegate)
                val createdEnv: Environment = if (options.delegate == AcceleratorEnum.NPU) {
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
                    if (options.delegate == AcceleratorEnum.NPU) {
                        qualcommOptions = CompiledModel.QualcommOptions(
                            htpPerformanceMode =
                                CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
                        )
                    }
                }

                environment = createdEnv
                model = CompiledModel.create(
                    assetManager,
                    options.model.fileName,
                    modelOptions,
                    createdEnv
                )
                Log.i(TAG, "Done creating CompiledModel from ${options.model.fileName} on ${options.delegate}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Create CompiledModel from ${options.model.fileName} on ${options.delegate} failed: ${e.message}", e)
            _error.emit(e)
        }
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            runCatching { model?.close() }
            model = null
            runCatching { environment?.close() }
            environment = null
        }
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    suspend fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(singleThreadDispatcher) {
                if (model == null) return@withContext
                val startTime = SystemClock.uptimeMillis()

                val rotation = -rotationDegrees / 90
                val (h, w) = Pair(224, 224)

                // Manual preprocessing
                var image = scaleBitmap(bitmap, w, h)
                image = rot90Clockwise(image, rotation)
                val inputFloatArray = normalize(image, 127.5f, 127.5f)

                val output = classifyWithCompiledModel(inputFloatArray)
                val categories = topCategories(
                    labels = labels,
                    output = output,
                    probabilityThreshold = options.probabilityThreshold,
                    resultCount = options.resultCount,
                )

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _classification.emit(ClassificationResult(categories, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.i(TAG, "Image classification error occurred: ${e.message}")
            _error.emit(e)
        }
    }

    private fun scaleBitmap(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private fun rot90Clockwise(image: Bitmap, numRotation: Int): Bitmap {
        val effectiveRotation = numRotation % 4
        if (effectiveRotation == 0) return image
        val matrix = android.graphics.Matrix()
        val (w, h) = Pair(image.width, image.height)
        matrix.postRotate(-90f * effectiveRotation)
        return Bitmap.createBitmap(image, 0, 0, w, h, matrix, false)
    }

    private fun normalize(image: Bitmap, mean: Float, stddev: Float): FloatArray {
        val (width, height) = Pair(image.width, image.height)
        val numPixels = width * height
        val pixelsIntArray = IntArray(numPixels)
        val outputFloatArray = FloatArray(numPixels * 3)
        image.getPixels(pixelsIntArray, 0, width, 0, 0, width, height)
        for (i in 0 until numPixels) {
            val pixel = pixelsIntArray[i]
            val r = android.graphics.Color.red(pixel).toFloat()
            val g = android.graphics.Color.green(pixel).toFloat()
            val b = android.graphics.Color.blue(pixel).toFloat()
            val idx = i * 3
            outputFloatArray[idx] = (r - mean) / stddev
            outputFloatArray[idx + 1] = (g - mean) / stddev
            outputFloatArray[idx + 2] = (b - mean) / stddev
        }
        return outputFloatArray
    }

    private fun classifyWithCompiledModel(inputFloatArray: FloatArray): FloatArray {
        val currentModel = model ?: return FloatArray(0)
        val inputBuffers = currentModel.createInputBuffers()
        val outputBuffers = currentModel.createOutputBuffers()

        inputBuffers[0].writeFloat(inputFloatArray)
        currentModel.run(inputBuffers, outputBuffers)

        val output = outputBuffers[0].readFloat()

        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }

        return output
    }


    /** Load metadata from model*/
    private fun getModelMetadata(litertBuffer: ByteBuffer): List<String> {
        val metadataExtractor = MetadataExtractor(litertBuffer)
        val labels = mutableListOf<String>()
        if (metadataExtractor.hasMetadata()) {
            val inputStream = metadataExtractor.getAssociatedFile("labels_without_background.txt")
            labels.addAll(readFileInputStream(inputStream))
            Log.i(
                TAG, "Successfully loaded model metadata ${metadataExtractor.associatedFileNames}"
            )
        }
        return labels
    }

    /** Retrieve Map<String, Int> from metadata file */
    private fun readFileInputStream(inputStream: InputStream): List<String> {
        val reader = BufferedReader(InputStreamReader(inputStream))

        val list = mutableListOf<String>()
        var line = ""
        while (reader.readLine().also { if (it != null) line = it } != null) {
            list.add(line)
        }

        reader.close()
        return list
    }


    enum class AcceleratorEnum {
        CPU, GPU, NPU
    }

    enum class Model(val fileName: String) {
        EfficientNetLite0("efficientnet_lite0.tflite"), EfficientNetLite2("efficientnet_lite2.tflite")
    }

    data class ClassificationResult(
        val categories: List<Category>, val inferenceTime: Long
    )

    data class Category(val label: String, val score: Float)

}