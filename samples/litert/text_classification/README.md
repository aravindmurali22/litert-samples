# LiteRT Text Classification Android Sample

This directory contains the Android Text Classification sample using the LiteRT `CompiledModel` API with MobileBERT (`mobile_bert.tflite`) and AverageWordVec (`word_vec.tflite`):

- [`kotlin_cpu/android`](kotlin_cpu/android) — CPU variant.
- [`kotlin_npu/android`](kotlin_npu/android) — CPU, GPU, and Qualcomm QNN HTP **NPU** (`Accelerator.NPU`) variant.

## NPU Mode (`kotlin_npu/android`)

The `kotlin_npu/android` project adds `Accelerator.NPU` (via `BuiltinNpuAcceleratorProvider` and `com.google.ai.edge.litert:litert-npu-runtime-qualcomm:2.3.0`) and `Accelerator.GPU` (`com.google.ai.edge.litert:litert-gpu:2.3.0`) alongside `Accelerator.CPU`, with a runtime `CPU` / `GPU` / `NPU` selector chip bar.

> **Model note:** `MobileBERT` uses standard transformer operators that can be delegated across CPU, GPU, and NPU. `AverageWordVec` uses custom string/lookup operators that may fall back to or require CPU execution.

### Building and Running `kotlin_npu/android`

1. **(Optional, for Qualcomm NPU hardware)** Fetch the Qualcomm QNN HTP runtime libraries into `app/src/main/jniLibs/arm64-v8a`:
   ```bash
   cd kotlin_npu/android
   ./fetch_qualcomm_library.sh
   ```
   Or pass an existing directory of Qualcomm QNN `.so` files via `-Pnpu.libs=/path/to/qnn/arm64-v8a`.

2. **Build and run unit tests**:
   ```bash
   cd kotlin_npu/android
   ./gradlew testDebugUnitTest assembleDebug
   ```
