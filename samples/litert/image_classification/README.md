# LiteRT Image Classification Samples

This directory contains Android image classification samples demonstrating how to use LiteRT (Google's new runtime for TensorFlow Lite) with the Compiled Model API. The samples classify real-time camera feeds into standard ImageNet categories.

## Overview

The Image Classification samples allow users to classify objects in real-time using optimized machine learning models. It demonstrates the use of the LiteRT Compiled Model API with support for hardware acceleration (CPU and GPU delegates).

## Available Implementations

### 1. kotlin_npu

An implementation utilizing the LiteRT Compiled Model API with support for **CPU**, **GPU**, and **NPU** acceleration (`BuiltinNpuAcceleratorProvider` + `CompiledModel.QualcommOptions`), allowing runtime switching across all three delegates in the settings bottom sheet.

### 2. kotlin_cpu_gpu

A standard implementation utilizing the Compiled Model API with support for CPU and GPU delegates.

**Features:**
-   **Real-time Inference**: Classifies objects from the camera feed or gallery images.
-   **Hardware Acceleration**: Switch between CPU, GPU, and NPU (`kotlin_npu`) delegates at runtime.
-   **Multiple Models**: Supports EfficientNet-Lite0 and EfficientNet-Lite2.
-   **Jetpack Compose**: Modern Android UI toolkit.

### Screenshots

| Object Classification | Object Classification |
| :---: | :---: |
| ![Cup Classification](./kotlin_cpu_gpu/android/output/cup.jpg) | ![Desk Classification](./kotlin_cpu_gpu/android/output/desk.jpg) |

## Technical Details

### Model Architecture
-   **Task**: Image Classification (ImageNet categories).
-   **Input**: 224x224x3 RGB image (preprocessed from camera stream or gallery bitmap).
-   **Output**: Probability scores for categories (loaded from metadata).
-   **Models**: 
    -   `efficientnet_lite0.tflite` (Default)
    -   `efficientnet_lite2.tflite`
-   **Format**: TensorFlow Lite (`.tflite`).

### Key Dependencies
-   **LiteRT** (`com.google.ai.edge.litert:litert`, `litert-gpu`, `litert-npu-runtime-qualcomm`)
-   **LiteRT Support & Metadata** (`com.google.ai.edge.litert:litert-support`, `litert-metadata`)
-   **Jetpack Compose** (UI)
-   **CameraX** (Camera feed)

### Architecture Components
-   **`ImageClassificationHelper`**: Handles model initialization, hardware delegate selection (CPU/GPU/NPU), and inference execution.
-   **`MainActivity`**: Setup of the main screen and UI components.
-   **`CameraScreen` / `GalleryScreen`**: Composables for live camera preview, gallery picking, and classification overlay.
-   **`MainViewModel`**: Manages UI state and communicates between the UI and the Helper.

## Getting Started

### Option 1: NPU + GPU + CPU (`kotlin_npu/android`)
1.  *(Optional for Qualcomm NPU hardware)* Populate `kotlin_npu/android/app/src/main/jniLibs/arm64-v8a` using `./fetch_qualcomm_library.sh` or pass `-Pnpu.libs=/path/to/qnn/libs` to Gradle.
2.  Open `kotlin_npu/android` in Android Studio (or run `./gradlew assembleDebug` / `./gradlew installDebug`).
3.  Switch between Camera and Gallery tabs and select **CPU**, **GPU**, or **NPU** under Delegate in the bottom sheet.

### Option 2: CPU + GPU (`kotlin_cpu_gpu/android`)
1.  Open `kotlin_cpu_gpu/android` in Android Studio.
2.  Build and run the application on an Android device.
3.  Point the camera at an object or pick a gallery image, and use the settings to switch between CPU and GPU acceleration or models.

