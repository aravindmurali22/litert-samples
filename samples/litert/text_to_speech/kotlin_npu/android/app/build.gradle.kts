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

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.compose.compiler)
}

// Optional: `-Pnpu.libs=/path/to/qairt_arm64_libs` bundles QNN HTP .so files into the APK.
val npuLibs = providers.gradleProperty("npu.libs").orElse(providers.systemProperty("npu.libs"))
val stagedNpuJniLibs = layout.buildDirectory.dir("generated/npuJniLibs")

val stageNpuLibs =
  tasks.register<Sync>("stageNpuLibs") {
    into(stagedNpuJniLibs)
    npuLibs.orNull
      ?.takeIf { it.isNotBlank() }
      ?.let { dir ->
        val src = rootProject.file(dir)
        require(src.isDirectory) { "npu.libs is not a directory: $src" }
        from(src) {
          include("libQnn*.so")
          into("arm64-v8a")
        }
      }
  }

android {
  namespace = "com.google.ai.edge.examples.text_to_speech"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.text_to_speech"
    minSdk = 31
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    buildConfigField("boolean", "NPU_LIBS", "${npuLibs.isPresent}")
    ndk { abiFilters += setOf("arm64-v8a") }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  kotlinOptions {
    jvmTarget = "17"
    freeCompilerArgs += "-Xskip-metadata-version-check"
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }

  packaging {
    jniLibs {
      useLegacyPackaging = true
      pickFirsts +=
        setOf(
          "**/libc++_shared.so",
          "**/libtensorflowlite_jni.so",
          "**/libtensorflowlite_gpu_jni.so",
        )
    }
  }

  sourceSets.named("main") { jniLibs.srcDir(stagedNpuJniLibs) }

  androidResources { noCompress += listOf("bin", "json", "gz") }
}

tasks.named("preBuild") { dependsOn(stageNpuLibs) }

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material2)
  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.litert)
  implementation(libs.litert.gpu)
  implementation(libs.litert.npu.runtime.qualcomm)

  testImplementation(libs.junit)
  debugImplementation(libs.androidx.ui.tooling)
}
