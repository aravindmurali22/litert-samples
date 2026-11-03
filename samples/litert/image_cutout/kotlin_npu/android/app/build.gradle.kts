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
  alias(libs.plugins.kotlin.android)
}

// The directory with the model files (install_to_device.sh's argument): the tokenizer tests read
// its tokenizer.json
// and are skipped without it; the other tests always run.
val wevisoneModels =
  providers.gradleProperty("wevisone.models").orElse(providers.systemProperty("wevisone.models"))

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
  namespace = "com.google.ai.edge.examples.image_cutout"
  compileSdk = 35
  buildToolsVersion = "35.0.0"

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.image_cutout"
    minSdk = 31
    targetSdk = 35
    versionCode = 1
    versionName = "0.1"
    buildConfigField("String", "LITERT_VERSION", "\"${libs.versions.litert.get()}\"")
    buildConfigField("boolean", "NPU_LIBS", "${npuLibs.isPresent}")
    ndk {
      abiFilters += "arm64-v8a"
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      // sample convenience: installable without a release keystore
      signingConfig = signingConfigs.getByName("debug")
    }
  }

  packaging {
    // The GPU and NPU accelerator libraries are loaded from the extracted native library
    // directory.
    jniLibs { useLegacyPackaging = true }
  }

  sourceSets.named("main") { jniLibs.srcDir(stagedNpuJniLibs) }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  buildFeatures {
    buildConfig = true
  }
  testOptions {
    unitTests.all {
      wevisoneModels.orNull
        ?.takeIf { it.isNotBlank() }
        ?.let { directory ->
          it.systemProperty("wevisone.models", rootProject.file(directory).absolutePath)
        }
      it.systemProperty("wevisone.moduleRoot", rootProject.projectDir.absolutePath)
      it.maxHeapSize = "3g"
      it.testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
      }
    }
  }
}

tasks.named("preBuild") { dependsOn(stageNpuLibs) }

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    // LiteRT 2.3.0's Kotlin API carries Kotlin 2.4 metadata; Google's own 2.3.0 samples on Kotlin
    // 2.2.21 pass this
    freeCompilerArgs.add("-Xskip-metadata-version-check")
  }
}

dependencies {
  // LiteRT CompiledModel (CPU / XNNPACK), GPU accelerator (ML Drift, OpenCL), and Qualcomm NPU
  implementation(libs.litert)
  implementation(libs.litert.gpu)
  implementation(libs.litert.npu.runtime.qualcomm)
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.activity.ktx)
  // Streaming reader for the 11.6 MB tokenizer.json, the same class on the phone and in the JVM
  // tests
  implementation(libs.gson)
  testImplementation(libs.junit)
  // the real org.json on the JVM (the Android jar's copy is a stub in unit tests)
  testImplementation(libs.org.json)
}
