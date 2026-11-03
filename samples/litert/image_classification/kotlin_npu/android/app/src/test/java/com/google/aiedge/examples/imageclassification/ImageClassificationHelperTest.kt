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

import com.google.ai.edge.litert.Accelerator
import com.google.aiedge.examples.imageclassification.ImageClassificationHelper.AcceleratorEnum
import org.junit.Assert.assertEquals
import org.junit.Test

class ImageClassificationHelperTest {
    @Test
    fun toAccelerator_mapsCpuGpuAndNpu() {
        assertEquals(Accelerator.CPU, ImageClassificationHelper.toAccelerator(AcceleratorEnum.CPU))
        assertEquals(Accelerator.GPU, ImageClassificationHelper.toAccelerator(AcceleratorEnum.GPU))
        assertEquals(Accelerator.NPU, ImageClassificationHelper.toAccelerator(AcceleratorEnum.NPU))
        assertEquals(3, AcceleratorEnum.entries.size)
    }

    @Test
    fun topCategories_appliesThresholdSortsDescendingAndLimitsCount() {
        val labels = listOf("tench", "goldfish", "white_shark", "tiger_shark")
        val scores = floatArrayOf(0.15f, 0.85f, 0.42f, 0.67f)
        val result = ImageClassificationHelper.topCategories(
            labels = labels,
            output = scores,
            probabilityThreshold = 0.3f,
            resultCount = 3,
        )

        assertEquals(3, result.size)
        assertEquals("goldfish", result[0].label)
        assertEquals(0.85f, result[0].score, 1e-6f)
        assertEquals("tiger_shark", result[1].label)
        assertEquals(0.67f, result[1].score, 1e-6f)
        assertEquals("white_shark", result[2].label)
        assertEquals(0.42f, result[2].score, 1e-6f)
    }
}
