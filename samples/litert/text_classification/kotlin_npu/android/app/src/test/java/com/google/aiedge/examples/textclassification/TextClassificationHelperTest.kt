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

package com.google.aiedge.examples.textclassification

import com.google.ai.edge.litert.Accelerator
import org.junit.Assert.assertEquals
import org.junit.Test

class TextClassificationHelperTest {
    @Test
    fun acceleratorEnum_mapsCpuGpuAndNpu() {
        assertEquals(
            listOf("CPU", "GPU", "NPU"),
            TextClassificationHelper.AcceleratorEnum.entries.map { it.name },
        )
        assertEquals(
            Accelerator.CPU,
            TextClassificationHelper.toAccelerator(TextClassificationHelper.AcceleratorEnum.CPU),
        )
        assertEquals(
            Accelerator.GPU,
            TextClassificationHelper.toAccelerator(TextClassificationHelper.AcceleratorEnum.GPU),
        )
        assertEquals(
            Accelerator.NPU,
            TextClassificationHelper.toAccelerator(TextClassificationHelper.AcceleratorEnum.NPU),
        )
    }

    @Test
    fun expectedTokenCount_matchesModelInputLengths() {
        assertEquals(
            128,
            TextClassificationHelper.expectedTokenCount(TextClassificationHelper.Model.MobileBERT),
        )
        assertEquals(
            256,
            TextClassificationHelper.expectedTokenCount(TextClassificationHelper.Model.AverageWordVec),
        )
    }

    @Test
    fun tokenizeWithVocabulary_removesPunctuationAndMapsUnknownToZero() {
        val vocab = mapOf("great" to 10, "movie" to 42)
        val tokens = TextClassificationHelper.tokenizeWithVocabulary("great, unknown movie!", vocab)
        assertEquals(listOf(10, 0, 42), tokens)
    }
}
