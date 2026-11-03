#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a LiteRT model on ImageNet-1K validation split."""

import argparse
import os
import sys

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from PIL import Image

from datasets import load_dataset
import evaluate
from tqdm import tqdm

from imagenet_preprocessing import (
    infer_input_size,
    pick_preprocess_config,
    preprocess_image,
)
import main as imagenet_main


def _default_model_path() -> str:
  return os.path.join(os.getcwd(), "mobilenet_v2.tflite")


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
  flat = scores.reshape(-1)
  if k <= 1:
    return np.array([int(np.argmax(flat))], dtype=np.int64)
  if flat.size <= k:
    return np.argsort(flat)[::-1]
  idx = np.argpartition(flat, -k)[-k:]
  return idx[np.argsort(flat[idx])[::-1]]


def _parse_args(argv: list[str]):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default=_default_model_path())
  parser.add_argument(
      "--arch",
      default=None,
      help=(
          "Optional model architecture name to select preprocessing. "
          "If set, overrides inference based on --model filename."
      ),
  )
  parser.add_argument(
      "--max_samples",
      type=int,
      default=0,
      help="If set, stop after this many samples.",
  )
  return parser.parse_args(argv)


def _iter_dataset(dataset, max_samples: int):
  count = 0
  for item in dataset:
    yield item
    count += 1
    if max_samples and count >= max_samples:
      break


def main(argv: list[str]) -> int:
  args = _parse_args(argv)

  if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model not found: {args.model}")

  model = CompiledModel.from_file(args.model)
  signature_index = 0
  channels = 3

  input_height, input_width, channels_first = infer_input_size(
      model, signature_index
  )
  preprocess_model_key = args.arch if args.arch else args.model
  preprocess = pick_preprocess_config(
      preprocess_model_key, input_height, input_width
  )
  layout = "NCHW" if channels_first else "NHWC"
  print(
      "Model input:",
      f"{input_height}x{input_width}",
      layout,
      f"resize={preprocess['resize_size']}",
      f"crop={preprocess['crop_height']}x{preprocess['crop_width']}",
  )

  input_buffers = model.create_input_buffers(signature_index)
  output_buffers = model.create_output_buffers(signature_index)
  output_requirements = model.get_output_buffer_requirements(0, signature_index)

  output_dtype = imagenet_main._pick_output_dtype(output_requirements)
  buffer_size = output_requirements.get("buffer_size", 0)
  itemsize = np.dtype(output_dtype).itemsize
  output_size = buffer_size // itemsize if itemsize else buffer_size
  if output_size == 0:
    raise ValueError("Output buffer size is zero")
  output_offset = 1 if output_size == 1001 else 0

  dataset = load_dataset(
      "imagenet-1k",
      split="validation",
      token=True,
  )

  accuracy_metric = evaluate.load("accuracy")
  correct_top5 = 0
  total = 0

  total_hint = (
      min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
  )
  iterator = _iter_dataset(dataset, args.max_samples)
  for example in tqdm(iterator, total=total_hint, unit="img"):
    image = example["image"]
    label = int(example["label"]) + output_offset

    input_array = preprocess_image(
        image,
        channels,
        preprocess["resize_size"],
        preprocess["crop_height"],
        preprocess["crop_width"],
        preprocess["mean"],
        preprocess["std"],
        preprocess["resample"],
        channels_first,
    )

    input_buffers[0].write(input_array)
    model.run_by_index(signature_index, input_buffers, output_buffers)

    output_array = imagenet_main._read_output(output_buffers[0], output_requirements)
    scores = output_array.reshape(-1)
    pred = int(np.argmax(scores))
    top5 = _topk_indices(scores, 5)

    accuracy_metric.add_batch(predictions=[pred], references=[label])
    if label in top5:
      correct_top5 += 1
    total += 1

  results = accuracy_metric.compute()
  top1 = float(results.get("accuracy", 0.0))
  top5 = float(correct_top5 / total) if total else 0.0

  print(f"Samples: {total}")
  print(f"Top-1 accuracy: {top1:.6f}")
  print(f"Top-5 accuracy: {top5:.6f}")
  if output_offset:
    print("Note: model output size is 1001; labels were offset by +1.")

  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
