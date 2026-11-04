#!/usr/bin/env bash
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Downloads Qualcomm AI Engine Direct (QNN) SDK v2.50.0.260828 and copies the HTP NPU
# runtime shared libraries into app/src/main/jniLibs/arm64-v8a/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

QAIRT_VERSION="2.50.0.260828"
QAIRT_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VERSION}/v${QAIRT_VERSION}.zip"
QAIRT_ZIP="${TMP_DIR}/qairt.zip"

echo "Downloading Qualcomm AI Runtime SDK v${QAIRT_VERSION}..."
curl -L "${QAIRT_URL}" -o "${QAIRT_ZIP}"

echo "Extracting SDK..."
unzip -q "${QAIRT_ZIP}" -d "${TMP_DIR}"

QNN_SDK_ROOT="$(find "${TMP_DIR}" -maxdepth 2 -type d -name "qairt*" | head -n 1)/${QAIRT_VERSION}"
if [[ ! -d "${QNN_SDK_ROOT}" ]]; then
  QNN_SDK_ROOT="$(find "${TMP_DIR}" -maxdepth 2 -type d -name "2.50.0*" | head -n 1)"
fi

DEST="${SCRIPT_DIR}/app/src/main/jniLibs/arm64-v8a"
mkdir -p "${DEST}"

cp -f "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so" "${DEST}/"
cp -f "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so" "${DEST}/"
cp -f "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${DEST}/"
cp -f "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnIr.so" "${DEST}/"
cp -f "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSaver.so" "${DEST}/"
cp -f "${QNN_SDK_ROOT}/lib/aarch64-android"/libQnnHtpV*Stub.so "${DEST}/"
cp -f "${QNN_SDK_ROOT}/lib/hexagon-v"*/unsigned/libQnnHtpV*Skel.so "${DEST}/"

echo "Qualcomm QNN HTP libraries staged in ${DEST}"
