#!/bin/bash

set -euo pipefail

# 값을 확인하세요.
pytorch_version="2.13.0"
cuda_version="12.6"
cudnn_version="9"
workdir="/purestorage/AILAB/AI_1/syshin/repository/ultralytics-syshin"
wandb_host=""
wandb_key="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"
# -----------------------------------------------------------------------------

if [ -n "${wandb_host}" ]; then
    wandb_host="--host ${wandb_host}"
fi

if [ ! -f "pytorch+pytorch+${pytorch_version}-cuda${cuda_version}-cudnn${cudnn_version}-runtime.sqsh" ]; then
    enroot import docker://pytorch/pytorch:${pytorch_version}-cuda${cuda_version}-cudnn${cudnn_version}-runtime
fi
enroot create -n ultralytics pytorch+pytorch+${pytorch_version}-cuda${cuda_version}-cudnn${cudnn_version}-runtime.sqsh
enroot start --root --rw --mount /purestorage:/purestorage ultralytics bash -c "
cd ${workdir} &&
rm -rf /var/lib/apt/lists/* &&
sed -i 's|http://archive.ubuntu.com|http://kr.archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources &&
sed -i 's|http://security.ubuntu.com|http://kr.archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources &&
apt update &&
apt install build-essential curl nano pkg-config wget -y &&
apt install libgl1 libglib2.0-0t64 -y &&
pip config set global.no-cache-dir false &&
pip install -e . --break-system-packages &&
pip install nncf onnx onnxruntime-gpu==1.26.0 onnxslim openvino ruff scipy tqdm wandb --break-system-packages &&
pip install -U "ray[default]" --break-system-packages &&
wandb login ${wandb_host} ${wandb_key} &&
yolo settings reset &&
yolo settings wandb=True
"
enroot export -f ultralytics
enroot remove -f ultralytics