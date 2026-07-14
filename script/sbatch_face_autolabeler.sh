#!/bin/bash

#SBATCH --job-name=ultralytics
#SBATCH --partition=h100
#SBATCH --cpus-per-task=224
#SBATCH --gpus=8
#SBATCH -o logs/%A.txt
#SBATCH --chdir=/purestorage/ailab/syshin/repository/ultralytics-syshin

unset RANK
unset LOCAL_RANK
unset WORLD_SIZE

set -euo pipefail

source .venv/bin/activate

# ultralytics 설정
export YOLO_CONFIG_DIR=${PWD}/.ultralytics
mkdir -p "${YOLO_CONFIG_DIR}"
export WANDB_API_KEY=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
yolo settings reset
yolo settings wandb=True

EPOCHS=${EPOCHS:-250}
NAME=$(TZ=Asia/Seoul date +%y%m%d)_face_autolabeler_ep${EPOCHS}
DATA=${DATA:-../../data/widerface/data.yaml}

if [ -d "runs/pose/${NAME}" ]; then
    echo "Error: runs/pose/${NAME} already exists. Aborting." >&2
    exit 1
fi

# Model scale -> (scale, mixup)
#   n: (0.5, 0.0)
#   s: (0.9, 0.05)
#   m: (0.9, 0.15)
#   l: (0.9, 0.15)
#   x: (0.9, 0.2)
yolo pose train \
    model=yolo26x-pose-no-e2e.yaml \
    pretrained=yolo26x-pose.pt \
    data=${DATA} \
    epochs=${EPOCHS} \
    batch=128 \
    device="${CUDA_VISIBLE_DEVICES}" \
    name="${NAME}" \
    optimizer=MuSGD \
    scale=0.9 \
    mixup=0.2

# wandb 로컬 로그 삭제 (이미 서버에 동기화됨)
rm -rf runs/pose/${NAME}/wandb

# RKNN export runs first in the isolated-rknn env: it regenerates best.onnx (opset 19),
# which would overwrite the plain ONNX export. Roll back to .venv when done.
source isolated-rknn/bin/activate

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3576 \
    quantize=16

source .venv/bin/activate

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    device=cuda

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino

python val.py \
    --weights runs/pose/${NAME}/weights \
    --task pose \
    --data ${DATA}
