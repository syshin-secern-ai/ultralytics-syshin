#!/bin/bash

#SBATCH --job-name=ultralytics
#SBATCH --partition=hopper
##SBATCH --nodelist=cubox01
#SBATCH --exclusive
#SBATCH --gpus=8
#SBATCH -o logs/%A.txt
#SBATCH --chdir=/purestorage/AILAB/AI_1/syshin/repository/ultralytics-syshin

unset RANK
unset LOCAL_RANK
unset WORLD_SIZE

set -euo pipefail

source .venv/bin/activate

# ultralytics 설정
export YOLO_CONFIG_DIR=${PWD}/.ultralytics
export WANDB_API_KEY=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
yolo settings reset
yolo settings wandb=True

NAME=$(TZ=Asia/Seoul date +%y%m%d)_idcard
DATA=../../data/idcard-passport-seg/data.yaml

if [ -d "runs/segment/${NAME}" ]; then
    echo "Error: runs/segment/${NAME} already exists. Aborting." >&2
    exit 1
fi

# Model scale -> (scale, mixup, copy_paste)
#   n: (0.5, 0.0, 0.1)
#   s: (0.9, 0.05, 0.15)
#   m: (0.9, 0.15, 0.4)
#   l: (0.9, 0.15, 0.5)
#   x: (0.9, 0.2, 0.6)
yolo segment train \
    model=yolo26n-seg.pt \
    data=${DATA} \
    epochs=300 \
    batch=128 \
    device="${CUDA_VISIBLE_DEVICES}" \
    name="${NAME}" \
    optimizer=MuSGD \
    mask_ratio=1 \
    perspective=0.0001 \
    scale=0.5 \
    mixup=0.0 \
    copy_paste=0.1

# RKNN exports run first in the isolated-rknn env: they regenerate best.onnx (opset 19),
# which would overwrite the plain ONNX export. Roll back to .venv when done.
source isolated-rknn/bin/activate

yolo segment export \
    model=runs/segment/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3576 \
    quantize=16

source .venv/bin/activate

yolo segment export \
    model=runs/segment/${NAME}/weights/best.pt \
    format=onnx \
    quantize=8 \
    fraction=0.388 \
    data=${DATA} \
    device=cuda

yolo segment export \
    model=runs/segment/${NAME}/weights/best.pt \
    format=onnx \
    device=cuda

yolo segment export \
    model=runs/segment/${NAME}/weights/best.pt \
    format=openvino \
    quantize=8 \
    fraction=0.388 \
    data=${DATA}

yolo segment export \
    model=runs/segment/${NAME}/weights/best.pt \
    format=openvino

python val.py \
    --weights runs/segment/${NAME}/weights \
    --task segment \
    --data ${DATA}
