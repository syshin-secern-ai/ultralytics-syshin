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

NAME=$(TZ=Asia/Seoul date +%y%m%d)_yunet26n_imgsz320
DATA=../../data/widerface/data.yaml

if [ -d "runs/pose/${NAME}" ]; then
    echo "Error: runs/pose/${NAME} already exists. Aborting." >&2
    exit 1
fi

yolo pose train \
    model=yunet26n.yaml \
    data=${DATA} \
    epochs=200 \
    batch=128 \
    device="${CUDA_VISIBLE_DEVICES}" \
    name="${NAME}" \
    optimizer=MuSGD \
    amp=False \
    imgsz=320

# RKNN exports run first in the isolated-rknn env: they regenerate best.onnx (opset 19),
# which would overwrite the plain ONNX export. Roll back to .venv when done.
source isolated-rknn/bin/activate

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn_raw \
    name=rk3576 \
    quantize=8 \
    fraction=0.093 \
    data=${DATA} \
    imgsz=320

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3576 \
    quantize=16 \
    imgsz=320

source .venv/bin/activate

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    quantize=8 \
    fraction=0.093 \
    data=${DATA} \
    device=cuda \
    imgsz=320

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    device=cuda \
    imgsz=320

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino \
    quantize=8 \
    fraction=0.093 \
    data=${DATA} \
    imgsz=320

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino \
    imgsz=320

python val.py \
    --weights runs/pose/${NAME}/weights \
    --task pose \
    --data ${DATA}

python eval_widerface.py \
    --model runs/pose/${NAME}/weights/best.pt \
    --task pose \
    --save_path runs/pose/${NAME}/widerface.txt
