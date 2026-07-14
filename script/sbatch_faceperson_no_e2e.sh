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

NAME=$(TZ=Asia/Seoul date +%y%m%d)_faceperson_no_e2e
DATA=../../data/wider-coco-ir-home-face-person/data.yaml

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
    model=yolo26n-pose-no-e2e.yaml \
    pretrained=yolo26n-pose.pt \
    data=${DATA} \
    epochs=550 \
    batch=128 \
    device="${CUDA_VISIBLE_DEVICES}" \
    name="${NAME}" \
    optimizer=MuSGD \
    scale=0.5 \
    mixup=0.0

# RKNN exports run first in the isolated-rknn env: they regenerate best.onnx (opset 19),
# which would overwrite the plain ONNX export. Roll back to .venv when done.
source isolated-rknn/bin/activate

# batch=2 RKNN exports run first: they write the same output directories as the
# batch=1 exports below, so stash them in weights/batch2/ before batch=1 regenerates them.
# rk3576 and rk3566 share those directories as best-<platform>.rknn, so both targets coexist.
yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn_raw \
    name=rk3576 \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA} \
    batch=2

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn_raw \
    name=rk3566 \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA} \
    batch=2

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3576 \
    quantize=16 \
    batch=2

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3566 \
    quantize=16 \
    batch=2

mkdir -p runs/pose/${NAME}/weights/batch2
mv runs/pose/${NAME}/weights/best_int8_rknn_raw_model \
    runs/pose/${NAME}/weights/best_rknn_model \
    runs/pose/${NAME}/weights/batch2/

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn_raw \
    name=rk3576 \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA}

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn_raw \
    name=rk3566 \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA}

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3576 \
    quantize=16

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=rknn \
    name=rk3566 \
    quantize=16

source .venv/bin/activate

# batch=2 exports run first: they write the same output filenames as the batch=1
# exports below, so stash them in weights/batch2/ before batch=1 regenerates them.
yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA} \
    device=cuda \
    batch=2

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    device=cuda \
    batch=2

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA} \
    batch=2

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino \
    batch=2

mkdir -p runs/pose/${NAME}/weights/batch2
mv runs/pose/${NAME}/weights/best.onnx \
    runs/pose/${NAME}/weights/best_int8.onnx \
    runs/pose/${NAME}/weights/best_openvino_model \
    runs/pose/${NAME}/weights/best_int8_openvino_model \
    runs/pose/${NAME}/weights/batch2/

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA} \
    device=cuda

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=onnx \
    device=cuda

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino \
    quantize=8 \
    fraction=0.01706 \
    data=${DATA}

yolo pose export \
    model=runs/pose/${NAME}/weights/best.pt \
    format=openvino

python val.py \
    --weights runs/pose/${NAME}/weights \
    --task pose \
    --data ${DATA}
