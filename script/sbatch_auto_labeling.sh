#!/bin/bash

#SBATCH --job-name=auto_labeling
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH -o logs/%A.txt
#SBATCH --chdir=/purestorage/ailab/syshin/repository/ultralytics-syshin

set -euo pipefail

source .venv/bin/activate

python dataset_code/utils/auto_labeling.py \
    --root "${ROOT:-../../data/open-images-face}" \
    --model_path "${MODEL:-runs/pose/261123_face_autolabeler_ep350/weights/best.pt}"
