#!/bin/bash

#SBATCH --comment="SEEUON paper dataset auto labeling"
#SBATCH --job-name=ultralytics
#SBATCH --partition=all
#SBATCH --nodelist=cubox05,cubox06,cubox07,cubox14,cubox15
#SBATCH --cpus-per-task=112
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:8
#SBATCH -o logs/%A.txt
#SBATCH --chdir=/purestorage/AILAB/AI_1/syshin/repository/ultralytics-syshin

master_node_hostname="cubox05"
master_node_address="172.100.100.24"

source .venv/bin/activate

srun \
--gpus=$SLURM_GPUS_PER_TASK \
bash script/ray_sbatch_inner.sh ${master_node_hostname} ${master_node_address}