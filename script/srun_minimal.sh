#!/bin/bash

srun \
--job-name=ultralytics \
--partition=all \
--nodelist=nv180 \
--cpus-per-task=16 \
--mem-per-cpu=2G \
--gpus=1 \
--chdir=/purestorage/ailab/syshin/repository/ultralytics-syshin \
--pty \
bash -c "
unset RANK;
unset LOCAL_RANK;
unset WORLD_SIZE;
source .venv/bin/activate;
bash
"