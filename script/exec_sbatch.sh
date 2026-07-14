#!/bin/bash

# sbatch는 로그인 노드(172.100.100.39)에서만 제출 가능하므로 ssh로 우회 제출한다.
# 사용법: bash script/exec_sbatch.sh script/sbatch_yunet26n.sh [job 스크립트 인자...]

set -euo pipefail

SUBMIT_HOST=syshin@172.100.100.39

if [ $# -lt 1 ]; then
    echo "usage: bash script/exec_sbatch.sh <sbatch_script> [args...]" >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: '$1' file not found." >&2
    exit 1
fi

# 로컬 checkout과 원격 리포 경로가 다르므로 리포 루트 기준 상대경로를 원격 리포 경로에 붙인다.
REMOTE_DIR=/purestorage/AILAB/AI_1/syshin/repository/ultralytics-syshin
SCRIPT=$REMOTE_DIR/$(realpath --relative-to="$(git rev-parse --show-toplevel)" "$1")
shift

# DATA 환경변수가 있으면 sbatch job까지 전달 (예: DATA=../../data/widerface/data.yaml)
ssh "$SUBMIT_HOST" "sbatch ${DATA:+--export=ALL,DATA=$DATA} '$SCRIPT' $*"
