#!/bin/bash

cd "$(dirname "$0")/.." || exit 1
[ "$(uname -m)" = "x86_64" ] || { echo "이 스크립트는 x86_64 전용입니다."; exit 1; }
export ULTRALYTICS_ISOLATED_VENVS=$PWD
.venv/bin/python .github/scripts/create-export-env.py --env isolated-rknn
