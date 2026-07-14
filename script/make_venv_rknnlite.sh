#!/bin/bash

cd "$(dirname "$0")/.." || exit 1
[ "$(uname -m)" = "aarch64" ] || { echo "이 스크립트는 aarch64 전용입니다."; exit 1; }

MIN_RT_VER=2.3.2  # 모델 변환에 쓴 rknn-toolkit2 버전, librknnrt는 이 이상이어야 함
RT_LIB=/usr/lib/librknnrt.so
if [ -f "$RT_LIB" ]; then
    RT_VER=$(grep -aoE 'librknnrt version: [0-9]+\.[0-9]+\.[0-9]+' "$RT_LIB" | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)
    echo "    librknnrt version: ${RT_VER:-unknown}"
    if [ -n "$RT_VER" ] && [ "$(printf '%s\n' "$MIN_RT_VER" "$RT_VER" | sort -V | head -1)" != "$MIN_RT_VER" ]; then
        echo "ERROR: librknnrt $RT_VER < $MIN_RT_VER 입니다."
        echo "       https://github.com/airockchip/rknn-toolkit2 의"
        echo "       rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so 로 교체하세요."
        exit 1
    fi
else
    echo "ERROR: $RT_LIB 가 없습니다. RKNPU2 런타임을 먼저 설치하세요."
    exit 1
fi

uv venv isolated-rknnlite --python 3.11
uv pip install --python isolated-rknnlite/bin/python -e . rknn-toolkit-lite2
