#!/bin/bash

cd "$(dirname "$0")/.." || exit 1
BOARD=${BOARD:-firefly@172.16.152.22}
BOARD_REPO=${BOARD_REPO:-syshin/repository/ultralytics-syshin}

echo "보드($BOARD)의 저장소를 git pull -r로 갱신합니다..."
ssh "$BOARD" "cd $BOARD_REPO && git pull -r" || exit 1

echo "보드($BOARD)의 기존 runs 폴더를 삭제합니다..."
ssh "$BOARD" "rm -rf $BOARD_REPO/runs" || exit 1

echo "runs를 보드($BOARD)로 복사합니다..."
scp -r runs "$BOARD:$BOARD_REPO/"
