#!/bin/bash
# 메인 컴퓨터에서 뷰어를 띄운 뒤 보드의 predict_rknn.py를 원격 실행하는 래퍼.
# 사용: script/run_rknn_board.sh --model <*_rknn_model | *_rknn_raw_model>
# 보드 주소/경로가 다르면 BOARD, BOARD_REPO 환경변수로 재정의한다.

cd "$(dirname "$0")/.." || exit 1
BOARD=${BOARD:-firefly@172.16.152.22}
BOARD_REPO=${BOARD_REPO:-syshin/repository/ultralytics-syshin}

.venv/bin/python viewer.py &
VIEWER_PID=$!
trap 'kill $VIEWER_PID 2>/dev/null' EXIT
sleep 2

ssh "$BOARD" "cd $BOARD_REPO && isolated-rknnlite/bin/python predict_rknn.py $*"
