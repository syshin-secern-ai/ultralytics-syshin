#!/bin/bash
# 메인 컴퓨터에서 뷰어를 띄운 뒤 보드의 pose_rknn_stream(C++)을 빌드·원격 실행하는 래퍼.
# 사용: script/run_rknn_board_cpp.sh <model.rknn> [host] [cam_port] [viewer_port] [conf] [iou]
# 보드 주소/경로가 다르면 BOARD, BOARD_REPO 환경변수로 재정의한다.

cd "$(dirname "$0")/.." || exit 1
BOARD=${BOARD:-firefly@172.16.152.22}
BOARD_REPO=${BOARD_REPO:-syshin/repository/ultralytics-syshin}

# -p로 mtime을 보존해 소스가 안 바뀌었으면 보드에서 재빌드를 건너뛴다.
scp -p cpp/pose_rknn_stream.cpp cpp/rknn_pose.hpp "$BOARD:$BOARD_REPO/cpp/" || exit 1

.venv/bin/python viewer.py &
VIEWER_PID=$!
trap 'kill $VIEWER_PID 2>/dev/null' EXIT
sleep 2

ssh "$BOARD" "cd $BOARD_REPO && { [ cpp/pose_rknn_stream -nt cpp/pose_rknn_stream.cpp ] && [ cpp/pose_rknn_stream -nt cpp/rknn_pose.hpp ] || { echo '소스가 갱신되어 보드에서 pose_rknn_stream을 빌드합니다...'; g++ -O2 -std=c++17 cpp/pose_rknn_stream.cpp -o cpp/pose_rknn_stream \$(pkg-config --cflags --libs opencv4) -lrknnrt -lpthread; }; } && cpp/pose_rknn_stream $*"
