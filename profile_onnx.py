import argparse
import os
import tempfile
from pathlib import Path

import onnxruntime as ort


def check_cuda_placement(onnx_path: Path) -> None:
    """CUDA EP로 1회 로드해 CPU로 fallback되는 노드가 없는지 확인.

    onnxruntime은 노드 배치 결과를 파이썬 API로 노출하지 않으므로,
    세션 생성 시 C 레벨 stderr로 나오는 verbose 배치 로그를 캡처해 판정한다.
    """
    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 0
    so.log_verbosity_level = 1  # "Node placements" 로그 활성화
    with tempfile.TemporaryFile() as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            sess = ort.InferenceSession(str(onnx_path), so, providers=["CUDAExecutionProvider"])
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        log = tmp.read().decode(errors="replace")
    if "CUDAExecutionProvider" not in sess.get_providers():
        print("CUDA placement: CUDA EP 로드 실패! (전부 CPU 실행)")
        return
    placements = [
        line.split("VerifyEachNodeIsAssignedToAnEp]")[-1].strip()
        for line in log.splitlines()
        if "VerifyEachNodeIsAssignedToAnEp" in line and "ExecutionProvider" in line
    ]
    if any("CPUExecutionProvider" in p for p in placements):
        print("CUDA placement: 일부 노드가 CPU로 fallback!")
    else:
        print("CUDA placement: 모든 노드가 CUDAExecutionProvider에 배치됨")
    for p in placements:
        print(" ", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_file", type=Path)
    args = parser.parse_args()

    check_cuda_placement(args.onnx_file)
