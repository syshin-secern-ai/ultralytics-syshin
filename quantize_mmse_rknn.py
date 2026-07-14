"""
기존 rknn_raw int8 export 산출물(onnx, dataset.txt, metadata.yaml)을 재사용해 quantized_algorithm=mmse로 재양자화한다.
기존 normal 빌드는 보존되고, 결과는 weights/best_int8_mmse_rknn_raw_model/에 생성된다.
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from rknn.api import RKNN  # noqa: F401
except ImportError:
    sys.exit("rknn-toolkit2를 찾을 수 없습니다. isolated-rknn/bin/python으로 실행하세요.")

from ultralytics.utils import YAML
from ultralytics.utils.export.rknn import onnx2rknn


def main(run: Path) -> None:
    weights = run / "weights"
    onnx_file = weights / "best_rknn_raw.onnx"
    src_dir = weights / "best_int8_rknn_raw_model"
    dst_dir = weights / "best_int8_mmse_rknn_raw_model"

    for f in (onnx_file, src_dir / "dataset.txt", src_dir / "metadata.yaml"):
        if not f.exists():
            sys.exit(f"필수 파일이 없습니다: {f} (format=rknn_raw quantize=8 export 산출물 필요)")

    # 기존 normal 빌드의 .rknn 파일명에서 target platform 파싱 (예: best_rknn_raw-rk3576.rknn)
    rknn_file = next(src_dir.glob("*.rknn"), None)
    if rknn_file is None:
        sys.exit(f"{src_dir}에 .rknn 파일이 없습니다.")
    platform = rknn_file.stem.rsplit("-", 1)[-1]

    dataset_txt = src_dir / "dataset.txt"
    missing = [p for p in dataset_txt.read_text().splitlines() if p and not Path(p).is_file()]
    if missing:
        sys.exit(
            f"calibration 이미지 {len(missing)}장이 존재하지 않습니다 (예: {missing[0]}).\n"
            "sbatch가 클러스터 경로로 생성한 dataset.txt라면 이 머신의 경로로 다시 export하세요."
        )

    metadata = YAML.load(src_dir / "metadata.yaml")
    print(f"{run.name}: {platform} mmse 양자화 시작")
    t0 = time.time()
    out = onnx2rknn(
        onnx_file=str(onnx_file),
        output_dir=dst_dir,
        name=platform,
        quantize=8,
        dataset=dataset_txt,
        metadata=metadata,
        batch=metadata.get("batch", 1),
        quantized_algorithm="mmse",
        prefix="RKNN mmse:",
    )
    print(f"완료: {out} ({(time.time() - t0) / 60:.1f}분)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    main(args.run)
