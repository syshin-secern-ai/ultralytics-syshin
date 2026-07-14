import argparse
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

VARIANTS = [
    "best.pt",
    "best.onnx",
    "best_openvino_model",
    "best_int8.onnx",
    "best_int8_openvino_model",
]


def main(weights: Path, task: str, data: Path) -> None:
    if weights.is_file():  # e.g. .../weights/best.pt passed directly
        weights = weights.parent
    run_dir = weights.parent  # e.g. runs/pose/260714_yunet26n

    results = {}
    fused_info = ""
    tmp = Path(tempfile.mkdtemp())
    try:
        for name in VARIANTS:
            model_path = weights / name
            if not model_path.exists():
                print(f"WARNING: {model_path} not found, skipping")
                continue
            model = YOLO(model_path, task)
            metrics = model.val(data=data, plots=False, project=tmp, name=name)
            results[name] = metrics.results_dict
            if name == "best.pt":
                imgsz = model.overrides.get("imgsz", 640)
                fused = model.model.fuse(verbose=False)
                fused_info = f"best.pt fused @ imgsz={imgsz}: {get_num_params(fused):,} params, {get_flops(fused, imgsz):.2f} GFLOPs\n"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not results:
        print(f"ERROR: no models found in {weights}")
        return

    keys = list(next(iter(results.values())))
    base = results.get("best.pt", {})
    cells = {}
    for name, rd in results.items():
        cells[name] = {}
        for k in keys:
            v = rd.get(k, float("nan"))
            s = f"{v:.6f}"
            if name != "best.pt" and k in base:
                s += f" ({v - base[k]:+.4f})"
            cells[name][k] = s

    kw = max(len(k) for k in keys)
    cols = {name: max(len(name), *(len(cells[name][k]) for k in keys)) for name in results}
    lines = [f"{'metric':<{kw}}  " + "  ".join(f"{name:>{w}}" for name, w in cols.items())]
    for k in keys:
        lines.append(f"{k:<{kw}}  " + "  ".join(f"{cells[name][k]:>{w}}" for name, w in cols.items()))
    table = fused_info + "\n".join(lines) + "\n"

    metrics_file = run_dir / "metrics.txt"
    metrics_file.write_text(table)
    print(f"\n{table}\nSaved to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="weights dir")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    main(args.weights, args.task, args.data)
