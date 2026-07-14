from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from widerface_evaluation.evaluation import PredictionDict, evaluation


def main(model: Path, task: str, valset: Path, save_path: Path | None, device: str | None = None) -> None:
    val_img_dir = Path(valset).resolve(strict=True)
    eval_dir = Path("widerface_evaluation").resolve(strict=True)

    model = YOLO(model, task)
    results = model.predict(
        val_img_dir / "**/*.jpg",
        conf=0.001,
        iou=0.7,
        max_det=3000,
        device=device,
        stream=True,
        verbose=False,
    )
    predictions: PredictionDict = {}
    for result in tqdm(results, total=len(list(val_img_dir.rglob("**/*.jpg"))), dynamic_ncols=True):
        event_name, image_name = Path(result.path).parts[-2:]

        face_boxes = result.boxes[result.boxes.cls == 0]
        if len(face_boxes) == 0:
            boxes = np.zeros((0, 5))
        else:
            xyxy = face_boxes.xyxy.cpu().numpy().astype(np.float64)
            xywh = face_boxes.xywh.cpu().numpy().astype(np.float64)
            conf = face_boxes.conf.clip(0, 1).cpu().numpy().astype(np.float64)
            boxes = np.stack((xyxy[:, 0], xyxy[:, 1], xywh[:, 2], xywh[:, 3], conf), axis=1)
        predictions.setdefault(event_name, {})[Path(image_name).stem] = boxes

    aps = evaluation(predictions, eval_dir / "ground_truth")
    mean_ap = sum(aps) / len(aps)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(f"Easy   Val AP: {aps[0]:.6f} ({aps[0] * 100:.2f} %)\n")
            f.write(f"Medium Val AP: {aps[1]:.6f} ({aps[1] * 100:.2f} %)\n")
            f.write(f"Hard   Val AP: {aps[2]:.6f} ({aps[2] * 100:.2f} %)\n")
            f.write(f"Mean   Val AP: {mean_ap:.6f} ({mean_ap * 100:.2f} %)\n")

    print("------------ Results ------------")
    print(f"Easy   Val AP: {aps[0]:.6f} ({aps[0] * 100:.2f} %)")
    print(f"Medium Val AP: {aps[1]:.6f} ({aps[1] * 100:.2f} %)")
    print(f"Hard   Val AP: {aps[2]:.6f} ({aps[2] * 100:.2f} %)")
    print(f"Mean   Val AP: {mean_ap:.6f} ({mean_ap * 100:.2f} %)")
    print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--valset", type=Path, default="../../data/WIDER_val")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_path", type=Path)
    args = parser.parse_args()

    main(args.model, args.task, args.valset, args.save_path, args.device)
