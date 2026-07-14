from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy import ndarray
from tqdm import tqdm

from ultralytics.utils.ops import clip_boxes, xywh2xyxy, xyxy2xywh


def make_crop_img(img: ndarray, boxes: ndarray, pad_ratio: float, xywh: bool = False) -> tuple[list[ndarray], ndarray]:
    """
    Boxes shape: (n, 4)
    crop_boxes format: xyxy.
    """
    if not xywh:
        boxes = xyxy2xywh(boxes)

    crop_boxes = boxes.astype(np.float32).copy()

    # 긴 변을 기준으로 정사각형 변환
    crop_boxes[..., 2:] = boxes[..., 2:].max(axis=1, keepdims=True)

    # 패딩
    pad = crop_boxes[..., 2:] * pad_ratio
    crop_boxes[..., 2:] += pad

    # 좌표 보정
    crop_boxes = xywh2xyxy(crop_boxes)
    crop_boxes = clip_boxes(crop_boxes, img.shape)
    crop_boxes = crop_boxes.round().astype(np.int32)

    # 이미지 크롭
    cropped_imgs = [img[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]] for xyxy in crop_boxes]
    return cropped_imgs, crop_boxes


def create_yaml_file():
    metadata = {
        "path": "../../data/face-mix-single-face",
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "names": {
            0: "face",
        },
    }
    with open(save_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


root = Path("../../data/face-mix").resolve(strict=True)
save_dir = root.with_name(root.name + "-single-face")
small_box_threshold = 10  # pixels
pad_ratio = 0.2


def main():
    images_dir = root / "images"
    labels_dir = root / "labels"
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    create_yaml_file()

    img_paths = sorted(images_dir.rglob("*.jpg"))
    for img_path in tqdm(img_paths, dynamic_ncols=True):
        img = cv2.imread(str(img_path))
        assert img is not None

        label_path = labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")
        label = label_path.read_text().splitlines()
        label = np.array([label_one.split() for label_one in label], np.float32)

        for i, label_one in enumerate(label):
            xywh = label_one[1:5].copy()
            xywh[0::2] *= img.shape[1]
            xywh[1::2] *= img.shape[0]
            kpts = label_one[5:].copy()

            # 필터링
            if xywh[2] < small_box_threshold or xywh[3] < small_box_threshold:
                continue
            if np.all(kpts[2::3] == 0):
                continue

            # 이미지 크롭
            cropped_img, crop_xyxy = make_crop_img(img, xywh[np.newaxis, :], pad_ratio, xywh=True)
            cropped_img, crop_xyxy = cropped_img[0], crop_xyxy[0]

            # 라벨 크롭
            cropped_label = label_one.copy()
            cropped_label[1:5:2] *= img.shape[1]
            cropped_label[2:5:2] *= img.shape[0]
            cropped_label[5::3] *= img.shape[1]
            cropped_label[6::3] *= img.shape[0]
            cropped_label[1] -= crop_xyxy[0]
            cropped_label[2] -= crop_xyxy[1]
            cropped_label[5::3] -= crop_xyxy[0]
            cropped_label[6::3] -= crop_xyxy[1]
            cropped_label[1:5:2] /= cropped_img.shape[1]
            cropped_label[2:5:2] /= cropped_img.shape[0]
            cropped_label[5::3] /= cropped_img.shape[1]
            cropped_label[6::3] /= cropped_img.shape[0]

            # 키포인트가 크롭 이미지 밖에 있으면 0으로 채움
            for kpt in np.split(cropped_label[5:], 5):
                x, y, v = kpt
                if v == 0:
                    kpt.fill(0)
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    kpt.fill(0)

            # 크롭 라벨 검증
            assert np.all(0 <= cropped_label[1:5]) and np.all(cropped_label[1:5] <= 1)
            assert np.all(0 <= cropped_label[[5, 6, 8, 9, 11, 12, 14, 15, 17, 18]]) and np.all(
                cropped_label[[5, 6, 8, 9, 11, 12, 14, 15, 17, 18]] <= 1
            )
            assert np.isin(cropped_label[[7, 10, 13, 16, 19]], [0, 1, 2]).all()

            # 크롭 이미지 저장
            save_path = save_dir / img_path.relative_to(root).with_stem(f"{img_path.stem}_{i}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cropped_img)

            # 크롭 라벨 저장
            save_path = save_dir / label_path.relative_to(root).with_stem(f"{img_path.stem}_{i}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"{int(cropped_label[0])} ")
                f.write(" ".join([f"{x:.6f}" for x in cropped_label[1:]]) + "\n")


if __name__ == "__main__":
    main()
