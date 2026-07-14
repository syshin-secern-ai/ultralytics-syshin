"""main.py가 만든 WIDER Face train/val YOLO 라벨의 키포인트를 WFLW 사람 GT와 ORFormer로 교정한다.

RetinaFace 키포인트 라벨은 노이즈가 커서 교정하고, 라벨이 없는 얼굴은 새로 채운다. WFLW(train/test)에 있는 얼굴은
사람 GT로, 나머지는 ORFormer(infer-kit FaceLandmarker) 좌우 반전 평균으로 바꾼다. WFLW test 1,943개 얼굴 기준
5점 NME: RetinaFace 라벨 5.09%, ORFormer 3.93%, 반전 평균 3.71%. WFLW 얼굴을 축소해 잰 ORFormer NME는
48px 3.86%, 32px 5.00%, 24px 6.30%, 16px 8.87%, 10px 13.19%다. 라벨이 큰 얼굴(중앙값 113px)에서도 5.09%라 작은
얼굴에선 더 나쁘므로, 라벨된 얼굴은 32px부터 교정한다. 라벨이 없던 얼굴은 같은 크기의 사람 라벨과 노이즈가 비슷한
24px부터 채운다 (val은 16px부터, 아래 참고). 박스와 기존 visibility는 그대로 두고, 새로 채운 키포인트의 visibility는 2다. 이미지 밖으로 나간
키포인트만 0이다. 이미지 한 변에 잘린 얼굴은 박스를 평균 얼굴 비율로 이미지 밖까지 늘려 새로 채우고, 두 변 이상 잘렸거나
교정하지 못한 잘린 얼굴은 키포인트를 모두 비운다. val은 키포인트 라벨이 없고 WFLW(WIDER train 이미지에서 뽑음)도 없으므로 ORFormer로 채우기만 한다. train은 작은
얼굴에도 RetinaFace 라벨이 있어(16-24px 62%) val을 24px부터 채우면 모델이 맞게 예측한 점을 val kobj_loss가 오답으로 센다.
그래서 val은 16px부터 채운다 (NME 8.87%).

labels/<split>을 제자리에서 고치므로 split마다 main.py 직후 한 번만 실행한다 (다시 돌리면 교정된 라벨을 또 교정한다).
인자로 split을 고를 수 있고, 없으면 train과 val을 모두 고친다.
infer_kit이 필요하므로 레포 루트에서 infer-kit 환경으로 실행한다 (pyyaml은 infer-kit에 없어 임시로 추가):
    uv run --project ../infer-kit --with pyyaml python dataset_code/widerface/refine_kpts.py [train] [val]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from infer_kit.models.face_landmarker import FaceLandmarker
from tqdm import tqdm

import ultralytics.utils.ops
from ultralytics.utils.metrics import bbox_ioa

root = Path("../../data/widerface").resolve()
wflw_dir = Path("../ORFormer/Dataset/WFLW").resolve()
fl_model = Path("../infer-kit/model_files/fl_261106.onnx")
WFLW5 = [96, 97, 54, 76, 82]  # WFLW 98점 중 5점 (이미지 기준 왼쪽/오른쪽 눈동자, 코끝, 왼쪽/오른쪽 입꼬리)
WFLW5_FLIP = [97, 96, 54, 82, 76]  # 반전 이미지에서 같은 5점
MIN_SIZE = 32  # sqrt(w * h) 픽셀. 라벨된 얼굴은 ORFormer가 라벨 수준이 되는 이 크기부터 교정한다
MIN_FILL_SIZE = 24  # 라벨이 없던 얼굴은 이 크기부터 채운다 (NME 6.5%, 같은 크기의 사람 라벨과 비슷하다고 추정)
VAL_MIN_FILL_SIZE = 16  # val은 train의 작은 얼굴 라벨 비율과 맞추려고 이 크기부터 채운다
EDGE = 0.002  # 정규화 박스가 이미지 경계에서 이 거리 안이면 잘린 얼굴로 본다
ASPECT = 0.78  # 얼굴 박스 w / h 중앙값 (경계에 닿지 않은 32px 이상 train 얼굴). 잘린 박스를 이 비율로 늘린다
MAX_FLIP_DIFF = 0.06  # 원본/반전 예측 차이(눈 사이 거리 비율). 이보다 크면 ORFormer가 라벨보다 낫지 않다
# 키포인트 라벨이 없는 얼굴의 ORFormer 크롭 추정(CALIB 특징)에 쓰는 평균 키포인트. 박스 중심 기준, sqrt(w * h)로
# 정규화했다. RetinaFace train 라벨 중 5점이 모두 있는 64px 이상 얼굴의 평균이다.
TEMPLATE = np.array([[-0.166, -0.127], [0.159, -0.131], [-0.003, 0.089], [-0.133, 0.273], [0.135, 0.268]])


def load_wflw() -> dict[str, list[np.ndarray]]:
    """WFLW train/test 98점 사람 GT를 파일명(WIDER와 같음)별로 읽는다."""
    faces = defaultdict(list)
    ann = wflw_dir / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
    for subset in ("train", "test"):
        for line in (ann / f"list_98pt_rect_attr_{subset}.txt").read_text().splitlines():
            t = line.split()
            faces[t[-1].rsplit("/")[-1]].append(np.array(t[:196], np.float64).reshape(98, 2))
    return faces


def extend_cut_boxes(boxes: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """이미지 한 변에 잘린 얼굴 박스를 ASPECT 비율로 이미지 밖까지 늘린다.

    WIDER 박스는 보이는 부분만 감싸서, 그대로 ORFormer에 넣으면 5점이 박스 안에 몰린다. 한 변만 잘리면 다른 축 길이가
    온전하므로 원래 크기를 추정할 수 있고, 두 변 이상 잘리면 알 수 없다.

    Args:
        boxes (np.ndarray): (N, 4) 픽셀 xyxy 얼굴 박스.
        width (int): 이미지 너비.
        height (int): 이미지 높이.

    Returns:
        (np.ndarray): (N, 4) 늘린 박스.
        (np.ndarray): (N, 4) 박스가 닿은 변 (왼쪽, 위, 오른쪽, 아래).
    """
    e = EDGE * np.array([width, height])
    side = np.c_[boxes[:, :2] <= e, boxes[:, 2:] >= [width, height] - e]
    b = boxes.copy()
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    fw, fh = np.maximum(w, ASPECT * h), np.maximum(h, w / ASPECT)
    left, top, right, bottom = (side & (side.sum(1) == 1)[:, None]).T
    b[left, 0] = b[left, 2] - fw[left]
    b[top, 1] = b[top, 3] - fh[top]
    b[right, 2] = b[right, 0] + fw[right]
    b[bottom, 3] = b[bottom, 1] + fh[bottom]
    return b, side


def refine_kpts(
    fl: FaceLandmarker,
    img: np.ndarray,
    boxes: np.ndarray,
    kpts: np.ndarray,
    wflw_faces: list[np.ndarray],
    skip: np.ndarray,
    min_fill_size: int,
) -> list[str]:
    """키포인트 xy를 제자리에서 교정하고, 라벨이 없던 키포인트는 visibility 2로 새로 채운다.

    Args:
        fl (FaceLandmarker): ORFormer 랜드마크 모델.
        img (np.ndarray): BGR 원본 이미지.
        boxes (np.ndarray): (N, 4) 픽셀 xyxy 얼굴 박스.
        kpts (np.ndarray): (N, 5, 3) 픽셀 xy와 visibility(0이면 라벨 없음). 제자리에서 고친다.
        wflw_faces (list[np.ndarray]): 이 이미지의 WFLW (98, 2) 사람 GT.
        skip (np.ndarray): (N,) ORFormer로 교정하지 않을 얼굴 (원래 크기를 알 수 없는 잘린 얼굴).
        min_fill_size (int): 라벨이 없던 얼굴을 채우는 최소 크기 sqrt(w * h) 픽셀.

    Returns:
        (list[str]): 얼굴마다 처리 결과 ("wflw", "orformer", "rejected", "kept"). 라벨이 없던 얼굴은 "fill:" 접두사.
    """
    labeled = (kpts[:, :, 2] > 0).all(1)
    status = ["kept"] * len(kpts)

    def update(j, xy, s):
        kpts[j, :, :2] = xy
        kpts[j, :, 2] = 2
        status[j] = s

    if wflw_faces:
        lmk_boxes = np.array([np.r_[p.min(0), p.max(0)] for p in wflw_faces])
        iou = bbox_ioa(lmk_boxes, boxes, iou=True)
        for pts, i in zip(wflw_faces, iou):
            j = int(i.argmax())
            if i[j] >= 0.3:
                update(j, pts[WFLW5], "wflw")

    w = img.shape[1] - 1  # 반전: x -> w - x (픽셀 중심 좌표)
    img_flip = None
    size = np.sqrt((boxes[:, 2:] - boxes[:, :2]).prod(1))
    for j in np.flatnonzero((size >= np.where(labeled, MIN_SIZE, min_fill_size)) & ~skip):
        if status[j] != "kept":
            continue
        img_flip = np.ascontiguousarray(img[:, ::-1]) if img_flip is None else img_flip
        x1, y1, x2, y2 = boxes[j]
        kpt = kpts[j, :, :2] if labeled[j] else (boxes[j, :2] + boxes[j, 2:]) / 2 + TEMPLATE * size[j]
        pred = fl.predict_box(img, boxes[j], kpt).landmarks[WFLW5]
        kpt_flip = kpt[[1, 0, 2, 4, 3]].copy()
        kpt_flip[:, 0] = w - kpt_flip[:, 0]
        pred_flip = fl.predict_box(img_flip, np.array([w - x2, y1, w - x1, y2]), kpt_flip).landmarks[WFLW5_FLIP]
        pred_flip[:, 0] = w - pred_flip[:, 0]
        diff = np.linalg.norm(pred - pred_flip, axis=1).mean() / np.linalg.norm(pred[0] - pred[1])
        if diff > MAX_FLIP_DIFF:
            status[j] = "rejected"
            continue
        update(j, (pred + pred_flip) / 2, "orformer")
    return [s if l else f"fill:{s}" for s, l in zip(status, labeled)]


def main():
    fl = FaceLandmarker(fl_model)
    wflw = load_wflw()
    for split in sys.argv[1:] or ("train", "val"):
        min_fill_size = MIN_FILL_SIZE if split == "train" else VAL_MIN_FILL_SIZE
        counts = defaultdict(int)
        for label_path in tqdm(sorted((root / "labels" / split).glob("*.txt")), f"Refine {split} keypoints"):
            label = np.loadtxt(label_path, ndmin=2)
            if not len(label):
                continue
            img_name = label_path.with_suffix(".jpg").name
            img = cv2.imread(str(root / "images" / split / img_name))
            height, width = img.shape[:2]
            boxes, side = extend_cut_boxes(
                ultralytics.utils.ops.xywhn2xyxy(label[:, 1:5], width, height), width, height
            )
            kpts = label[:, 5:].reshape(-1, 5, 3) * [width, height, 1]
            kpts[side.any(1)] = 0  # 잘린 얼굴의 RetinaFace 라벨은 박스 안에 몰려 있다. 늘린 박스로 새로 채운다
            wflw_faces = wflw.get(img_name, []) if split == "train" else []  # WFLW는 WIDER train 이미지에서만 뽑았다
            # 늘린 박스가 이미지 밖까지 가므로 검은 테두리를 붙여 ORFormer 크롭이 이미지 안에 있게 한다
            pad = int(np.ceil(max(0, -boxes.min(), (boxes[:, 2:] - [width, height]).max())))
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            kpts[:, :, :2] += pad
            wflw_faces = [p + pad for p in wflw_faces]
            for status in refine_kpts(fl, img, boxes + pad, kpts, wflw_faces, side.sum(1) > 1, min_fill_size):
                counts[status] += 1
            kpts[:, :, :2] -= pad

            kpts[:, :, :2] /= [width, height]
            # 이미지 밖 키포인트는 main.py처럼 0으로 채운다. 잘린 얼굴은 보이는 점만 남는다
            kpts[((kpts[:, :, :2] < 0) | (kpts[:, :, :2] > 1)).any(2)] = 0
            label[:, 5:] = kpts.reshape(len(label), -1)
            np.savetxt(label_path, label, fmt=["%d"] + ["%.6f"] * (label.shape[1] - 1))
        print(f"{split} keypoints: {dict(counts)}")


if __name__ == "__main__":
    main()
