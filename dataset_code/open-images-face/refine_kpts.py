"""목록(txt)에 든 Open Images 오토라벨의 키포인트를 ORFormer 좌우 반전 평균으로 교정한다 (widerface/refine_kpts.py 재사용).

5점이 모두 있고 이미지 경계에 닿지 않은 32px(sqrt(w * h)) 이상 얼굴을 교정한다. COCO val의 WholeBody 사람 키포인트로 잰
5점 NME 중앙값(오토라벨러 261125 ep150 -> ORFormer): 32-48px 4.87 -> 3.67%, 48-96px 3.87 -> 3.53%. 96px 이상은
2.78 -> 3.67%지만 눈(눈동자 vs 눈 윤곽 중심)과 코끝의 정의 차이가 대부분이고 입꼬리는 ORFormer가 더 가깝다.
원본/반전 예측이 어긋나는 얼굴(약 13%)은 그대로 둔다. 박스와 visibility는 바꾸지 않는다. 목록에 없는 이미지와 val은
건드리지 않는다.

라벨을 제자리에서 고치므로 한 번만 실행한다 (다시 돌리면 교정된 라벨을 또 교정한다). I N을 주면 목록을 N등분한 I번째만
처리한다.
사용법 (레포 루트, infer-kit 환경):
    uv run --project ../infer-kit --with pyyaml python dataset_code/open-images-face/refine_kpts.py LIST_TXT [I N]
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from infer_kit.models.face_landmarker import FaceLandmarker
from tqdm import tqdm

from ultralytics.utils.ops import xywhn2xyxy

sys.path.insert(0, str(Path(__file__).parents[1] / "widerface"))
from refine_kpts import EDGE, MIN_SIZE, fl_model, refine_kpts

NEVER_FILL = 10**9  # 키포인트가 없는 얼굴(뒤통수 등)은 채우지 않는다

list_txt = Path(sys.argv[1]).resolve()
i, n = (int(x) for x in sys.argv[2:4]) if len(sys.argv) > 2 else (0, 1)
fl = FaceLandmarker(fl_model)
counts = Counter()
for line in tqdm(list_txt.read_text().split()[i::n]):
    img_path = list_txt.parent / line
    label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    label = np.loadtxt(label_path, ndmin=2)
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]
    boxes = xywhn2xyxy(label[:, 1:5], width, height)
    kpts = label[:, 5:].reshape(-1, 5, 3) * [width, height, 1]
    size = np.sqrt((boxes[:, 2:] - boxes[:, :2]).prod(1))
    e = EDGE * np.array([width, height])
    cut = np.c_[boxes[:, :2] <= e, boxes[:, 2:] >= [width, height] - e].any(1)
    counts.update(refine_kpts(fl, img, boxes, kpts, [], cut, NEVER_FILL))
    counts.update(target=int(((kpts[:, :, 2] > 0).all(1) & (size >= MIN_SIZE) & ~cut).sum()))
    kpts[:, :, :2] /= [width, height]
    kpts[((kpts[:, :, :2] < 0) | (kpts[:, :, :2] > 1)).any(2)] = 0  # 이미지 밖 점은 비운다
    label[:, 5:] = kpts.reshape(len(label), -1)
    np.savetxt(label_path, label, fmt=["%d"] + ["%.6f"] * (label.shape[1] - 1))
print(dict(counts))
