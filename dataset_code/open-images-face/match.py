"""Open Images 'Human face' GT 박스로 오토라벨을 검증해, 학습에 쓸 이미지 목록(txt)을 만든다.

GT 얼굴과 오토라벨 얼굴이 빠짐없이 1:1로 매칭(IoU>=0.5)되는 이미지만 목록에 넣는다. 라벨은 오토라벨 것(WIDER 기준
박스 + 5점 키포인트)을 그대로 쓰므로 새로 쓰지 않는다. 탈락한 이미지는 목록에서 빠질 뿐 지우지 않는다.
박스는 정규화 좌표 그대로 비교한다(IoU는 축별 스케일에 불변).
사용법: match.py GT_LABEL_DIR AUTO_LABEL_DIR LIST_TXT
"""

import sys
from collections import Counter
from pathlib import Path

import torch

from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

GT, AUTO, LIST = (Path(x) for x in sys.argv[1:4])


def boxes(p):
    """라벨 파일의 박스를 xyxy로 읽는다. 파일이 없으면 빈 텐서."""
    rows = [[float(v) for v in x.split()[1:5]] for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return xywh2xyxy(torch.tensor(rows).reshape(-1, 4))


c, kept = Counter(), []
for p in sorted(GT.glob("*.txt")):
    g, a = boxes(p), boxes(AUTO / p.name)
    c["images"] += 1
    if len(g) == 0:
        c["no_gt"] += 1
    elif len(g) != len(a):
        c["count_mismatch"] += 1
    else:
        best, idx = box_iou(g, a).max(1)
        if best.min() < 0.5 or len(idx.unique()) < len(idx):  # GT마다 서로 다른 오토라벨이 붙어야 1:1
            c["iou_mismatch"] += 1
        else:
            c["kept"] += 1
            c["faces"] += len(g)
            kept.append(f"./images/{AUTO.name}/{p.stem}.jpg\n")
LIST.write_text("".join(kept))
print(dict(c))
