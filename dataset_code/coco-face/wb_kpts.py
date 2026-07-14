"""오토라벨 얼굴 중 COCO-WholeBody 얼굴과 매칭되는 것(IoU>=0.5)의 5점 키포인트를 사람이 찍은 랜드마크로 교체한다.

박스는 오토라벨 것(WIDER 기준)을 유지하고, 매칭 안 된 얼굴은 그대로 복사한다.
68점 -> 5점: 눈 중심 36-41 / 42-47, 코끝 30, 입꼬리 48 / 54. 교체한 점의 visibility는 2.
사용법: wb_kpts.py WB_JSON SRC_LABEL_DIR DST_LABEL_DIR
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WB, SRC, DST = (Path(x) for x in sys.argv[1:4])


def iou(a, b):
    """두 xywh 박스의 IoU."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


d = json.loads(WB.read_text())
sizes = {i["id"]: (i["width"], i["height"]) for i in d["images"]}
faces = {}
for a in d["annotations"]:
    fk = np.array(a["face_kpts"], dtype=float).reshape(68, 3)
    if a["face_valid"] and (fk[:, 2] > 0).all():  # 일부 점만 라벨된 얼굴은 건너뜀
        f = fk[:, :2]
        five = np.stack([f[36:42].mean(0), f[42:48].mean(0), f[30], f[48], f[54]])
        faces.setdefault(a["image_id"], []).append((a["face_box"], five))
del d

DST.mkdir(parents=True, exist_ok=False)
c = Counter()
for p in sorted(SRC.glob("*.txt")):
    W, H = sizes[int(p.stem)]
    rows = [x.split() for x in p.read_text().splitlines() if x.strip()]
    boxes = [
        [(float(r[1]) - float(r[3]) / 2) * W, (float(r[2]) - float(r[4]) / 2) * H, float(r[3]) * W, float(r[4]) * H]
        for r in rows
    ]
    gts = faces.get(int(p.stem), [])
    pairs = sorted(((iou(g[0], b), gi, bi) for gi, g in enumerate(gts) for bi, b in enumerate(boxes)), reverse=True)
    used_g, used_b = set(), set()
    for s, gi, bi in pairs:  # IoU 높은 순으로 1:1 그리디 매칭
        if s < 0.5:
            break
        if gi in used_g or bi in used_b:
            continue
        used_g.add(gi)
        used_b.add(bi)
        k = (gts[gi][1] / [W, H]).clip(0, 1)
        rows[bi][5:] = [f"{v:.6f}" for x, y in k for v in (x, y, 2.0)]
    c["images"] += 1
    c["faces"] += len(rows)
    c["wb_faces"] += len(gts)
    c["replaced"] += len(used_b)
    (DST / p.name).write_text("".join(" ".join(r) + "\n" for r in rows))
print(dict(c))
