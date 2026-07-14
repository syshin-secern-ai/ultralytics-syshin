"""Open Images V7에서 'Human face' 박스가 있는 이미지만 받아 YOLO 형식 GT 라벨을 만든다.

IsGroupOf(여러 얼굴을 한 박스로 묶음)나 IsDepiction(그림, 조각상) 얼굴이 있는 이미지는 통째로 뺀다.
fiftyone은 다운로드에만 쓰고 박스는 csv에서 직접 읽는다(fiftyone의 MongoDB가 Ubuntu 26.04에서 뜨지 않는다).
이미지는 fiftyone 폴더에 두고 images/SPLIT에 심볼릭 링크만 건다.
GT 박스는 오토라벨 검증용이라 labels_gt에 둔다(auto_labeling.py가 labels를 지우고 새로 만든다).
사용법: main.py [SPLIT=train] [MAX_SAMPLES]  (SPLIT: train, val, test)
"""

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

FACE = "/m/0dzct"  # Human face
root = Path("../../data/open-images-face").resolve()
split = sys.argv[1] if len(sys.argv) > 1 else "train"

fo.config.dataset_zoo_dir = str(root / "fiftyone")
foz.download_zoo_dataset(
    "open-images-v7",
    split="validation" if split == "val" else split,  # fiftyone의 split 이름
    label_types=["detections"],
    classes=["Human face"],
    max_samples=int(sys.argv[2]) if len(sys.argv) > 2 else None,
)
src = root / "fiftyone" / "open-images-v7" / ("validation" if split == "val" else split)
imgs = {p.stem: p for p in (src / "data").glob("*.jpg")}

faces, bad = defaultdict(list), set()
with open(src / "labels" / "detections.csv") as f:
    col = {k: i for i, k in enumerate(next(f).strip().split(","))}
    for line in f:
        if f",{FACE}," not in line:
            continue
        r = line.split(",")
        i = r[col["ImageID"]]
        if i not in imgs:
            continue
        if r[col["IsGroupOf"]] == "1" or r[col["IsDepiction"]] == "1":
            bad.add(i)
        x0, x1, y0, y1 = (float(r[col[k]]) for k in ("XMin", "XMax", "YMin", "YMax"))
        faces[i].append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n")

img_dir, lb_dir = root / "images" / split, root / "labels_gt" / split
for d in img_dir, lb_dir:  # 둘 다 다시 만들 수 있는 파생물이라 지우고 새로 만든다
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
keep = sorted(set(faces) - bad)
for i in keep:
    (img_dir / f"{i}.jpg").symlink_to(imgs[i])
    (lb_dir / f"{i}.txt").write_text("".join(faces[i]))
print(f"{len(imgs):,} images -> {len(keep):,} after dropping IsGroupOf/IsDepiction")
