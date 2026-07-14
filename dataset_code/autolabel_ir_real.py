"""Autolabel D:/data/ir-face-person (WSL: /mnt/d/data/ir-face-person) IR person images.

- person box: yolo26x-pose.pt, class 1 in output labels, keypoints all dummy 0
- face box:   faceperson_autolabeler/weights/best.pt (class 0 only), class 0 in
  output labels, predicted 5-point landmarks as keypoints
- exactly one person per image: keep largest box/face only
- no person detected -> move image to images/no_person (checked first)
- no face detected   -> move image to images/no_face
"""

import argparse
import shutil
from pathlib import Path

import cv2

from ultralytics import YOLO

REPO = Path(__file__).resolve().parent
ROOT = Path("/mnt/d/data/ir-face-person")
IMG_DIR = ROOT / "images" / "val"
LBL_DIR = ROOT / "labels" / "val"
NO_PERSON_DIR = ROOT / "images" / "no_person"
NO_FACE_DIR = ROOT / "images" / "no_face"

PERSON_CONF = 0.3
FACE_CONF = 0.3
BATCH = 32
NUM_KPT = 5


def largest_cls0(result):
    """Return index of the largest class-0 box in a Results object, or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best_i, best_area = None, -1.0
    for i in range(len(boxes)):
        if int(boxes.cls[i]) != 0:
            continue
        w, h = float(boxes.xywh[i][2]), float(boxes.xywh[i][3])
        if w * h > best_area:
            best_area, best_i = w * h, i
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N images (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="no label writes, no file moves")
    args = ap.parse_args()

    for d in (LBL_DIR, NO_PERSON_DIR, NO_FACE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    person_model = YOLO("yolo26x-pose.pt")
    face_model = YOLO(str(REPO / "faceperson_autolabeler" / "weights" / "best.pt"))
    print("person model classes:", person_model.names, flush=True)
    print("face model classes:", face_model.names, flush=True)

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    total = len(imgs)
    print(f"images to process: {total}", flush=True)

    n_labeled = n_no_person = n_no_face = n_bad = 0

    for start in range(0, total, BATCH):
        chunk = imgs[start : start + BATCH]
        arrays, paths = [], []
        for p in chunk:
            im = cv2.imread(str(p))
            if im is None:
                print(f"UNREADABLE: {p.name}", flush=True)
                n_bad += 1
                continue
            arrays.append(im)
            paths.append(p)
        if not arrays:
            continue

        pres = person_model.predict(arrays, conf=PERSON_CONF, device=0, half=True, verbose=False)
        fres = face_model.predict(arrays, conf=FACE_CONF, device=0, half=True, verbose=False)

        for p, pr, fr in zip(paths, pres, fres):
            pi = largest_cls0(pr)
            if pi is None:
                n_no_person += 1
                if args.dry_run:
                    print(f"[dry] no person -> {p.name}", flush=True)
                else:
                    shutil.move(str(p), str(NO_PERSON_DIR / p.name))
                continue

            fi = largest_cls0(fr)
            if fi is None:
                n_no_face += 1
                if args.dry_run:
                    print(f"[dry] no face -> {p.name}", flush=True)
                else:
                    shutil.move(str(p), str(NO_FACE_DIR / p.name))
                continue

            # face row: class 0, predicted 5-point landmarks (visibility 2)
            fcx, fcy, fw, fh = (float(v) for v in fr.boxes.xywhn[fi])
            kpts = fr.keypoints.xyn[fi]  # (5, 2) normalized
            kpt_str = " ".join(f"{float(x):.6f} {float(y):.6f} 2.000000" for x, y in kpts)
            face_line = f"0 {fcx:.6f} {fcy:.6f} {fw:.6f} {fh:.6f} {kpt_str}"

            # person row: class 1, dummy keypoints all 0
            pcx, pcy, pw, ph = (float(v) for v in pr.boxes.xywhn[pi])
            dummy = " ".join(["0.000000"] * (NUM_KPT * 3))
            person_line = f"1 {pcx:.6f} {pcy:.6f} {pw:.6f} {ph:.6f} {dummy}"

            n_labeled += 1
            if args.dry_run:
                print(f"[dry] {p.name}\n  {face_line}\n  {person_line}", flush=True)
            else:
                (LBL_DIR / f"{p.stem}.txt").write_text(face_line + "\n" + person_line + "\n")

        done = min(start + BATCH, total)
        if (start // BATCH) % 20 == 0 or done == total:
            print(
                f"progress {done}/{total}  labeled={n_labeled} no_person={n_no_person} "
                f"no_face={n_no_face} unreadable={n_bad}",
                flush=True,
            )

    print(
        f"DONE  labeled={n_labeled} no_person={n_no_person} no_face={n_no_face} unreadable={n_bad}",
        flush=True,
    )


if __name__ == "__main__":
    main()
