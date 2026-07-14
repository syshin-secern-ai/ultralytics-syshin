"""Convert a COCO-format annotation JSON to YOLO txt labels.

Example:
    python coco_to_yolo.py \
        --coco _annotations.coco.json \
        --images-dir images \
        --labels-dir labels \
        --class-map flag=0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_class_map(items: list[str] | None) -> dict[str, int]:
    if not items:
        return {}
    mapping: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--class-map entry must be name=index, got: {item!r}")
        name, idx = item.split("=", 1)
        mapping[name.strip()] = int(idx)
    return mapping


def build_category_lookup(categories: list[dict], overrides: dict[str, int]) -> dict[int, int]:
    """Map COCO category_id -> YOLO class index.

    Categories sharing a name collapse to the same YOLO index. Names listed in
    `overrides` use the given index; remaining names get sequential indices in
    first-seen order.
    """
    name_to_yolo: dict[str, int] = dict(overrides)
    next_idx = max(overrides.values(), default=-1) + 1
    cat_to_yolo: dict[int, int] = {}
    for cat in categories:
        name = cat["name"]
        if name not in name_to_yolo:
            name_to_yolo[name] = next_idx
            next_idx += 1
        cat_to_yolo[cat["id"]] = name_to_yolo[name]
    return cat_to_yolo


def index_images(images_dir: Path) -> dict[str, Path]:
    """Map image file name -> path (recursive)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    index: dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            index[p.name] = p
    return index


def to_yolo_bbox(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    clamp = lambda v: min(1.0, max(0.0, v))
    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True, type=Path, help="COCO json path")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Root dir of images; .txt files mirror this layout under --labels-dir. "
        "If omitted, all labels are written flat into --labels-dir.",
    )
    parser.add_argument("--labels-dir", required=True, type=Path, help="Output dir for YOLO .txt files")
    parser.add_argument(
        "--class-map",
        nargs="*",
        help="Force category name -> YOLO index, e.g. --class-map flag=0 ball=1",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not create empty .txt files for images with no annotations",
    )
    args = parser.parse_args()

    with args.coco.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_to_yolo = build_category_lookup(coco["categories"], parse_class_map(args.class_map))
    images = {img["id"]: img for img in coco["images"]}

    name_index = index_images(args.images_dir) if args.images_dir else {}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = missing = 0

    for img_id, img in images.items():
        file_name = img["file_name"]
        img_w, img_h = img["width"], img["height"]
        anns = anns_by_image.get(img_id, [])

        if not anns and args.skip_empty:
            skipped += 1
            continue

        if args.images_dir:
            src = name_index.get(Path(file_name).name)
            if src is None:
                print(f"[warn] image not found under {args.images_dir}: {file_name}")
                missing += 1
                continue
            rel = src.relative_to(args.images_dir).with_suffix(".txt")
            out_path = args.labels_dir / rel
        else:
            out_path = args.labels_dir / (Path(file_name).stem + ".txt")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for ann in anns:
            if ann.get("iscrowd"):
                continue
            cls = cat_to_yolo[ann["category_id"]]
            cx, cy, nw, nh = to_yolo_bbox(ann["bbox"], img_w, img_h)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    used = sorted(set(cat_to_yolo.values()))
    print(f"wrote {written} label files | skipped(empty) {skipped} | missing-image {missing}")
    print(f"yolo class indices used: {used}")


if __name__ == "__main__":
    main()
