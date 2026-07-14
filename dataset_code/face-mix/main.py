import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

widerface_root = Path("../../data/widerface").resolve(strict=True)
coco_face_root = Path("../../data/coco-face").resolve(strict=True)
face_mix_root = Path("../../data/face-mix").resolve()


def main():
    # Reset destination directory
    shutil.rmtree(face_mix_root, ignore_errors=True)

    # Make dataset structure
    (face_mix_root / "images" / "train").mkdir(parents=True)
    (face_mix_root / "images" / "val").mkdir(parents=True)
    (face_mix_root / "labels" / "train").mkdir(parents=True)
    (face_mix_root / "labels" / "val").mkdir(parents=True)

    # Move widerface data
    widerface_src_paths = sorted((widerface_root / "images").rglob("**/*.jpg"))
    widerface_src_paths += sorted((widerface_root / "labels").rglob("**/*.txt"))
    for src_path in tqdm(widerface_src_paths, "Move widerface data", dynamic_ncols=True):
        parts = list(src_path.parts)
        parts[-4] = face_mix_root.name
        dest_path = Path(*parts)
        src_path.rename(dest_path)

    # Move coco-face data
    coco_face_src_paths = sorted((coco_face_root / "images").rglob("**/*.jpg"))
    coco_face_src_paths += sorted((coco_face_root / "labels").rglob("**/*.txt"))
    for src_path in tqdm(coco_face_src_paths, "Move coco-face data", dynamic_ncols=True):
        parts = list(src_path.parts)
        parts[-4] = face_mix_root.name
        dest_path = Path(*parts)
        src_path.rename(dest_path)

    # Remove useless files
    shutil.rmtree(widerface_root)
    shutil.rmtree(coco_face_root)

    # Create yaml file
    metadata = {
        "path": "../../data/face-mix",
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "names": {
            0: "face",
        },
    }
    with open(face_mix_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


if __name__ == "__main__":
    main()
