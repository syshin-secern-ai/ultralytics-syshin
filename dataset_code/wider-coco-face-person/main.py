import shutil
import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm

wider_face_person_root = Path("../../data/wider-face-person").resolve(strict=True)
coco_face_person_root = Path("../../data/coco-face-person").resolve(strict=True)
wider_coco_face_person_root = Path("../../data/wider-coco-face-person").resolve()


def main():
    # Reset destination directory
    shutil.rmtree(wider_coco_face_person_root, ignore_errors=True)

    # Make dataset structure
    (wider_coco_face_person_root / "images" / "train").mkdir(parents=True)
    (wider_coco_face_person_root / "images" / "val").mkdir(parents=True)
    (wider_coco_face_person_root / "labels" / "train").mkdir(parents=True)
    (wider_coco_face_person_root / "labels" / "val").mkdir(parents=True)

    # Move wider-face-person data
    wider_face_person_src_paths = sorted((wider_face_person_root / "images").rglob("**/*.jpg"))
    wider_face_person_src_paths += sorted((wider_face_person_root / "labels").rglob("**/*.txt"))
    for src_path in tqdm(wider_face_person_src_paths, "Move wider-face-person data", dynamic_ncols=True):
        parts = list(src_path.parts)
        parts[-4] = wider_coco_face_person_root.name
        dest_path = Path(*parts)
        src_path.rename(dest_path)

    # Move coco-face-person data
    coco_face_person_src_paths = sorted((coco_face_person_root / "images").rglob("**/*.jpg"))
    coco_face_person_src_paths += sorted((coco_face_person_root / "labels").rglob("**/*.txt"))
    for src_path in tqdm(coco_face_person_src_paths, "Move coco-face-person data", dynamic_ncols=True):
        parts = list(src_path.parts)
        parts[-4] = wider_coco_face_person_root.name
        dest_path = Path(*parts)
        src_path.rename(dest_path)

    # Add coco background data
    print("Unzip 'train2017.zip'...")
    with zipfile.ZipFile(wider_coco_face_person_root.parent / "train2017.zip") as f:
        f.extractall(wider_coco_face_person_root)
    print("Unzip 'val2017.zip'...")
    with zipfile.ZipFile(wider_coco_face_person_root.parent / "val2017.zip") as f:
        f.extractall(wider_coco_face_person_root)
    coco_train_image_paths: list[Path] = sorted(Path(wider_coco_face_person_root / "train2017").rglob("*.jpg"))
    coco_val_image_paths: list[Path] = sorted(Path(wider_coco_face_person_root / "val2017").rglob("*.jpg"))
    for coco_train_image_path in tqdm(coco_train_image_paths, "Move coco background train images", dynamic_ncols=True):
        coco_train_image_path.replace(wider_coco_face_person_root / "images" / "train" / coco_train_image_path.name)
        label_path = wider_coco_face_person_root / "labels" / "train" / coco_train_image_path.with_suffix(".txt").name
        label_path.touch()
    for coco_val_image_path in tqdm(coco_val_image_paths, "Move coco background val images", dynamic_ncols=True):
        coco_val_image_path.replace(wider_coco_face_person_root / "images" / "val" / coco_val_image_path.name)
        label_path = wider_coco_face_person_root / "labels" / "val" / coco_val_image_path.with_suffix(".txt").name
        label_path.touch()

    # Remove useless files
    shutil.rmtree(wider_face_person_root)
    shutil.rmtree(coco_face_person_root)
    shutil.rmtree(wider_coco_face_person_root / "train2017")
    shutil.rmtree(wider_coco_face_person_root / "val2017")

    # Create yaml file
    metadata = {
        "path": "../../data/wider-coco-face-person",
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "names": {
            0: "face",
            1: "person",
        },
    }
    with open(wider_coco_face_person_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


if __name__ == "__main__":
    main()
