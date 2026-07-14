import shutil
import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm

root = Path("../../data/coco-person").resolve()


def extract_archives():
    print("Unzip 'train2017.zip'...")
    with zipfile.ZipFile(root.parent / "train2017.zip") as f:
        f.extractall(root)

    print("Unzip 'val2017.zip'...")
    with zipfile.ZipFile(root.parent / "val2017.zip") as f:
        f.extractall(root)

    print("Unzip 'coco2017labels-pose.zip'...")
    with zipfile.ZipFile(root.parent / "coco2017labels-pose.zip") as f:
        f.extractall(root)
    shutil.move(root / "coco-pose" / "labels", root)
    (root / "coco-pose" / "train2017.txt").rename(root / "train2017.txt")
    (root / "coco-pose" / "val2017.txt").rename(root / "val2017.txt")
    (root / "labels" / "train2017").rename(root / "labels" / "train")
    (root / "labels" / "val2017").rename(root / "labels" / "val")

    # Remove useless files
    shutil.rmtree(root / "coco-pose")


def process_images():
    # Read image paths file
    train_image_names = (root / "train2017.txt").read_text().splitlines()
    val_image_names = (root / "val2017.txt").read_text().splitlines()
    train_image_names = sorted([i.rsplit("/")[-1] for i in train_image_names])
    val_image_names = sorted([i.rsplit("/")[-1] for i in val_image_names])

    # Make directories
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Move images
    for train_image_name in tqdm(train_image_names, "Move train images", dynamic_ncols=True):
        (root / "train2017" / train_image_name).rename(train_dir / train_image_name)
    for val_image_name in tqdm(val_image_names, "Move val images", dynamic_ncols=True):
        (root / "val2017" / val_image_name).rename(val_dir / val_image_name)

    # Remove useless files
    shutil.rmtree(root / "train2017")
    shutil.rmtree(root / "val2017")
    (root / "train2017.txt").unlink()
    (root / "val2017.txt").unlink()


def convert_labels():
    label_paths: list[Path] = sorted((root / "labels").rglob("**/*.txt"))
    for label_path in tqdm(label_paths, "Convert labels", dynamic_ncols=True):
        label = label_path.read_text().splitlines()
        label = [" ".join(i.split()[:5]) for i in label]
        label_path.write_text("\n".join(label) + "\n")


def create_yaml_file():
    metadata = {
        "path": "../../data/coco-person",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def main():
    shutil.rmtree(root, ignore_errors=True)
    extract_archives()
    process_images()
    convert_labels()
    create_yaml_file()


if __name__ == "__main__":
    main()
