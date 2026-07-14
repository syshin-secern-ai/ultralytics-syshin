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

    print("Unzip 'coco2017labels.zip'...")
    with zipfile.ZipFile(root.parent / "coco2017labels.zip") as f:
        f.extractall(root)
    shutil.move(root / "coco" / "labels", root)
    (root / "labels" / "train2017").rename(root / "labels" / "train")
    (root / "labels" / "val2017").rename(root / "labels" / "val")

    # Remove useless files
    shutil.rmtree(root / "coco")


def process_images():
    (root / "images").mkdir()
    (root / "train2017").rename(root / "images" / "train")
    (root / "val2017").rename(root / "images" / "val")


def convert_labels():
    # 사람이 없는 이미지도 빈 라벨 파일을 갖도록 이미지 기준으로 순회한다
    image_paths: list[Path] = sorted((root / "images").rglob("*.jpg"))
    for image_path in tqdm(image_paths, "Convert labels", dynamic_ncols=True):
        label_path = root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        label = label_path.read_text().splitlines() if label_path.exists() else []
        label_path.write_text("".join(f"{i}\n" for i in label if i.split()[0] == "0"))


def create_yaml_file():
    metadata = {
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
