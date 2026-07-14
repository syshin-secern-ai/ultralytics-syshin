import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

import ultralytics.utils.ops

root = Path("../../data/widerface").resolve()
face_class = 0


def process_images():
    # Extract image files
    print("Unzip 'WIDER_train.zip' and 'WIDER_val.zip'...")
    with zipfile.ZipFile(root.parent / "WIDER_train.zip") as f:
        f.extractall(root)
    with zipfile.ZipFile(root.parent / "WIDER_val.zip") as f:
        f.extractall(root)

    # Make directories
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Move images
    train_img_paths: list[Path] = sorted((root / "WIDER_train").rglob("**/*.jpg"))
    val_img_paths: list[Path] = sorted((root / "WIDER_val").rglob("**/*.jpg"))
    for train_img_path in tqdm(train_img_paths, "Move train images", dynamic_ncols=True):
        shutil.move(root / train_img_path, train_dir)
    for val_img_path in tqdm(val_img_paths, "Move val images", dynamic_ncols=True):
        shutil.move(root / val_img_path, val_dir)

    # Remove useless directories
    shutil.rmtree(root / "WIDER_train")
    shutil.rmtree(root / "WIDER_val")


def process_labels():
    # Extract label files
    shutil.rmtree(root / "labels", ignore_errors=True)
    (root / "train_label.txt").unlink(missing_ok=True)
    (root / "val_label.txt").unlink(missing_ok=True)
    with zipfile.ZipFile(root.parent / "retinaface_gt_v1.1.zip") as f:
        f.extractall(root)
    (root / "train" / "label.txt").rename(root / "train_label.txt")
    (root / "val" / "label.txt").rename(root / "val_label.txt")
    shutil.rmtree(root / "train")
    shutil.rmtree(root / "val")
    shutil.rmtree(root / "test")

    # Make directories
    train_dir = root / "labels" / "train"
    val_dir = root / "labels" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Read labels
    train_labels = {}
    val_labels = {}
    lines = (root / "train_label.txt").read_text().splitlines()
    for line in lines:
        if line.startswith("#"):
            file_name = line.rsplit("/")[-1]
            train_labels[file_name] = []
        else:
            label = [float(i) for i in line.split(" ")]
            train_labels[file_name].append(label)
    lines = (root / "val_label.txt").read_text().splitlines()
    for line in lines:
        if line.startswith("#"):
            file_name = line.rsplit("/")[-1]
            val_labels[file_name] = []
        else:
            label = [float(i) for i in line.split(" ")]
            val_labels[file_name].append(label)

    # Convert train labels
    for file_name, label in tqdm(train_labels.items(), "Convert train labels", dynamic_ncols=True):
        img = Image.open(root / "images" / "train" / file_name)
        label = np.array(label)
        with open((train_dir / file_name).with_suffix(".txt"), "w") as f:
            for label_one in label:
                # Split into box and keypoint
                box = label_one[:4].copy()
                kpt = label_one[4:19].copy()

                # Reformat
                box = ultralytics.utils.ops.ltwh2xyxy(box)
                box = ultralytics.utils.ops.xyxy2xywhn(box, img.width, img.height, clip=True)
                kpt[0::3] /= img.width
                kpt[1::3] /= img.height
                kpt[2::3].round()
                kpt[2::3] += 1  # 0-2 visible

                # Post-process
                if (box[2] * img.width) < 6 or (box[3] * img.height) < 6:
                    continue  # Exclude this label_one
                for i in np.split(kpt, 5):
                    # Zero fill not labeled keypoint
                    if i[-1] == 0:
                        i.fill(0)
                    # Zero fill if coordinates are outside image bounds (0-1 normalized)
                    if np.any(i[:2] < 0) or np.any(i[:2] > 1):
                        i.fill(0)

                f.write(f"{face_class} {' '.join([f'{i:.6f}' for i in np.concatenate((box, kpt))])}\n")

    # Convert val labels
    for file_name, label in tqdm(val_labels.items(), "Convert val labels", dynamic_ncols=True):
        img = Image.open(root / "images" / "val" / file_name)
        label = np.array(label)
        with open((val_dir / file_name).with_suffix(".txt"), "w") as f:
            for label_one in label:
                box = label_one[:4].copy()
                kpt = np.zeros(15, np.float64)

                # Reformat
                box = ultralytics.utils.ops.ltwh2xyxy(box)
                box = ultralytics.utils.ops.xyxy2xywhn(box, img.width, img.height, clip=True)

                f.write(f"{face_class} {' '.join([f'{i:.6f}' for i in np.concatenate((box, kpt))])}\n")

    # Remove useless files
    (root / "train_label.txt").unlink()
    (root / "val_label.txt").unlink()


def create_yaml_file():
    metadata = {
        "path": "../../data/widerface",
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 2, 4, 3],
        "names": {
            face_class: "face",
        },
    }
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def main():
    shutil.rmtree(root, ignore_errors=True)
    process_images()
    process_labels()
    create_yaml_file()


if __name__ == "__main__":
    main()
