# https://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from ultralytics.utils.ops import xyxy2xywhn

root = Path("../../data/cnn").resolve()
face_class = 0


@dataclass
class CnnLabel:
    image_name: Path
    box: np.ndarray
    kpt: np.ndarray


def create_yaml_file():
    metadata = {
        "path": "../../data/cnn",
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
    image_dir = root / "images/val"
    label_dir = root / "labels/val"
    shutil.rmtree(label_dir, ignore_errors=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    train_label_path = root / "trainImageList.txt"
    test_label_path = root / "testImageList.txt"
    raw_text = train_label_path.read_text(encoding="utf-8").splitlines()
    raw_text.extend(test_label_path.read_text(encoding="utf-8").splitlines())
    raw_labels = [line.split() for line in raw_text]
    raw_labels = [
        CnnLabel(
            image_name=Path(Path(line[0].replace("\\", "/")).name),
            box=np.array([int(line[1]), int(line[3]), int(line[2]), int(line[4])]),
            kpt=np.array([float(x) for x in line[5:15]]),
        )
        for line in raw_labels
    ]

    labels: list[CnnLabel] = []
    for label in tqdm(raw_labels, "Convert labels", dynamic_ncols=True):
        image = Image.open(image_dir / label.image_name).convert("RGB")
        box = xyxy2xywhn(label.box, image.width, image.height, clip=True)
        assert isinstance(box, np.ndarray)
        kpt = label.kpt.copy()
        kpt[::2] /= image.width
        kpt[1::2] /= image.height
        kpt = np.concatenate([kpt.reshape((5, 2)), np.full((5, 1), 2.0)], axis=1).flatten()
        labels.append(CnnLabel(image_name=label.image_name, box=box, kpt=kpt))

    for label in tqdm(labels, "Save labels", dynamic_ncols=True):
        label_path = label_dir / label.image_name.with_suffix(".txt")
        label_path.write_text(
            f"0 {' '.join([f'{i:.6f}' for i in label.box])} {' '.join([f'{i:.6f}' for i in label.kpt])}\n",
            encoding="utf-8",
        )

    create_yaml_file()


if __name__ == "__main__":
    main()
