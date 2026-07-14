from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import ultralytics.utils.ops

root = Path("../../data/widerface").resolve(strict=True)
box_size_min = 90
box_size_max = 100


def main():
    label_paths = sorted((root / "labels/val").rglob("**/*.txt"))

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        image_path = root / "images" / "val" / (label_path.stem + ".jpg")

        image = cv2.imread(str(image_path))
        assert image is not None
        raw_label = label_path.read_text("utf-8").splitlines()

        new_label = []
        for label_one in raw_label:
            box = np.array([float(i) for i in label_one.split(" ")[1:5]])
            h = box[3] * image.shape[0]
            if box_size_min <= h <= box_size_max:
                new_label.append(label_one + "\n")
            else:
                xyxy = ultralytics.utils.ops.xywhn2xyxy(box, image.shape[1], image.shape[0])
                xyxy = xyxy.round().astype(np.int32)
                image[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]] = 0
        label_path.open("w").writelines(new_label)

        cv2.imwrite(str(image_path), image)


if __name__ == "__main__":
    main()
