import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Settings
class_id = 0
src_root = Path("../../data/passport_coco")
dst_root = Path("../../data/passport-seg")
image_dir_path = src_root.joinpath("passport")
label_path = src_root.joinpath("passport.json")

# Path variables
dst_images_train_path = dst_root.joinpath("images", "train")
dst_labels_train_path = dst_root.joinpath("labels", "train")
dst_images_val_path = dst_root.joinpath("images", "val")
dst_labels_val_path = dst_root.joinpath("labels", "val")

# Make directories
dst_images_train_path.mkdir(parents=True, exist_ok=True)
dst_labels_train_path.mkdir(parents=True, exist_ok=True)
dst_images_val_path.mkdir(parents=True, exist_ok=True)
dst_labels_val_path.mkdir(parents=True, exist_ok=True)

# Load coco format label
with open(label_path, encoding="utf-8") as f:
    coco_label = json.load(f)

# Map image id and image file name
image_id_filename = {image["id"]: Path(image["file_name"]) for image in coco_label["images"]}

for anno in tqdm(coco_label["annotations"], dynamic_ncols=True):
    # Convert annotation to yolo format
    segment = np.array(anno["segmentation"][0]).astype(np.float32).reshape(4, 2)
    segment[..., 0] = segment[..., 0] / anno["width"]
    segment[..., 1] = segment[..., 1] / anno["height"]
    segment = segment.flatten()

    # Copy image corresponding to annotation
    image_filename = image_id_filename[anno["image_id"]]
    shutil.copy(
        image_dir_path.joinpath(image_filename),
        dst_images_train_path.joinpath(f"{image_filename.stem}.jpg"),
    )

    # Save yolo format annotation
    new_label = f"{class_id} {' '.join([f'{s:.6f}' for s in segment])}\n"
    dst_labels_train_path.joinpath(f"{image_filename.stem}.txt").write_text(new_label, "utf-8")
