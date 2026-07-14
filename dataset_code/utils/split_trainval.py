import pathlib
import random

from tqdm import tqdm

root = pathlib.Path("../../data/bezel")
num_val = 200


def main():
    all_images = list((root / "images" / "train").glob("*.jpg"))

    random.shuffle(all_images)
    val_images = all_images[:num_val]
    val_stems = [i.stem for i in val_images]

    for val_stem in tqdm(val_stems, dynamic_ncols=True):
        src_image_path = root / "images" / "train" / f"{val_stem}.jpg"
        src_label_path = root / "labels" / "train" / f"{val_stem}.txt"
        dst_image_path = root / "images" / "val" / f"{val_stem}.jpg"
        dst_label_path = root / "labels" / "val" / f"{val_stem}.txt"

        src_image_path.rename(dst_image_path)
        src_label_path.rename(dst_label_path)


if __name__ == "__main__":
    main()
