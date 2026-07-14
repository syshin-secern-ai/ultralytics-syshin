import pathlib
import shutil

from tqdm import tqdm

root = pathlib.Path("../../data/bezel").resolve(strict=True)


def remap_image_based():
    image_paths = sorted((root / "images").rglob("**/*.jpg"))

    new_images = root / "new_images"
    (new_images / "train").mkdir(parents=True, exist_ok=True)
    (new_images / "val").mkdir(parents=True, exist_ok=True)
    new_labels = root / "new_labels"
    (new_labels / "train").mkdir(parents=True, exist_ok=True)
    (new_labels / "val").mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, dynamic_ncols=True):
        label_path = root / "labels" / image_path.parent.name / image_path.with_suffix(".txt").name
        image_path.rename(new_images.joinpath(*image_path.parts[-2:]))
        label_path.rename(new_labels.joinpath(*label_path.parts[-2:]))

    shutil.rmtree(root / "images")
    shutil.rmtree(root / "labels")
    shutil.move(new_images, root / "images")
    shutil.move(new_labels, root / "labels")


def remap_label_based():
    label_paths = sorted((root / "labels").rglob("**/*.txt"))

    new_images = root / "new_images"
    (new_images / "train").mkdir(parents=True, exist_ok=True)
    (new_images / "val").mkdir(parents=True, exist_ok=True)
    new_labels = root / "new_labels"
    (new_labels / "train").mkdir(parents=True, exist_ok=True)
    (new_labels / "val").mkdir(parents=True, exist_ok=True)

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        image_path = root / "images" / label_path.parent.name / label_path.with_suffix(".jpg").name
        image_path.rename(new_images.joinpath(*image_path.parts[-2:]))
        label_path.rename(new_labels.joinpath(*label_path.parts[-2:]))

    shutil.rmtree(root / "images")
    shutil.rmtree(root / "labels")
    shutil.move(new_images, root / "images")
    shutil.move(new_labels, root / "labels")


def main():
    # remap_image_based()
    remap_label_based()


if __name__ == "__main__":
    main()
