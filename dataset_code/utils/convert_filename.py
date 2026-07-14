import pathlib
import shutil

from tqdm import tqdm

root = pathlib.Path("../../data/bezel").resolve(strict=True)
zfill_width = 6


def main():
    image_paths = sorted(root.joinpath("images").rglob("**/*.*"))
    label_paths = sorted(root.joinpath("labels").rglob("**/*.*"))

    new_images = root.joinpath("new_images")
    new_images.joinpath("train").mkdir(parents=True, exist_ok=True)
    new_images.joinpath("val").mkdir(parents=True, exist_ok=True)
    new_labels = root.joinpath("new_labels")
    new_labels.joinpath("train").mkdir(parents=True, exist_ok=True)
    new_labels.joinpath("val").mkdir(parents=True, exist_ok=True)

    image_path: pathlib.Path
    label_path: pathlib.Path
    for idx, (image_path, label_path) in tqdm(
        enumerate(zip(image_paths, label_paths, strict=True)), total=len(image_paths), dynamic_ncols=True
    ):
        new_image_path = new_images.joinpath(
            image_path.parent.name, f"{str(idx).zfill(zfill_width)}{image_path.suffix}"
        )
        new_label_path = new_labels.joinpath(
            label_path.parent.name, f"{str(idx).zfill(zfill_width)}{label_path.suffix}"
        )
        image_path.rename(new_image_path)
        label_path.rename(new_label_path)

    shutil.rmtree(root.joinpath("images"))
    shutil.rmtree(root.joinpath("labels"))
    shutil.move(new_images, root.joinpath("images"))
    shutil.move(new_labels, root.joinpath("labels"))


if __name__ == "__main__":
    main()
