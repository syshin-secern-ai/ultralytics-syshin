import pathlib
import shutil

from tqdm import tqdm

root = pathlib.Path("/raid/objects365").resolve(strict=True)
extract_class_id = 0


def main():
    label_paths = sorted(root.joinpath("labels").rglob("**/*.txt"))

    new_images = root.joinpath("new_images")
    new_images.joinpath("train").mkdir(parents=True, exist_ok=True)
    new_images.joinpath("val").mkdir(parents=True, exist_ok=True)
    new_labels = root.joinpath("new_labels")
    new_labels.joinpath("train").mkdir(parents=True, exist_ok=True)
    new_labels.joinpath("val").mkdir(parents=True, exist_ok=True)

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        with open(label_path) as f:
            raw_label = f.readlines()

        exist_extract_class_id = False
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            if class_id == extract_class_id:
                exist_extract_class_id = True

        if exist_extract_class_id:
            image_path = root.joinpath("images", label_path.parent.name, f"{label_path.stem}.jpg")
            try:
                image_path.rename(new_images.joinpath(*image_path.parts[-2:]))
                label_path.rename(new_labels.joinpath(*label_path.parts[-2:]))
            except Exception as e:
                print(e)

    shutil.rmtree(root.joinpath("images"))
    shutil.rmtree(root.joinpath("labels"))
    shutil.move(new_images, root.joinpath("images"))
    shutil.move(new_labels, root.joinpath("labels"))


if __name__ == "__main__":
    main()
