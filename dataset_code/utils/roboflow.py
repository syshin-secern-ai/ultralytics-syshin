import pathlib

from tqdm import tqdm

root = pathlib.Path("../../data/bezel").resolve(strict=True)


def main():
    label_paths = sorted((root / "labels").rglob("**/*.txt"))

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        new_name = label_path.stem.split("_jpg")[0] + ".txt"
        label_path.rename(label_path.parent / new_name)


if __name__ == "__main__":
    main()
