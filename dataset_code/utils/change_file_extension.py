import pathlib

from tqdm import tqdm

root = pathlib.Path("D:/data/camera").resolve(strict=True)


def main():
    image_paths = sorted(root.joinpath("images").rglob("**/*.jpeg"))
    for image_path in tqdm(image_paths, dynamic_ncols=True):
        image_path.rename(image_path.with_suffix(".jpg"))


if __name__ == "__main__":
    main()
