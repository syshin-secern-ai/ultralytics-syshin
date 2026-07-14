from pathlib import Path

from tqdm import tqdm

root = Path("../../data/home-collected-face-person").resolve(strict=True)


def main():
    image_paths: list[Path] = sorted(root.joinpath("images").rglob("**/*.*"))

    for image_path in tqdm(image_paths, dynamic_ncols=True):
        label_path = root / "labels" / image_path.parts[-2] / image_path.with_suffix(".txt").name
        label_path.touch()


if __name__ == "__main__":
    main()
