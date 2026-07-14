from pathlib import Path

from tqdm import tqdm

src_root = Path("../../data/home-collected-paper-v0").resolve(strict=True)
compare_root = Path("../../data/home-collected-paper-v3").resolve(strict=True)


def main():
    image_paths = sorted(src_root.rglob("**/*.jpg"))
    for image_path in tqdm(image_paths, dynamic_ncols=True):
        if (compare_root / "images" / "train" / image_path.name).exists() or (
            compare_root / "images" / "val" / image_path.name
        ).exists():
            image_path.unlink()


if __name__ == "__main__":
    main()
