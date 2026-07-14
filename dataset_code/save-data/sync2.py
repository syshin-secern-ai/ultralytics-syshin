from pathlib import Path

from tqdm import tqdm

root = Path("../../data/detect_wrong_faceperson").resolve(strict=True)


def main():
    original_dir = root / "original"
    plot_dir = root / "plot"

    original_images: list[Path] = sorted(original_dir.glob("*.*"))
    for original_image in tqdm(original_images, dynamic_ncols=True):
        if not (plot_dir / original_image.name).exists():
            original_image.unlink()


if __name__ == "__main__":
    main()
