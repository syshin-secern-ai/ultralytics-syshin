import shutil
from pathlib import Path

from tqdm import tqdm

root = Path("../../data/detect_wrong_faceperson").resolve(strict=True)


def main():
    dst_original_dir = root / "original"
    dst_plot_dir = root / "plot"

    user_dirs: list[Path] = sorted(root.glob("*"))
    dst_original_dir.mkdir()
    dst_plot_dir.mkdir()

    for user_dir in tqdm(user_dirs, dynamic_ncols=True):
        original_dir = user_dir / "original"
        plot_dir = user_dir / "plot"

        original_images: list[Path] = sorted(original_dir.glob("*.*"))
        for original_image in original_images:
            name = f"{user_dir.name.replace('.', '')}_{original_image.name}"
            original_image.rename(dst_original_dir / name)

        plot_images: list[Path] = sorted(plot_dir.glob("*.*"))
        for plot_image in plot_images:
            name = f"{user_dir.name.replace('.', '')}_{plot_image.name}"
            plot_image.rename(dst_plot_dir / name)

        shutil.rmtree(user_dir)


if __name__ == "__main__":
    main()
