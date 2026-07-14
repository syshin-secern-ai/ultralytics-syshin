from pathlib import Path

from tqdm import tqdm

root = Path("../../data/detect_wrong_faceperson").resolve(strict=True)


def main():
    user_dirs: list[Path] = sorted(root.glob("*"))
    for user_dir in tqdm(user_dirs, dynamic_ncols=True):
        original_dir = user_dir / "original"
        plot_dir = user_dir / "plot"

        original_files: list[Path] = sorted(original_dir.glob("*.*"))
        for original_file in original_files:
            if not (plot_dir / original_file.name).exists():
                original_file.unlink()


if __name__ == "__main__":
    main()
