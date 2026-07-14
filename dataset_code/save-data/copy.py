import shutil
from pathlib import Path

from tqdm import tqdm

src_root = Path("../../data/save_data").resolve(strict=True)
dst_root = Path("../../data/detect_wrong_faceperson").resolve()


def main():
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True)

    user_dirs: list[Path] = sorted(src_root.glob("*"))
    for user_dir in tqdm(user_dirs, dynamic_ncols=True):
        src_dir = user_dir / "detect_wrong" / "other_people"
        shutil.copytree(src_dir, dst_root / user_dir.name)


if __name__ == "__main__":
    main()
