import shutil
from pathlib import Path

from tqdm import tqdm

src_root = Path("../../data/save_data").resolve(strict=True)
dst_root = Path("../../data/home-collected-bezel").resolve()


def main():
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True)
    train_image_dir = dst_root / "images" / "train"
    train_image_dir.mkdir(parents=True)
    (dst_root / "images" / "val").mkdir(parents=True)
    (dst_root / "labels" / "train").mkdir(parents=True)
    (dst_root / "labels" / "val").mkdir(parents=True)

    image_paths = sorted(src_root.rglob("**/*.jpg"))
    for image_path in tqdm(image_paths, dynamic_ncols=True):
        dst_name = f"{image_path.parent.parent.name}_{image_path.parent.name.replace('fas_', '')}_{image_path.name}"
        dst_path = train_image_dir / dst_name
        image_path.rename(dst_path)


if __name__ == "__main__":
    main()
