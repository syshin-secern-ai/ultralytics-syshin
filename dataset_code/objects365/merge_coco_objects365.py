import pathlib
import shutil

from tqdm import tqdm

coco_root = pathlib.Path("/raid/coco-face-person")
objects365_root = pathlib.Path("/raid/objects365_faceperson").resolve(strict=True)
dest_root = pathlib.Path("/raid/coco-objects365-faceperson").resolve()


def main():
    shutil.move(objects365_root, dest_root)

    coco_paths = sorted(coco_root.rglob("**/*.*"))
    for coco_path in tqdm(coco_paths, dynamic_ncols=True):
        coco_path.rename(dest_root.joinpath(*coco_path.parts[-3:]))


if __name__ == "__main__":
    main()
