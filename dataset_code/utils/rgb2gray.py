import pathlib

import cv2
from tqdm import tqdm

root = pathlib.Path("D:/data/data").resolve(strict=True)


def main():
    img_paths = sorted(root.rglob("**/*.jpg"))

    for img_path in tqdm(img_paths, dynamic_ncols=True):
        img = cv2.imread(str(img_path))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(img_path), gray_img)


if __name__ == "__main__":
    main()
