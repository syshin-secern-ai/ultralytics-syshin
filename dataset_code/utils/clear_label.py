from pathlib import Path

from tqdm import tqdm

root = Path("D:/data/face-mix").resolve(strict=True)


def main():
    label_paths: list[Path] = sorted(root.joinpath("labels").rglob("**/*.txt"))

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        label_path.write_text("")


if __name__ == "__main__":
    main()
