import pathlib

from tqdm import tqdm

root = pathlib.Path("D:/data/annotated_img_label_250326/zero-person").resolve(strict=True)


def main():
    label_paths: list[pathlib.Path] = sorted(root.joinpath("labels").rglob("**/*.txt"))

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        with open(label_path) as f:
            raw_label = f.readlines()

        new_label = []
        for label_one in raw_label:
            cls = f"{int(label_one.split(' ')[0])}"
            element = [f"{float(i):.6f}" for i in label_one.split(" ")[1:]]
            new_label.append(f"{cls} {' '.join(element)}{' 0.000000' * 15}")

        with open(label_path, "w") as f:
            f.writelines(new_label)


if __name__ == "__main__":
    main()
