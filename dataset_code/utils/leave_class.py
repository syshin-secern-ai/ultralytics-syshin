import pathlib

from tqdm import tqdm

root = pathlib.Path("/raid/objects365").resolve(strict=True)
leave_class_id = 0


def main():
    label_paths = sorted(root.joinpath("labels").rglob("**/*.txt"))

    for label_path in tqdm(label_paths, dynamic_ncols=True):
        with open(label_path) as f:
            raw_label = f.readlines()

        label = []
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            if class_id == leave_class_id:
                label.append(label_one)

        with open(label_path, "w") as f:
            f.writelines(label)


if __name__ == "__main__":
    main()
