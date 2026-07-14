from pathlib import Path

from tqdm import tqdm

root = Path("../../data/home-collected-face-person").resolve(strict=True)


def main():
    label_paths = sorted((root / "labels").rglob("*.txt"))
    for label_path in tqdm(label_paths, dynamic_ncols=True):
        label = label_path.read_text().splitlines()
        class_id = [int(label_one.split()[0]) for label_one in label]

        num_face = class_id.count(0)
        num_person = class_id.count(1)
        if num_face >= 2:
            print(f"{label_path.name} {num_face=}.")
        if num_person == 0 or num_person >= 2:
            print(f"{label_path.name} {num_person=}.")


if __name__ == "__main__":
    main()
