import shutil
from functools import reduce
from pathlib import Path

import yaml
from tqdm import tqdm

coco_face_root = Path("../../data/coco-face").resolve(strict=True)
coco_person_root = Path("../../data/coco-person").resolve(strict=True)
coco_face_person_root = Path("../../data/coco-face-person").resolve()
face_class = 0
person_class = 1
kpt_shape = [5, 3]


def process_images():
    coco_face_person_root.mkdir(parents=True)
    shutil.move(coco_face_root / "images", coco_face_person_root)

    # Remove useless files
    shutil.rmtree(coco_person_root / "images")


def process_labels():
    (coco_face_person_root / "labels" / "train").mkdir(parents=True)
    (coco_face_person_root / "labels" / "val").mkdir(parents=True)

    face_label_paths: list[Path] = sorted((coco_face_root / "labels").rglob("**/*.txt"))
    person_label_paths: list[Path] = sorted((coco_person_root / "labels").rglob("**/*.txt"))

    for face_label_path, person_label_path in tqdm(
        zip(face_label_paths, person_label_paths, strict=True),
        "Process labels",
        total=len(face_label_paths),
        dynamic_ncols=True,
    ):
        assert face_label_path.name == person_label_path.name

        # Read label
        face_label = face_label_path.read_text().splitlines()
        person_label = person_label_path.read_text().splitlines()

        # Convert label
        face_label = [f"{face_class} {' '.join(i.split()[1:])}" for i in face_label]
        kpt = " 0.000000" * reduce(lambda x, y: x * y, kpt_shape)
        person_label = [f"{person_class} {' '.join(i.split()[1:])}{kpt}" for i in person_label]
        face_person_label = face_label + person_label

        # Write label
        face_person_label_path = coco_face_person_root.joinpath(*face_label_path.parts[-3:])
        face_person_label_path.write_text("\n".join(face_person_label) + "\n")

    # Remove useless files
    shutil.rmtree(coco_face_root)
    shutil.rmtree(coco_person_root)


def create_yaml_file():
    metadata = {
        "path": "../../data/coco-face-person",
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": kpt_shape,
        "flip_idx": [1, 0, 2, 4, 3],
        "names": {
            face_class: "face",
            person_class: "person",
        },
    }
    with open(coco_face_person_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def main():
    shutil.rmtree(coco_face_person_root, ignore_errors=True)
    process_images()
    process_labels()
    create_yaml_file()


if __name__ == "__main__":
    main()
