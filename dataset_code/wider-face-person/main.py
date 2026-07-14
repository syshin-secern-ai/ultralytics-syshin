from functools import reduce
from pathlib import Path

import yaml
from tqdm import tqdm

from ultralytics import YOLO

widerface_root = Path("../../data/widerface").resolve(strict=True)
wider_face_person_root = Path("../../data/wider-face-person").resolve()
face_class = 0
person_class = 1
kpt_shape = [5, 3]
conf = 0.25
plot = False


def main():
    widerface_root.rename(wider_face_person_root)
    wider_face_person_images_dir = wider_face_person_root / "images"

    # Initialize model
    model = YOLO("yolo11x.pt")
    results = model.predict(
        wider_face_person_images_dir / "**/*.jpg",
        classes=[0],
        conf=conf,
        iou=0.7,
        batch=64,
        stream=True,
        verbose=False,
    )

    # Auto labeling
    kpt = " 0.000000" * reduce(lambda x, y: x * y, kpt_shape)
    for result in tqdm(
        results,
        "Auto labeling",
        total=len(list(wider_face_person_images_dir.rglob("**/*.jpg"))),
        dynamic_ncols=True,
    ):
        person_label = [
            f"{person_class} {i[0]:.6f} {i[1]:.6f} {i[2]:.6f} {i[3]:.6f}{kpt}\n" for i in result.boxes.xywhn
        ]
        label_path = (
            wider_face_person_root
            / "labels"
            / Path(result.path).parent.name
            / Path(result.path).with_suffix(".txt").name
        )
        with open(label_path, "a") as f:
            f.writelines(person_label)

        if plot:
            filename = wider_face_person_root.joinpath("plots", *Path(result.path).parts[-2:])
            filename.parent.mkdir(parents=True, exist_ok=True)
            result.plot(
                line_width=2,
                kpt_radius=3,
                color_mode="instance",
                save=True,
                filename=str(filename),
            )

    # Create yaml file
    metadata = {
        "path": "../../data/wider-face-person",
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": kpt_shape,
        "flip_idx": [1, 0, 2, 4, 3],
        "names": {
            face_class: "face",
            person_class: "person",
        },
    }
    with open(wider_face_person_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


if __name__ == "__main__":
    main()
