import pathlib
import shutil

from tqdm import tqdm

face_root = pathlib.Path("/raid/objects365_face").resolve(strict=True)
person_root = pathlib.Path("/raid/objects365_person").resolve(strict=True)
faceperson_root = pathlib.Path("/raid/objects365_faceperson").resolve()


def process_images():
    person_image_paths = sorted(person_root.joinpath("images").rglob("**/*.jpg"))
    for person_image_path in tqdm(person_image_paths, "Process images", dynamic_ncols=True):
        person_image_path.rename(faceperson_root.joinpath(*person_image_path.parts[-3:]))


def process_labels():
    person_label_paths = sorted(person_root.joinpath("labels").rglob("**/*.txt"))
    for person_label_path in tqdm(person_label_paths, "Process labels", dynamic_ncols=True):
        faceperson_label_path = faceperson_root.joinpath(*person_label_path.parts[-3:])
        face_label_path = face_root.joinpath(*person_label_path.parts[-3:])

        if face_label_path.is_file():
            face_label = face_label_path.read_text("utf-8")
            faceperson_label_path.write_text(face_label, "utf-8")

        person_label = person_label_path.read_text("utf-8")
        with open(faceperson_label_path, "a", encoding="utf-8") as f:
            f.write(person_label)


def main():
    if faceperson_root.exists():
        print("faceperson_root가 존재하여 삭제하고 진행합니다.")
        shutil.rmtree(faceperson_root)

    faceperson_root.mkdir()
    faceperson_root.joinpath("images", "train").mkdir(parents=True)
    faceperson_root.joinpath("images", "val").mkdir(parents=True)
    faceperson_root.joinpath("labels", "train").mkdir(parents=True)
    faceperson_root.joinpath("labels", "val").mkdir(parents=True)

    process_images()
    process_labels()


if __name__ == "__main__":
    main()
