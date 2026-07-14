from pathlib import Path

from tqdm import tqdm

root = Path("../../data/passport-seg").resolve(strict=True)
dst_class_id = 1

label_paths = list(root.joinpath("labels").rglob("**/*.txt"))
for label_path in tqdm(label_paths, dynamic_ncols=True):
    label = label_path.read_text("utf-8")
    label_path.write_text(f"{dst_class_id}{label[1:]}", "utf-8")
