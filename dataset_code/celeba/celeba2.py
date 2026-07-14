import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

data_dir = Path("../../data/celeba")
src_dir = data_dir / "img_fr_celeba"
dst_dir = data_dir / "images"
split_data = pd.DataFrame(
    [
        {
            "filename": line.split()[0],
            "split": int(line.split()[1]),
        }
        for line in (data_dir / "list_eval_partition.txt").read_text().splitlines()
    ]
)
split_data = split_data.set_index("filename")
print(split_data)
print(split_data.value_counts("split").sort_index())

for img_path in tqdm(list((src_dir).glob("*.jpg")), dynamic_ncols=True):
    split = split_data.loc[img_path.name, "split"]
    if split == 0 or split == 1:
        shutil.move(img_path, dst_dir / "train" / img_path.name)
    elif split == 2:
        shutil.move(img_path, dst_dir / "val" / img_path.name)
