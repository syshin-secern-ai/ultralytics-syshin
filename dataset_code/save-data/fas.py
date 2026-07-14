import shutil
from pathlib import Path

from tqdm import tqdm

root = Path("../../data/save_data").resolve(strict=True)
allowed_dirs = ["fas_smartphone", "fas_tablet"]

user_dirs = sorted(root.glob("*"))

for user_dir in tqdm(user_dirs, dynamic_ncols=True):
    for subdir in sorted(user_dir.glob("*")):
        if subdir.name not in allowed_dirs:
            shutil.rmtree(subdir)
