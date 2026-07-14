from __future__ import annotations

import pathlib

import numpy as np
from tqdm import tqdm

root = pathlib.Path("D:/data/camera").resolve(strict=True)
nc = 3


def bincount_class_ids(
    label_paths: list[pathlib.Path],
) -> tuple[np.ndarray, np.ndarray]:
    class_ids = []
    for label_path in tqdm(label_paths, dynamic_ncols=True):
        with open(label_path) as f:
            raw_label = f.readlines()
        for label_one in raw_label:
            class_id = int(label_one.split(" ")[0])
            class_ids.append(class_id)
    class_ids = np.array(class_ids)
    instances_per_class = np.bincount(class_ids, minlength=nc)
    total_instances = class_ids.size
    return instances_per_class, total_instances


def print_result(instances_per_class: np.ndarray):
    result_str = ""
    for i in range(nc):
        result_str += f"{i}: {instances_per_class[i]}\n"
    print(result_str, end="")


def main():
    train_label_paths = sorted(root.joinpath("labels", "train").rglob("**/*.txt"))
    val_label_paths = sorted(root.joinpath("labels", "val").rglob("**/*.txt"))

    train_instances_per_class, train_total_instances = bincount_class_ids(train_label_paths)
    val_instances_per_class, val_total_instances = bincount_class_ids(val_label_paths)

    print("trainset instances per class")
    print_result(train_instances_per_class)
    print(f"trainset total instances: {train_total_instances}\n")

    print("valset instances per class")
    print_result(val_instances_per_class)
    print(f"valset total instances: {val_total_instances}")


if __name__ == "__main__":
    main()
