from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import ray
import torch
from ray.experimental import tqdm_ray
from tqdm import tqdm


def save_txt(result, txt_file: str | Path):
    label = []
    for j, d in enumerate(result.boxes):
        label_one = ""
        cls = int(d.cls)
        box = d.xywhn.flatten()
        label_one += f"{cls} {' '.join([f'{i:.6f}' for i in box])}"
        if result.keypoints is not None:
            if result.keypoints.has_visible:
                if cls == 0:
                    kpt = torch.cat(
                        (
                            result.keypoints[j].xyn,
                            result.keypoints[j].conf.unsqueeze(2),
                        ),
                        2,
                    ).squeeze()
                    kpt[kpt[..., 2] < 0.25] = 0
                    kpt[..., 2][kpt[..., 2] > 0.25] = 2
                    kpt = kpt.flatten()
                elif cls == 1:
                    kpt = torch.zeros_like(result.keypoints[j].data).flatten()
                else:
                    raise ValueError(f"Invalid class id: {cls}")
            else:
                kpt = result.keypoints[j].xyn.flatten()
            label_one += f" {' '.join([f'{i:.6f}' for i in kpt])}"
        label.append(label_one)

    with open(txt_file, "a") as f:
        f.writelines([label_one + "\n" for label_one in label])


@ray.remote(num_gpus=1)
def worker(
    img_paths: list[Path],
    model_path: Path,
    task: str,
    conf: float,
    plot: bool,
):
    from ultralytics import YOLO

    model = YOLO(model_path, task)

    for img_path in tqdm_ray.tqdm(img_paths):
        results = model.predict(img_path, conf=conf, stream=True, verbose=False)
        for result in results:
            parts = list(Path(result.path).with_suffix(".txt").parts)
            parts[-3] = "labels"
            label_path = Path(*parts)
            save_txt(result, label_path)
            if plot:
                result.plot(
                    line_width=2,
                    kpt_radius=3,
                    save=True,
                    filename=result.path.replace("images", "plots"),
                )


def main(
    ray_address: str,
    root: Path,
    model_path: Path,
    task: str,
    conf: float,
    plot: bool,
    external_img_dir: Path,
    valset_ratio: float,
):
    if external_img_dir is None:
        external_img = False
    else:
        external_img = True

    # Make initial directories
    if external_img:
        shutil.rmtree(root, ignore_errors=True)
    else:
        shutil.rmtree(root / "labels", ignore_errors=True)
        shutil.rmtree(root / "plots", ignore_errors=True)
    root.joinpath("images", "train").mkdir(parents=True, exist_ok=True)
    root.joinpath("images", "val").mkdir(parents=True, exist_ok=True)
    root.joinpath("labels", "train").mkdir(parents=True)
    root.joinpath("labels", "val").mkdir(parents=True)
    if plot:
        root.joinpath("plots", "train").mkdir(parents=True)
        root.joinpath("plots", "val").mkdir(parents=True)

    # Make image symbolic links
    if external_img:
        print("Make image symbolic links...")
        print("This may take a long time depending on the number of images.")
        external_img_paths: list[Path] = list(external_img_dir.rglob("**/*.jpg"))
        random.shuffle(external_img_paths)
        num_val_img = round(len(external_img_paths) * valset_ratio)
        external_train_img_paths = external_img_paths[num_val_img:]
        external_val_img_paths = external_img_paths[:num_val_img]
        for img_path in tqdm(external_train_img_paths, "Trainset", dynamic_ncols=True):
            root.joinpath("images", "train", img_path.name).symlink_to(img_path)
        for img_path in tqdm(external_val_img_paths, "Valset", dynamic_ncols=True):
            root.joinpath("images", "val", img_path.name).symlink_to(img_path)
        print("Done.")
    else:
        external_train_img_paths = None
        external_val_img_paths = None

    # Ray initialize
    ray.init(ray_address)
    assert ray.is_initialized()
    assert ray.available_resources()["GPU"].is_integer()
    num_gpus = int(ray.available_resources()["GPU"])

    # Split the source across 'n' GPUs.
    print(f"Split the source across '{num_gpus}' GPUs...")
    print("This may take a long time depending on the number of images.")
    if external_img:
        train_img_paths = [root / "images" / "train" / i.name for i in external_train_img_paths]
        val_img_paths = [root / "images" / "val" / i.name for i in external_val_img_paths]
    else:
        train_img_paths: list[Path] = list(root.joinpath("images", "train").glob("*.jpg"))
        val_img_paths: list[Path] = list(root.joinpath("images", "val").glob("*.jpg"))
    img_paths = sorted(train_img_paths + val_img_paths)
    sub_img_paths: list[list[Path]] = [i.tolist() for i in np.array_split(img_paths, num_gpus)]
    print("Done.")
    print(f"Trainset: {len(train_img_paths):,}")
    print(f"Valset: {len(val_img_paths):,}")
    print(f"All: {(len(img_paths)):,}")

    # DEBUG
    # worker(img_paths, model_path, task, conf, False)

    # Process
    ray.get([worker.remote(s, model_path, task, conf, plot) for s in sub_img_paths])
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray_address", type=str, default="ray://172.100.100.34:10001")
    parser.add_argument("--root", type=Path, default="../../data/coco-face")
    parser.add_argument("--model_path", type=Path, default="runs/pose/autolabeler/weights/last.pt")
    parser.add_argument("--task", type=str, default="pose")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--external_img_dir", type=Path)
    parser.add_argument("--valset_ratio", type=float, default=0.01)
    opt = parser.parse_args()

    main(
        opt.ray_address,
        opt.root,
        opt.model_path,
        opt.task,
        opt.conf,
        opt.plot,
        opt.external_img_dir,
        opt.valset_ratio,
    )
