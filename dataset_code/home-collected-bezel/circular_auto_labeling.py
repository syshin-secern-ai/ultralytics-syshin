from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm


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


def move_invalid_image(src_image_path: Path, invalid_dir: Path, sub_dir_name: str):
    dst_image_path = invalid_dir / src_image_path.parent.name / sub_dir_name / src_image_path.name
    dst_image_path.parent.mkdir(parents=True, exist_ok=True)
    src_image_path.rename(dst_image_path)


def is_small_box_inside_large_box(large_box: np.ndarray, small_box: np.ndarray, pad=0) -> bool:
    x1, y1, x2, y2 = large_box
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    r_x1, r_y1, r_x2, r_y2 = small_box
    return bool(r_x1 >= x1 and r_y1 >= y1 and r_x2 <= x2 and r_y2 <= y2)


@ray.remote(num_gpus=1)
def worker(
    image_paths: list[Path],
    label_dir: Path,
    invalid_dir: Path,
    plot_dir: Path,
    fd_path: Path,
    bd_path: Path,
    fd_conf: float,
    bd_conf: float,
):
    from ultralytics import YOLO

    fd = YOLO(fd_path, "pose")
    bd = YOLO(bd_path, "detect")

    for image_path in tqdm(image_paths):
        fd_results = fd.predict(image_path, conf=fd_conf, verbose=False)
        bd_results = bd.predict(image_path, conf=bd_conf, verbose=False)
        for fd_result, bd_result in zip(fd_results, bd_results):
            # 얼굴 없는 사진 필터링
            if fd_result.boxes.shape[0] == 0:
                move_invalid_image(image_path, invalid_dir, "no_face")
                continue

            # 베젤 없는 사진 필터링
            if bd_result.boxes.shape[0] == 0:
                move_invalid_image(image_path, invalid_dir, "no_bezel")
                continue

            # 베젤 박스 안에 얼굴 박스가 있지 않는 사진 필터링
            face_in_bezel_index = []
            for fd_box in fd_result.boxes.xyxy:
                for i, bd_box in enumerate(bd_result.boxes.xyxy):
                    if is_small_box_inside_large_box(bd_box.cpu().numpy(), fd_box.cpu().numpy(), pad=5):
                        face_in_bezel_index.append(i)
            if len(face_in_bezel_index) == 0:
                move_invalid_image(image_path, invalid_dir, "no_face_in_bezel")
                continue

            # 베젤 박스가 2개 이상이면 가장 작은 면적의 박스만 남김
            bd_result = bd_result[face_in_bezel_index]
            if bd_result.boxes.shape[0] > 1:
                areas = bd_result.boxes.xywh[:, 2] * bd_result.boxes.xywh[:, 3]
                min_area_index = torch.argmin(areas)
                bd_result = bd_result[min_area_index]

            # 라벨링
            save_txt(
                bd_result,
                label_dir / image_path.parent.name / image_path.with_suffix(".txt").name,
            )

            # plot
            plotted_img = fd_result.plot(line_width=2, kpt_radius=3)
            plotted_img = bd_result.plot(line_width=2, kpt_radius=3, img=plotted_img)
            cv2.imwrite(plot_dir / image_path.parent.name / image_path.name, plotted_img)


def main():
    # Args
    ray_address = "ray://172.100.100.10:33333"
    root = Path("../../data/home-collected-bezel-v4").resolve(strict=True)
    label_dir = root / "labels"
    invalid_dir = root / "invalid"
    plot_dir = root / "plots"
    fd_path = Path("runs/pose/autolabeler/weights/last.pt")
    bd_path = Path("runs/detect/train4/weights/last.pt")
    fd_conf = 0.25
    bd_conf = 0.25

    # Path initialize
    print("Path initialize...")
    shutil.rmtree(label_dir, ignore_errors=True)
    shutil.rmtree(plot_dir, ignore_errors=True)
    (label_dir / "train").mkdir(parents=True, exist_ok=True)
    (label_dir / "val").mkdir(parents=True, exist_ok=True)
    (plot_dir / "train").mkdir(parents=True, exist_ok=True)
    (plot_dir / "val").mkdir(parents=True, exist_ok=True)
    print("Done.")

    # Ray initialize
    ray.init(ray_address)
    assert ray.is_initialized()
    assert ray.available_resources()["GPU"].is_integer()
    num_gpus = int(ray.available_resources()["GPU"])

    # Split the source across 'n' GPUs.
    print(f"Split the source across '{num_gpus}' GPUs...")
    print("This may take a long time depending on the number of images.")
    train_img_paths = list((root / "images" / "train").glob("*.jpg"))
    val_img_paths = list((root / "images" / "val").glob("*.jpg"))
    img_paths = sorted(train_img_paths + val_img_paths)
    sub_img_paths: list[list[Path]] = [i.tolist() for i in np.array_split(img_paths, num_gpus)]
    print("Done.")
    print(f"Trainset: {len(train_img_paths):,}")
    print(f"Valset: {len(val_img_paths):,}")
    print(f"All: {(len(img_paths)):,}")

    # Labeling
    print("Labeling...")
    ray.get(
        [
            worker.remote(
                s,
                label_dir,
                invalid_dir,
                plot_dir,
                fd_path,
                bd_path,
                fd_conf,
                bd_conf,
            )
            for s in sub_img_paths
        ]
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
