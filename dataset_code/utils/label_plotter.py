from __future__ import annotations

import argparse
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import check_det_dataset, find_dataset_yaml
from ultralytics.engine.results import OBB, Boxes, Keypoints, Masks, Probs
from ultralytics.utils.ops import segments2boxes, xywh2xyxy, xywhn2xyxy, xyxyxyxy2xywhr
from ultralytics.utils.plotting import Annotator, colors


class Plotter:
    def __init__(
        self,
        data: dict,
        task: str,
        image_path: Path,
        label_path: Path,
    ):
        self.data = data
        self.task = task
        self.image_path = image_path
        self.label_path = label_path

        self.names: dict[int, str] = self.data["names"]
        self.path = str(self.image_path)
        self.orig_img = cv2.imread(str(self.image_path))
        self.orig_label = self.label_path.read_text().splitlines()
        self.orig_shape: tuple[int, int] = self.orig_img.shape[:2]

        self.boxes: Boxes | None = None
        self.masks: Masks | None = None
        self.probs: Probs | None = None
        self.keypoints: Keypoints | None = None
        self.obb: OBB | None = None

        # Convert label
        if len(self.orig_label) > 0:
            pytype_label = [[float(i) for i in label_one.split(" ")] for label_one in self.orig_label]
            if self.task == "detect":
                label = torch.tensor(pytype_label)
                assert label.shape[-1] == 5, "label does not conform to YOLO Detect format."
                xyxy = xywhn2xyxy(label[..., 1:5], self.orig_shape[1], self.orig_shape[0])
                cls = label[..., 0:1]
                conf = torch.ones_like(cls)
                self.boxes = Boxes(torch.cat((xyxy, conf, cls), dim=-1), self.orig_shape)

            elif self.task == "segment":
                label = [torch.tensor(i) for i in pytype_label]
                # Unnomalize
                for label_one in label:
                    label_one[1::2] *= self.orig_shape[1]
                    label_one[2::2] *= self.orig_shape[0]

                segments = [label_one[1:].reshape(-1, 2) for label_one in label]
                boxes = torch.tensor(segments2boxes(segments))
                boxes = xywh2xyxy(boxes)
                cls = torch.tensor([i[0] for i in label]).unsqueeze(1)
                conf = torch.ones_like(cls)
                self.boxes = Boxes(torch.cat((boxes, conf, cls), dim=-1), self.orig_shape)

                masks = np.zeros((len(label), *self.orig_shape), dtype=np.float32)
                for mask, segment in zip(masks, segments, strict=True):
                    cv2.fillPoly(mask, [segment.round().to(torch.int32).numpy()], [1])
                self.masks = Masks(torch.tensor(masks), self.orig_shape)

            elif self.task == "pose":
                label = torch.tensor(pytype_label)
                xyxy = xywhn2xyxy(label[..., 1:5], self.orig_shape[1], self.orig_shape[0])
                cls = label[..., 0:1]
                conf = torch.ones_like(cls)
                self.boxes = Boxes(torch.cat((xyxy, conf, cls), dim=-1), self.orig_shape)
                keypoints = label[..., 5:].reshape(-1, *data["kpt_shape"])
                keypoints[..., 0] *= self.orig_shape[1]
                keypoints[..., 1] *= self.orig_shape[0]
                self.keypoints = Keypoints(keypoints, self.orig_shape)

            elif self.task == "obb":
                label = torch.tensor(pytype_label)
                assert label.shape[-1] == 9, "label does not conform to YOLO OBB format."
                cls = label[..., 0:1]
                conf = torch.ones_like(cls)
                xywhr = label[..., 1:9]
                xywhr[..., 0::2] *= self.orig_shape[1]
                xywhr[..., 1::2] *= self.orig_shape[0]
                xywhr = xyxyxyxy2xywhr(xywhr)
                self.obb = OBB(torch.cat((xywhr, conf, cls), dim=-1), self.orig_shape)

    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | None = None,
        im_gpu: torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Plot detection results on an input BGR image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | None): Image to plot on. If None, uses original image.
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

        Returns:
            (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = int(d.cls), float(d.conf) if conf else None, int(d.id.item()) if d.is_track else None
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))  # RGBA box

        # Plot Pose results
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.result(pil)


def main(root: Path, task: str, output_dir: Path | None = None):
    assert root.is_dir(), "root does not exist or is not a directory."

    # Process paths
    yaml_path = find_dataset_yaml(root)
    data = check_det_dataset(str(yaml_path), autodownload=False)
    image_paths: list[Path] = sorted((root / "images").rglob("**/*.*"))
    label_paths: list[Path] = sorted((root / "labels").rglob("**/*.txt"))

    # Initialize output directory
    output_dir = root / "plots" if output_dir is None else output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)

    # Plot label and save plotted image
    for image_path, label_path in tqdm(
        zip(image_paths, label_paths, strict=True), total=len(image_paths), dynamic_ncols=True
    ):
        assert image_path.stem == label_path.stem, "image and label do not match."

        plotter = Plotter(data, task, image_path, label_path)
        plotter.plot(
            conf=False,
            line_width=2,
            kpt_radius=2,
            labels=False,
            save=True,
            filename=str(output_dir.joinpath(*image_path.parts[-2:])),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=Path)
    opt = parser.parse_args()

    main(opt.root, opt.task, opt.output_dir)
