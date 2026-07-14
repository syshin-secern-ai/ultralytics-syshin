# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip

from .base import BaseBackend


class RKNNBackend(BaseBackend):
    """Rockchip RKNN inference backend for Rockchip NPU hardware.

    Loads and runs inference with RKNN models (.rknn files) using the RKNN-Toolkit-Lite2 runtime. Only supported on
    Rockchip devices with NPU hardware (e.g., RK3588, RK3566).
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Rockchip RKNN model from a .rknn file or model directory.

        Args:
            weight (str | Path): Path to the .rknn file or directory containing the model.

        Raises:
            OSError: If not running on a Rockchip device.
            RuntimeError: If model loading or runtime initialization fails.
        """
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference on the Rockchip NPU.

        Args:
            im (torch.Tensor): Input image tensor in BHWC format, normalized to [0, 1].

        Returns:
            (list): Model predictions as a list of output arrays.
        """
        h, w = im.shape[1:3]
        im = (im.cpu().numpy() * 255).astype("uint8")
        im = im if isinstance(im, (list, tuple)) else [im]
        y = self.model.inference(inputs=im)
        if y[0].ndim == 4:  # rknn_raw per-stride raw outputs (single-output models emit one BCN tensor)
            return [self._decode_stride(y, h)]
        # INT8 exports use input-relative coordinates so a single per-tensor scale preserves class scores.
        if (
            self.metadata.get("args", {}).get("quantize") == 8
            and self.task in {"detect", "segment", "pose", "obb"}
            and not self.end2end
        ):
            kpt_start = 4 + len(self.names)  # pose keypoints follow the box (4) and class-score (nc) channels
            for x in y:
                if x.ndim == 3:
                    x[:, [0, 2]] *= w
                    x[:, [1, 3]] *= h
                    if self.task == "pose":
                        x[:, kpt_start::3] *= w
                        x[:, kpt_start + 1 :: 3] *= h
        return y

    def _decode_stride(self, y: list, imgsz: int) -> torch.Tensor:
        """Decode ``rknn_raw`` per-stride raw outputs into the standard (bs, 4+nc+nk, anchors) inference tensor.

        Output order matches the ONNX export: bbox_s8, cls_s8, bbox_s16, cls_s16, bbox_s32, cls_s32, then kpt_s8,
        kpt_s16, kpt_s32 for pose models (detect ends after bbox/cls). Strides are derived from each bbox feature map
        size. cls and keypoint visibility are already sigmoid-activated on-graph, and bbox is raw ltrb grid-unit
        distances (reg_max=1, no DFL). The math mirrors ``Detect._get_decode_boxes`` / ``Pose26.kpts_decode``, so the
        result feeds ``non_max_suppression`` like any non-export inference output.
        """
        from ultralytics.utils.tal import dist2bbox, make_anchors

        nl = len(y) // (3 if self.task == "pose" else 2)  # number of stride levels
        bbox = [torch.from_numpy(y[2 * i]) for i in range(nl)]
        cls = [torch.from_numpy(y[2 * i + 1]) for i in range(nl)]
        anchors, strides = (a.transpose(0, 1) for a in make_anchors(bbox, [imgsz // b.shape[2] for b in bbox], 0.5))
        dbox = dist2bbox(torch.cat([b.flatten(2) for b in bbox], 2), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        pred = [dbox, torch.cat([c.flatten(2) for c in cls], 2)]
        if self.task == "pose":
            bs = y[0].shape[0]
            k = torch.cat([torch.from_numpy(o).flatten(2) for o in y[2 * nl :]], 2).view(bs, *self.kpt_shape, -1)
            xy = (k[:, :, :2] + anchors) * strides
            pred.append(torch.cat([xy, k[:, :, 2:3]], 2).view(bs, self.kpt_shape[0] * self.kpt_shape[1], -1))
        return torch.cat(pred, 1)
