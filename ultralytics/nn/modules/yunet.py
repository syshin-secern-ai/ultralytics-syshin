# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YuNet building blocks ported from libfacedetection.train (https://github.com/ShiqiYu/libfacedetection.train).

Submodule attribute names (conv1, conv2, bn, relu) are kept identical to the original
implementation so that pretrained YuNet checkpoints can be transferred by state_dict key mapping.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .head import Pose26

__all__ = ("Add", "Conv4LayerBlock", "ConvDPUnit", "YuNetPoseHead", "YuNetStem")


class YuNetStem(nn.Module):
    """YuNet stem (Conv_head in the original code): 3x3 stride-2 conv + BN + ReLU + ConvDPUnit."""

    def __init__(self, c1: int, cm: int, c2: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, cm, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(cm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ConvDPUnit(cm, c2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu1(self.bn1(self.conv1(x))))


class Conv4LayerBlock(nn.Module):
    """YuNet backbone stage (Conv4layerBlock in the original code): two stacked ConvDPUnits."""

    def __init__(self, c1: int, c2: int, bn_relu: bool = True):
        super().__init__()
        self.conv1 = ConvDPUnit(c1, c1, True)
        self.conv2 = ConvDPUnit(c1, c2, bn_relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class YuNetPoseHead(Pose26):
    """Pose26-compatible head using the original YuNet head structure (libfacedetection.train).

    Per level: optional shared ConvDPUnit stack, then one plain ConvDPUnit (no BN/ReLU) per
    branch, exactly like YuNetHead. Output semantics follow YOLO26 instead of the original:
    - no objectness branch (cls score is the unified confidence)
    - box branch regresses ltrb distances (dist2bbox), requires reg_max=1
    - keypoint branch outputs nk = kpts * (x, y, visibility); a sigma branch is kept for RLE

    Loss (PoseLoss26/TAL), decode, and validation are inherited unchanged from Pose26.
    """

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (17, 3),
        shared_convs: int = 1,
        reg_max=1,
        end2end=False,
        ch: tuple = (),
    ):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        # Shared per-level stack (yunet_n: 1, yunet_s: 0; empty Sequential acts as identity).
        self.shared = nn.ModuleList(nn.Sequential(*(ConvDPUnit(x, x, True) for _ in range(shared_convs))) for x in ch)
        # Rebuild branches in the original YuNet style (replaces parent-built modules).
        self.cv2 = nn.ModuleList(ConvDPUnit(x, 4 * self.reg_max, False) for x in ch)  # box
        self.cv3 = nn.ModuleList(ConvDPUnit(x, self.nc, False) for x in ch)  # cls
        self.cv4 = nn.ModuleList(nn.Identity() for _ in ch)  # pose feature = shared feature
        self.cv4_kpts = nn.ModuleList(ConvDPUnit(x, self.nk, False) for x in ch)
        self.cv4_sigma = nn.ModuleList(ConvDPUnit(x, self.nk_sigma, False) for x in ch)
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    def forward(self, x: list[torch.Tensor]):
        """Perform forward pass through the YuNet Pose head and return predictions."""
        if self.export and self.format == "rknn_raw":
            # The raw per-stride export branch applies branch heads directly (bypassing forward_head),
            # so apply the shared stack here; forward_head is not called on that path.
            x = [self.shared[i](x[i]) for i in range(self.nl)]
        return super().forward(x)

    def forward_head(self, x: list[torch.Tensor], **heads) -> dict[str, torch.Tensor]:
        """Apply the shared ConvDPUnit stack, then the standard Pose26 branch heads."""
        x = [self.shared[i](x[i]) for i in range(self.nl)]
        return super().forward_head(x, **heads)

    def bias_init(self):
        """Detect.bias_init adapted to ConvDPUnit branches (final conv is .conv2, not [-1])."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a.conv2.bias.data[:] = 2.0  # box
            b.conv2.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)  # cls
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a.conv2.bias.data[:] = 2.0
                b.conv2.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class ConvDPUnit(nn.Module):
    """1x1 pointwise conv + 3x3 depthwise conv, optionally followed by BN + ReLU."""

    def __init__(self, c1: int, c2: int, bn_relu: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=True, groups=c2)
        self.withBNRelu = bn_relu
        if bn_relu:
            self.bn = nn.BatchNorm2d(c2)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(x))
        if self.withBNRelu:
            x = self.relu(self.bn(x))
        return x


class Add(nn.Module):
    """Elementwise sum of a list of same-shape tensors (used for TFPN top-down fusion)."""

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        out = x[0]
        for t in x[1:]:
            out = out + t
        return out
