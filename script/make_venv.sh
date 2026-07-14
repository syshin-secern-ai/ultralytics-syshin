#!/bin/bash

set -euo pipefail

uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e .
uv pip install nncf onnx onnxruntime-gpu==1.26.0 onnxslim openvino ruff scipy tqdm wandb
uv pip install ray
ln -sfn /usr/share/fonts/truetype/dejavu "$(.venv/bin/python -c 'import cv2, pathlib; print(pathlib.Path(cv2.__file__).parent / "qt")')/fonts"
uv cache clean
uv pip list --outdated