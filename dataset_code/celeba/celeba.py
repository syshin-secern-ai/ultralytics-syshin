import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from standalone.code.face_aligner import FaceAligner
from tqdm import tqdm

fa = FaceAligner()
data_dir = Path("../../data/img_align_celeba")
result_dir = Path("../../data/img_fr_celeba")
landmark_data = pd.DataFrame(
    [
        {
            "filename": line.split()[0],
            "landmark": np.array(line.split()[1:]).astype(np.int32).reshape((5, 2)),
        }
        for line in Path("../../data/list_landmarks_align_celeba.txt").read_text().splitlines()[2:]
    ]
)
landmark_data = landmark_data.set_index("filename")

shutil.rmtree(result_dir, ignore_errors=True)
result_dir.mkdir(parents=True, exist_ok=True)

for img_path in tqdm(list(data_dir.glob("*.jpg")), dynamic_ncols=True):
    img = cv2.imread(str(img_path))
    kpts = landmark_data.loc[img_path.name, "landmark"]
    aligned_face_img = fa.align(img, kpts)
    cv2.imwrite(str(result_dir / img_path.name), aligned_face_img)
