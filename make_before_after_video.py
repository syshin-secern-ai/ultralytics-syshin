import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2

from ultralytics import YOLO


def render(model_path: Path, task: str, source: Path, output: Path, fps: float, conf: float) -> None:
    model = YOLO(model_path, task)
    results = model.predict(source, stream=True, conf=conf)

    writer = None
    for r in results:
        plotted_img = r.plot(line_width=2, kpt_radius=2)
        if writer is None:
            h, w = plotted_img.shape[:2]
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        writer.write(plotted_img)
    writer.release()


def main(old_model: Path, new_model: Path, task: str, source: Path, output: Path, fps: float, conf: float) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        old_video = Path(tmp_dir) / "old.mp4"
        new_video = Path(tmp_dir) / "new.mp4"
        render(old_model, task, source, old_video, fps, conf)
        render(new_model, task, source, new_video, fps, conf)

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                old_video,
                "-i",
                new_video,
                "-c:v",
                "libopenh264",
                "-crf",
                "18",
                "-filter_complex",
                "hstack=inputs=2",
                output,
                "-y",  # Overwrite output file if it exists
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("comparison.mp4"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    main(args.old_model, args.new_model, args.task, args.source, args.output, args.fps, args.conf)
