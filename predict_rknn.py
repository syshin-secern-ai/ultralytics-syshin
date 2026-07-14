import argparse
import socket
import struct
import threading
import time
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.utils import YAML


def main(
    model: Path,
    host: str,
    cam_port: int,
    viewer_port: int,
    conf: float,
    iou: float,
) -> None:
    task = YAML.load((model if model.is_dir() else model.parent) / "metadata.yaml")["task"]
    model = YOLO(model, task)

    cap = cv2.VideoCapture(f"http://{host}:{cam_port}/")

    # 추론이 카메라 fps보다 느리면 VideoCapture 버퍼에 오래된 프레임이 쌓여 지연이 누적되므로,
    # 읽기 스레드가 프레임을 계속 소비하며 최신 한 장만 남기고 나머지는 버린다.
    latest = None
    running = True

    def reader() -> None:
        nonlocal latest, running
        while running:
            ret, img = cap.read()
            if not ret:
                running = False
                break
            latest = img

    threading.Thread(target=reader, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # WSL 미러드 네트워킹에서는 뷰어가 닫혀도 FIN/RST가 안 오는 경우가 있어 send가 버퍼만 채우다
    # 무기한 블록된다. 송신 타임아웃을 걸어 그 경우에도 send 실패로 종료되게 한다.
    sock.settimeout(5.0)
    sock.connect((host, viewer_port))
    print(f"BOARD: Connected to viewer at {host}:{viewer_port}")

    try:
        while True:
            img = latest
            if img is None:
                if not running:
                    print("BOARD: No frame.")
                    break
                time.sleep(0.005)
                continue
            latest = None

            results = model.predict(
                source=img,
                conf=conf,
                iou=iou,
            )

            result = results[0]
            plotted_img = result.plot(line_width=2, kpt_radius=2)

            kpts = result.keypoints
            if kpts is not None and kpts.conf is not None:
                for xy, kpt_conf in zip(kpts.xy.cpu().numpy(), kpts.conf.cpu().numpy()):
                    for (x, y), c in zip(xy, kpt_conf):
                        if x == 0 and y == 0:
                            continue
                        cv2.putText(
                            plotted_img,
                            f"{c:.2f}",
                            (int(x) + 3, int(y) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            ok, buf = cv2.imencode(".jpg", plotted_img)
            if not ok:
                continue
            data = buf.tobytes()
            sock.sendall(struct.pack(">I", len(data)) + data)
    except (BrokenPipeError, ConnectionResetError, socket.timeout):
        print("BOARD: Viewer closed (q pressed). Exiting.")
    finally:
        running = False
        sock.close()
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", type=str, default="172.16.151.180")
    parser.add_argument("--cam-port", type=int, default=5000)
    parser.add_argument("--viewer-port", type=int, default=5001)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    main(args.model, args.host, args.cam_port, args.viewer_port, args.conf, args.iou)
