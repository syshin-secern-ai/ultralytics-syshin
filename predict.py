import argparse

import cv2

from ultralytics import YOLO


def main(model: str, task: str) -> None:
    model = YOLO(model, task)

    cap = cv2.VideoCapture("http://127.0.0.1:5000/")
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("No frame.")
            break

        results = model.predict(
            source=img,
            conf=0.25,
        )

        result = results[0]
        plotted_img = result.plot(line_width=2, kpt_radius=2)

        kpts = result.keypoints
        if kpts is not None and kpts.conf is not None:
            for xy, conf in zip(kpts.xy.cpu().numpy(), kpts.conf.cpu().numpy()):
                for (x, y), c in zip(xy, conf):
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

        cv2.imshow("0", plotted_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    main(args.model, args.task)
