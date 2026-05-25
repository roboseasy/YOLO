import time

import cv2
from ultralytics import YOLO

WEIGHTS = "/home/khw/workspace/yolo/outputs/runs/green_cube_v1/weights/best.pt"
CAMERA_INDEX = 4
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = 0


def main():
    print(f"[모델 로드] {WEIGHTS}")
    model = YOLO(WEIGHTS)
    class_names = model.names

    print(f"[카메라 열기] index={CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={CAMERA_INDEX}).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    window_name = "YOLO Inference (q/ESC to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print("[안내] q 또는 ESC: 종료")

    prev_time = time.time()
    fps_smooth = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[경고] 프레임을 읽지 못했습니다.")
                break

            results = model.predict(
                frame,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device=DEVICE,
                verbose=False,
            )
            annotated = results[0].plot()

            now = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

            num_det = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(
                annotated,
                f"FPS {fps_smooth:5.1f}  det {num_det}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if (annotated.shape[1], annotated.shape[0]) != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
                annotated = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    _ = class_names


if __name__ == "__main__":
    main()
