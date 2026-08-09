import os
import time

import cv2
from ultralytics import YOLO

from frame_source import (
    ANY_CAMERA,
    FrameSourceError,
    LeKiwiStream,
    LocalCamera,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "outputs", "runs", "cube", "weights", "best.pt")

# "lekiwi" : 라즈베리파이의 lekiwi_host 가 ZMQ 로 뿌리는 영상 (랜선 직결)
# "local"  : 노트북에 꽂은 USB 웹캠
SOURCE = "lekiwi"

# --- SOURCE = "lekiwi" ---
REMOTE_IP = "10.42.0.61"          # 랜선 직결. 노트북은 10.42.0.1
CAMERA_NAME = "front"             # host 가 발행하는 이름. 실행 시 다시 고를 수 있다.
PORT_OBSERVATIONS = 5556
HOST_COLOR_MODE = "rgb"           # host 가 보내는 채널 순서 (rgb | bgr)

# --- SOURCE = "local" ---
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
MAX_PROBE_INDEX = 10

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = 0


def list_available_cameras(max_index: int = MAX_PROBE_INDEX):
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((idx, w, h))
            cap.release()
    return available


def prompt_camera_index(available):
    if not available:
        raise RuntimeError("사용 가능한 카메라가 없습니다.")

    print("[사용 가능한 카메라]")
    for idx, w, h in available:
        print(f"  - index {idx}  ({w}x{h})")

    valid = {idx for idx, _, _ in available}
    default_idx = available[0][0]
    while True:
        raw = input(f"사용할 카메라 index 입력 (기본 {default_idx}): ").strip()
        if raw == "":
            return default_idx
        if raw.isdigit() and int(raw) in valid:
            return int(raw)
        print(f"  -> 유효한 index가 아닙니다. {sorted(valid)} 중에서 선택하세요.")


def prompt_camera_name(names, default):
    """host 가 발행 중인 카메라 중 하나를 고른다.

    키트마다 'front' 와 'wrist' 가 뒤바뀌어 있는 경우가 있어서 물어본다."""
    fallback = default if default in names else names[0]
    if len(names) == 1:
        return names[0]

    print("[host 가 발행 중인 카메라]")
    for n, name in enumerate(names, 1):
        tag = "   <- 기본값" if name == fallback else ""
        print(f"  {n}) {name}{tag}")

    while True:
        raw = input(f"사용할 카메라 (번호 또는 이름, 기본 {fallback}): ").strip()
        if raw == "":
            return fallback
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw in names:
            return raw
        print(f"  -> '{raw}' 는 없는 카메라입니다. {names} 중에서 선택하세요.")


def open_source():
    if SOURCE == "local":
        available = list_available_cameras()
        camera_index = prompt_camera_index(available)
        return LocalCamera(camera_index, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    if SOURCE == "lekiwi":
        cap = LeKiwiStream(
            remote_ip=REMOTE_IP,
            cam_name=ANY_CAMERA,
            port=PORT_OBSERVATIONS,
            host_color_mode=HOST_COLOR_MODE,
        )
        cap.use_camera(prompt_camera_name(cap.camera_names, CAMERA_NAME))
        return cap

    raise RuntimeError(f"알 수 없는 SOURCE '{SOURCE}' — 'local' 또는 'lekiwi'.")


def main():
    print(f"[모델 로드] {WEIGHTS}")
    model = YOLO(WEIGHTS)
    class_names = model.names

    try:
        cap = open_source()
    except FrameSourceError as e:
        raise SystemExit(f"[오류] {e}")

    print(f"[소스] {cap.describe()}")

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
