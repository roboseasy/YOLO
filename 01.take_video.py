import os
import time
from datetime import datetime

import cv2

from frame_source import (
    ANY_CAMERA,
    FrameSourceError,
    LeKiwiStream,
    LocalCamera,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "videos")
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
DEFAULT_FPS = 30.0

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
FPS_PROBE_SECONDS = 2.0


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


def measure_fps(cap, seconds=FPS_PROBE_SECONDS):
    """들어오는 프레임 속도를 실제로 재서 VideoWriter 에 넣을 값을 얻는다.

    lekiwi host 는 FPS 를 알려주지 않고, 실제 속도는 네트워크와 host 주기에
    따라 달라진다. 틀린 값을 넣으면 저장된 영상이 빨라지거나 느려진다."""
    print(f"[측정] 프레임 속도 확인 중 ({seconds:g}초)...")
    frames = 0
    start = time.time()
    while time.time() - start < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = time.time() - start
    if frames < 2 or elapsed <= 0:
        return DEFAULT_FPS
    return frames / elapsed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        cap = open_source()
    except FrameSourceError as e:
        raise SystemExit(f"[오류] {e}")

    print(f"[소스] {cap.describe()}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("[오류] 연결은 됐지만 프레임이 오지 않습니다.")
    height, width = frame.shape[:2]

    fps = cap.fps or measure_fps(cap)
    print(f"[정보] {width}x{height} @ {fps:.1f} FPS 로 저장합니다.")

    print("[안내] 스페이스바: 녹화 시작/중지, q 또는 ESC: 종료")

    writer = None
    output_path = None
    recording = False
    record_start = 0.0

    window_name = "Recorder (Space=Start/Stop, q/ESC=Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[경고] 프레임을 읽지 못했습니다.")
                break

            if (frame.shape[1], frame.shape[0]) != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
                display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            else:
                display = frame.copy()
            if recording:
                elapsed = time.time() - record_start
                cv2.circle(display, (25, 25), 10, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"REC {elapsed:5.1f}s",
                    (45, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                if writer is not None:
                    writer.write(frame)
            else:
                cv2.putText(
                    display,
                    "IDLE - press SPACE to record",
                    (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}.mp4")
                    writer = cv2.VideoWriter(output_path, FOURCC, fps, (width, height))
                    if not writer.isOpened():
                        print(f"[오류] VideoWriter를 열 수 없습니다: {output_path}")
                        writer = None
                        continue
                    recording = True
                    record_start = time.time()
                    print(f"[녹화 시작] {output_path}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"[저장 완료] {output_path}")
                    output_path = None
            elif key in (ord("q"), 27):
                if recording and writer is not None:
                    writer.release()
                    writer = None
                    print(f"[저장 완료] {output_path}")
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
