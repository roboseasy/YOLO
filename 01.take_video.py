import os
import time
from datetime import datetime

import cv2

OUTPUT_DIR = "/home/khw/workspace/yolo/outputs/videos"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
DEFAULT_FPS = 30.0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
MAX_PROBE_INDEX = 10


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    available = list_available_cameras()
    camera_index = prompt_camera_index(available)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다 (index={camera_index}).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = DEFAULT_FPS

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
