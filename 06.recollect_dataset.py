"""지금 모델로 자동 라벨링하며 학습 데이터를 더 쌓는다.

LeKiwi 카메라(front / wrist) 프레임을 추론해서, 잡힌 박스가 전부 --save-conf
(기본 0.90) 이상인 프레임만 원본 이미지 + YOLO 라벨로 dataset/train 에 이어 붙인다.
"""
import argparse
import glob
import os
import time

import cv2
import yaml
from ultralytics import YOLO

from frame_source import ANY_CAMERA, FrameSourceError, LeKiwiStream

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "train", "images")
LABELS_DIR = os.path.join(DATASET_DIR, "train", "labels")

# 아래는 전부 기본값이고, 명령줄 옵션으로 덮어쓴다. (--help 참고)
#   python 06.recollect_dataset.py --remote=192.168.0.206
DEFAULT_REMOTE_IP = "192.168.0.201"
DEFAULT_CAMERA_NAME = "front"         # host 가 발행하는 이름. 실행 시 다시 고를 수 있다.
DEFAULT_PORT_OBSERVATIONS = 5556
DEFAULT_HOST_COLOR_MODE = "rgb"       # host 가 보내는 채널 순서 (rgb | bgr)

DEFAULT_SAVE_CONF = 0.90              # 박스가 전부 이 이상일 때만 저장
DEFAULT_CONF_THRES = 0.25             # 이보다 낮은 검출은 '없는 것'으로 본다
DEFAULT_IOU_THRES = 0.45
DEFAULT_INTERVAL_S = 1.0              # 저장 최소 간격. 30 FPS 를 다 담으면 거의 같은 사진만 쌓인다
DEFAULT_DEVICE = "0"


def latest_weights():
    """가장 최근에 학습한 best.pt. 다시 학습하면 자동으로 새 모델을 쓴다."""
    paths = glob.glob(os.path.join(BASE_DIR, "outputs", "runs", "*", "weights", "best.pt"))
    return max(paths, key=os.path.getmtime) if paths else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="지금 모델로 자동 라벨링해서 dataset/train 에 학습 데이터를 더 쌓는다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", default=latest_weights(), metavar="PATH",
        help="라벨링에 쓸 .pt 가중치 (기본: outputs/runs 에서 가장 최근 best.pt)",
    )
    parser.add_argument(
        "--remote", "--remote-ip", dest="remote_ip", default=DEFAULT_REMOTE_IP,
        metavar="IP", help="lekiwi host 의 IP",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT_OBSERVATIONS,
        help="lekiwi host 의 관측 스트림 포트",
    )
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA_NAME, metavar="NAME",
        help="lekiwi 카메라 이름 (실행 중에 다시 고를 수 있다)",
    )
    parser.add_argument(
        "--color-mode", choices=("rgb", "bgr"), default=DEFAULT_HOST_COLOR_MODE,
        help="host 가 보내는 채널 순서. 색이 뒤집혀 보이면 바꾼다",
    )
    parser.add_argument(
        "--save-conf", type=float, default=DEFAULT_SAVE_CONF,
        help="모든 박스가 이 confidence 이상인 프레임만 저장",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF_THRES,
        help="검출 하한. 이 이상 ~ --save-conf 미만인 박스가 있으면 그 프레임은 버린다",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU_THRES, help="NMS IoU 임계값",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S, metavar="SEC",
        help="저장 최소 간격(초)",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE, help="추론 장치 (0, 0,1, cpu ...)",
    )
    args = parser.parse_args(argv)
    if not args.weights:
        parser.error("outputs/runs/*/weights/best.pt 가 없습니다. 먼저 03.train.py 로 학습하거나 --weights 를 주세요.")
    args.device = normalize_device(args.device)
    return args


def normalize_device(raw):
    """ultralytics 는 GPU 를 문자열 '0' 보다 정수 0 으로 받는 쪽이 안전하다."""
    text = str(raw).strip()
    return int(text) if text.isdigit() else text


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


def load_class_map(model_names):
    """모델 클래스 id -> data.yaml 클래스 id. 이름이 안 맞으면 라벨이 엉뚱한 클래스로 쌓이므로 멈춘다."""
    with open(os.path.join(DATASET_DIR, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    missing = [n for n in model_names.values() if n not in names]
    if missing:
        raise SystemExit(
            f"[오류] 모델 클래스 {missing} 가 dataset/data.yaml 의 names {names} 에 없습니다.\n"
            "  --weights 로 이 데이터셋을 학습한 가중치를 주세요."
        )
    return {i: names.index(n) for i, n in model_names.items()}


def make_label_lines(classes, confs, xywhn, class_map, save_conf):
    """박스가 전부 save_conf 이상이면 YOLO 라벨 줄들, 아니면 None.

    애매한 박스만 빼고 저장하면 그 물체가 '배경'으로 학습되므로 프레임째 버린다."""
    if not confs or min(confs) < save_conf:
        return None
    return [
        f"{class_map[int(c)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for c, (x, y, w, h) in zip(classes, xywhn)
    ]


def save_sample(frame, lines, stem):
    # 라벨 먼저: 이미지만 남으면 '박스 없는 사진'으로 학습되지만, 라벨만 남으면 무시된다.
    label_path = os.path.join(LABELS_DIR, f"{stem}.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    image_path = os.path.join(IMAGES_DIR, f"{stem}.png")
    if not cv2.imwrite(image_path, frame):
        os.remove(label_path)
        raise SystemExit(f"[오류] 이미지를 저장하지 못했습니다: {image_path}")


def _self_check():
    m = {0: 0}
    assert make_label_lines([], [], [], m, 0.95) is None
    assert make_label_lines([0, 0], [0.99, 0.5], [[.5, .5, .1, .1]] * 2, m, 0.95) is None
    assert make_label_lines([0], [0.97], [[.5, .25, .1, .2]], m, 0.95) == ["0 0.500000 0.250000 0.100000 0.200000"]


def main(argv=None):
    _self_check()
    args = parse_args(argv)

    print(f"[모델 로드] {args.weights}")
    model = YOLO(args.weights)
    class_map = load_class_map(model.names)

    try:
        cap = LeKiwiStream(
            remote_ip=args.remote_ip,
            cam_name=ANY_CAMERA,
            port=args.port,
            host_color_mode=args.color_mode,
        )
    except FrameSourceError as e:
        raise SystemExit(f"[오류] {e}")
    cap.use_camera(prompt_camera_name(cap.camera_names, args.camera))
    print(f"[소스] {cap.describe()}")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    session = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"recollect_{cap.cam_name}_{session}"
    print(f"[저장 위치] {IMAGES_DIR}  ({prefix}_*.png)")
    print(f"[조건] 모든 박스 conf >= {args.save_conf}, {args.interval:g}초마다 최대 1장")
    print("[안내] 스페이스바: 수집 시작/중지, q 또는 ESC: 종료")

    window_name = "Recollect (Space=Start/Stop, q/ESC=Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    collecting = False
    saved = 0
    last_save = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[경고] 프레임을 읽지 못했습니다.")
                break

            result = model.predict(
                frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False,
            )[0]
            boxes = result.boxes
            confs = boxes.conf.tolist()
            lines = make_label_lines(
                boxes.cls.tolist(), confs, boxes.xywhn.tolist(), class_map, args.save_conf,
            )

            now = time.time()
            if collecting and lines and now - last_save >= args.interval:
                save_sample(frame, lines, f"{prefix}_{saved + 1:05d}")
                saved += 1
                last_save = now

            if lines:
                why = f"OK x{len(lines)}"
            elif confs:
                why = f"skip: min conf {min(confs):.2f}"
            else:
                why = "skip: no box"
            state = "REC" if collecting else "PAUSE"
            annotated = result.plot()
            cv2.putText(
                annotated,
                f"{state}  saved {saved}  {why}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if collecting else (200, 200, 200),
                2,
            )
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                collecting = not collecting
                print(f"[{'수집 시작' if collecting else '수집 중지'}] 지금까지 {saved}장")
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[완료] {saved}장을 dataset/train 에 추가했습니다. 03.train.py 로 다시 학습하세요.")


if __name__ == "__main__":
    main()
