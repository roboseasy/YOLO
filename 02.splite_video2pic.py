import os

import cv2

INPUT_VIDEO = "/home/khw/workspace/yolo/outputs/videos/20260525_194151.mp4"
OUTPUT_ROOT = "/home/khw/workspace/yolo/outputs/pictures"


def main():
    if not os.path.isfile(INPUT_VIDEO):
        raise FileNotFoundError(f"입력 동영상을 찾을 수 없습니다: {INPUT_VIDEO}")

    video_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    output_dir = os.path.join(OUTPUT_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열 수 없습니다: {INPUT_VIDEO}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[정보] 총 프레임 수: {total}")
    print(f"[저장 위치] {output_dir}")

    index = 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            filename = f"{video_name}_{index:08d}.png"
            out_path = os.path.join(output_dir, filename)
            if not cv2.imwrite(out_path, frame):
                print(f"[오류] 저장 실패: {out_path}")
                break

            if index % 50 == 0:
                print(f"  - {index} 프레임 저장 완료")
            index += 1
    finally:
        cap.release()

    print(f"[완료] 총 {index - 1}장의 이미지를 저장했습니다.")


if __name__ == "__main__":
    main()
