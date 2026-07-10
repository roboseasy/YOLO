import glob
import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "outputs", "videos")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs", "pictures")


def select_video(video_dir: str) -> str:
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"동영상 폴더를 찾을 수 없습니다: {video_dir}")

    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        raise FileNotFoundError(f"{video_dir} 에 mp4 파일이 없습니다.")

    print("동영상을 고르세요")
    for i, path in enumerate(videos, start=1):
        print(f"  {i}. {os.path.basename(path)}")

    while True:
        raw = input(f"번호 입력 (1-{len(videos)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            return videos[int(raw) - 1]
        print(f"  -> 1 부터 {len(videos)} 사이의 번호를 입력하세요.")


def main():
    input_video = select_video(VIDEO_DIR)

    video_name = os.path.splitext(os.path.basename(input_video))[0]
    output_dir = os.path.join(OUTPUT_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열 수 없습니다: {input_video}")

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
