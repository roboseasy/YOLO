# YOLO: 데이터 수집부터 라벨링 그리고 모델 학습과 추론까지

YOLO 기반 객체 탐지 프로젝트의 전체 파이프라인을 다룹니다. 웹캠으로 직접 데이터를 수집하고, 프레임 단위로 분할해 라벨링한 뒤, YOLO 모델을 학습시켜 추론까지 수행하는 과정을 단계별로 정리합니다.

## 프로젝트 구조

```
yolo/
├── 01.take_video.py          # 웹캠으로 mp4 동영상 촬영
├── 02.splite_video2pic.py    # 동영상을 프레임 단위 이미지로 분할
├── outputs/                  # 생성물 저장소 (git 추적 제외)
│   ├── videos/               # 촬영된 mp4 파일
│   └── pictures/             # 분할된 프레임 이미지
└── README.md
```

## 진행 단계

### 1. 데이터 수집 — 동영상 촬영
[01.take_video.py](01.take_video.py)

- 실행 시 사용 가능한 카메라 목록을 터미널에 출력하고 사용자가 인덱스를 선택
- **Space**: 녹화 시작/중지 토글, **q / ESC**: 종료
- 저장 경로: `outputs/videos/YYYYMMDD_HHMMSS.mp4`

```bash
python 01.take_video.py
```

### 2. 데이터 전처리 — 프레임 분할
[02.splite_video2pic.py](02.splite_video2pic.py)

- `INPUT_VIDEO` 변수로 지정한 mp4 파일을 모든 프레임으로 분할
- 저장 경로: `outputs/pictures/<영상이름>/<영상이름>_00000001.png` …

```bash
python 02.splite_video2pic.py
```

### 3. 라벨링 (예정)
LabelImg / Roboflow / CVAT 등을 이용해 YOLO 포맷으로 바운딩 박스 라벨링.

### 4. 모델 학습 (예정)
Ultralytics YOLO로 커스텀 데이터셋 학습.

### 5. 추론 (예정)
학습된 가중치로 이미지·동영상·실시간 웹캠 추론.

## 요구사항

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- (이후 단계) Ultralytics, PyTorch 등

## 비고

`outputs/` 폴더는 [.gitignore](.gitignore)로 git 추적에서 제외되어 있어 영상·이미지 결과물은 원격 저장소에 업로드되지 않습니다.
