# YOLO: 데이터 수집부터 라벨링 그리고 모델 학습과 추론까지

YOLO 기반 객체 탐지 프로젝트의 전체 파이프라인을 다룹니다. 웹캠으로 직접 데이터를 수집하고, 프레임 단위로 분할해 라벨링한 뒤, YOLO 모델을 학습시켜 추론까지 수행하는 과정을 단계별로 정리합니다.

## 프로젝트 구조

```
yolo/
├── frame_source.py           # 프레임 출처(로컬 웹캠 / LeKiwi ZMQ 스트림)
├── 01.take_video.py          # mp4 동영상 촬영
├── 02.splite_video2pic.py    # 동영상을 프레임 단위 이미지로 분할
├── outputs/                  # 생성물 저장소 (git 추적 제외)
│   ├── videos/               # 촬영된 mp4 파일
│   └── pictures/             # 분할된 프레임 이미지
└── README.md
```

## 진행 단계

### 1. 데이터 수집 — 동영상 촬영
[01.take_video.py](01.take_video.py)

영상 출처는 파일 위쪽 `SOURCE` 로 고릅니다. 기본값은 `"lekiwi"` 입니다.

| `SOURCE` | 영상 출처 |
| --- | --- |
| `"lekiwi"` | 라즈베리파이의 `lekiwi_host` 가 ZMQ 로 뿌리는 스트림 (랜선 직결) |
| `"local"` | 노트북에 꽂은 USB 웹캠 |

- 실행 시 사용 가능한 카메라(로컬은 index, LeKiwi 는 `front` / `wrist`)를 출력하고 선택
- **Space**: 녹화 시작/중지 토글, **q / ESC**: 종료
- 저장 경로: `outputs/videos/YYYYMMDD_HHMMSS.mp4`

```bash
python 01.take_video.py
```

#### LeKiwi 카메라로 찍기

랜선 직결 기준입니다.

```
노트북 10.42.0.1  ──(유선)──  LeKiwi(라즈베리파이) 10.42.0.61
                              :5556  observation (영상 + 팔 관절값)  ← 우리가 쓰는 것
                              :5555  command (팔/베이스 제어)        ← 안 씀
```

**① 로봇에 host 를 먼저 띄웁니다.** 이게 안 떠 있으면 스크립트는 한 줄도 못 돕니다.

```bash
ssh roboseasy@10.42.0.61
./start_lekiwi_host.sh          # 이 SSH 창은 켜 둔 채로 둘 것 (닫으면 host 도 죽음)
```

`No command available` · `Stopping the base` 경고는 정상입니다. 이 저장소는 영상만 받고
명령(5555)은 안 보내기 때문입니다.

**② 노트북에서 촬영합니다.**

```bash
ping -c1 10.42.0.61             # 먼저 연결 확인
python 01.take_video.py
```

- `lerobot` 를 `yolo` 환경에 깔 필요는 **없습니다.** host 가 보내는 메시지는 그냥 JSON 이라
  `pyzmq` 만 있으면 받습니다.
- 한 host 에는 **클라이언트를 하나만** 붙이세요. teleop·record 가 같이 붙어 있으면 프레임을
  나눠 가져서 뚝뚝 끊깁니다.
- 색이 이상하면(보라가 분홍, 노랑이 파랑) `HOST_COLOR_MODE` 를 `bgr` 로 바꿉니다. lerobot 의
  `OpenCVCameraConfig` 기본값이 RGB 라 host 가 채널을 뒤집어 보내는 것을 되돌리는 값입니다.
- host 는 FPS 를 알려주지 않아서, 녹화 전에 2초간 실제 프레임 속도를 재서 그 값으로 저장합니다.

### 2. 데이터 전처리 — 프레임 분할
[02.splite_video2pic.py](02.splite_video2pic.py)

- `INPUT_VIDEO` 변수로 지정한 mp4 파일을 모든 프레임으로 분할
- 저장 경로: `outputs/pictures/<영상이름>/<영상이름>_00000001.png` …

```bash
python 02.splite_video2pic.py
```

### 3. 라벨링 — Roboflow Auto-Label
분할된 이미지를 [Roboflow](https://roboflow.com)에 업로드한 뒤 Auto-Label 기능으로 자동 바운딩 박스 라벨링을 수행합니다.

- **비용**: 이미지 500장당 약 **5 크레딧** 소모
- **전략**: 크레딧 절약을 위해 **먼저 500장만 자동 라벨링**하여 작은 데이터셋을 확보
- 라벨링 결과는 YOLO 포맷으로 export하여 학습에 사용

### 4. 모델 학습 — 500장 데이터셋으로 시범 학습
우선 자동 라벨링된 500장만으로 Ultralytics YOLO 학습을 진행해 파이프라인이 정상 동작하는지, 성능이 어느 정도 나오는지 확인합니다. 결과를 보고 추가 데이터 라벨링 여부를 결정합니다.

### 5. 추론 (예정)
학습된 가중치로 이미지·동영상·실시간 웹캠 추론.

## 요구사항

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- (이후 단계) Ultralytics, PyTorch 등

## 비고

`outputs/` 폴더는 [.gitignore](.gitignore)로 git 추적에서 제외되어 있어 영상·이미지 결과물은 원격 저장소에 업로드되지 않습니다.
