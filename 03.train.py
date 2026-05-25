from ultralytics import YOLO

DATA_YAML = "/home/khw/workspace/yolo/dataset/data.yaml"
PROJECT_DIR = "/home/khw/workspace/yolo/outputs/runs"
RUN_NAME = "green_cube_v1"

PRETRAINED = "yolo26n.pt"
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
DEVICE = 0


def main():
    model = YOLO(PRETRAINED)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=20,
        save=True,
        plots=True,
    )

    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_val",
    )
    print("[검증 결과]")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
