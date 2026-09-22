from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL = (
    ROOT
    / "models"
    / "best.pt"
)

DATA = (
    ROOT
    / "dataset"
    / "data.yaml"
)


if not MODEL.exists():

    raise FileNotFoundError(
        "models/best.pt not found.\n"
        "Run train.py first."
    )


if not DATA.exists():

    raise FileNotFoundError(
        "dataset/data.yaml not found.\n"
        "Run prepare_dataset.py first."
    )


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


model = YOLO(
    str(MODEL)
)


metrics = model.val(

    data=str(DATA),

    split="val",

    imgsz=640,

    conf=0.25

)


print()
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"Precision       : {metrics.box.mp:.4f}"
)

print(
    f"Recall          : {metrics.box.mr:.4f}"
)

print(
    f"mAP@0.50        : {metrics.box.map50:.4f}"
)

print(
    f"mAP@0.50:0.95   : {metrics.box.map:.4f}"
)