from pathlib import Path
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT
    / "dataset"
    / "data.yaml"
)

RUNS = ROOT / "runs"

MODELS = ROOT / "models"

RUNS.mkdir(
    parents=True,
    exist_ok=True
)

MODELS.mkdir(
    parents=True,
    exist_ok=True
)


if not DATASET.exists():

    raise FileNotFoundError(
        "data.yaml not found.\n"
        "Run this first:\n"
        "python prepare_dataset.py"
    )


print("=" * 60)
print("STARTING YOLOv8 TRAINING")
print("=" * 60)


model = YOLO("yolov8n.pt")


results = model.train(

    data=str(DATASET),

    epochs=50,

    imgsz=640,

    batch=8,

    workers=2,

    project=str(RUNS),

    name="industrial_defect",

    patience=10,

    pretrained=True,

    cache=False,

    verbose=True
)


best_model = (
    RUNS
    / "industrial_defect"
    / "weights"
    / "best.pt"
)


if best_model.exists():

    destination = (
        MODELS
        / "best.pt"
    )

    shutil.copy2(
        best_model,
        destination
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best model:\n{destination}"
    )

else:

    print(
        "WARNING: best.pt was not found."
    )