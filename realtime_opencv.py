from pathlib import Path
import time

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "models"
    / "best.onnx"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CAMERA_INDEX = 0

IMAGE_SIZE = 640

CONFIDENCE = 0.25


# NEU defect classes
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"ONNX model not found:\n"
        f"{MODEL_PATH}\n\n"
        "Run:\n"
        "python export_onnx.py"
    )


print("=" * 70)
print("WEEK 3 - REAL-TIME INDUSTRIAL DEFECT DETECTION")
print("=" * 70)

print(
    f"\nLoading model:\n{MODEL_PATH}"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = YOLO(
    str(MODEL_PATH),
    task="detect"
)


print("Model loaded successfully.")


# --------------------------------------------------
# OPEN CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam.\n"
        "Check your camera connection."
    )


# Try to request 640x480
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# --------------------------------------------------
# FPS VARIABLES
# --------------------------------------------------

previous_time = time.perf_counter()

fps = 0.0


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_frames = 0

total_defects = 0


print()
print("Camera started.")
print("Press Q to quit.")


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:

        print(
            "Could not read frame."
        )

        break


    total_frames += 1


    # --------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------

    results = model.predict(

        source=frame,

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE,

        verbose=False

    )


    result = results[0]


    frame_defects = 0


    # --------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates


            confidence = float(
                box.conf[0]
                .cpu()
            )


            class_id = int(
                box.cls[0]
                .cpu()
            )


            if (
                0 <= class_id
                < len(CLASS_NAMES)
            ):

                class_name = (
                    CLASS_NAMES[class_id]
                )

            else:

                class_name = str(
                    class_id
                )


            frame_defects += 1


            # --------------------------------------------------
            # DRAW BOX
            # --------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 0, 255),

                2

            )


            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(

                frame,

                label,

                (x1, max(25, y1 - 10)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 0, 255),

                2

            )


    total_defects += frame_defects


    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    current_time = time.perf_counter()

    elapsed = (
        current_time
        - previous_time
    )


    if elapsed > 0:

        instant_fps = (
            1.0
            / elapsed
        )

        # Smooth FPS
        fps = (
            0.9 * fps
            + 0.1 * instant_fps
        )


    previous_time = current_time


    # --------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    cv2.putText(

        frame,

        f"Defects: {frame_defects}",

        (20, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    cv2.putText(

        frame,

        "Press Q to quit",

        (20, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # --------------------------------------------------
    # SHOW FRAME
    # --------------------------------------------------

    cv2.imshow(

        "Industrial Defect Detection",

        frame

    )


    # --------------------------------------------------
    # QUIT
    # --------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("REAL-TIME DETECTION STOPPED")
print("=" * 70)

print(
    f"Frames processed: {total_frames}"
)

print(
    f"Total detections: {total_defects}"
)