from pathlib import Path

import base64
import time

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.responses import Response

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "best.pt"
)


CLASS_NAMES = [

    "crazing",

    "inclusion",

    "patches",

    "pitted_surface",

    "rolled_in_scale",

    "scratches",

]


app = FastAPI(

    title="Industrial Defect Detection API",

    description=(
        "YOLOv8 API for "
        "metal surface defect detection"
    ),

    version="1.0.0"

)


model = None


REQUEST_COUNT = Counter(

    "defect_api_requests_total",

    "Total prediction requests"

)


DEFECT_COUNT = Counter(

    "defects_detected_total",

    "Total detected defects"

)


REQUEST_LATENCY = Histogram(

    "defect_api_request_latency_seconds",

    "Prediction latency"

)


@app.on_event("startup")
def load_model():

    global model

    if MODEL_PATH.exists():

        model = YOLO(
            str(MODEL_PATH)
        )


@app.get("/")
def home():

    return {

        "project":
            "Real-Time Industrial Defect Detection",

        "model_loaded":
            model is not None,

        "classes":
            CLASS_NAMES

    }


@app.get("/health")
def health():

    return {

        "status": "ok",

        "model_loaded":
            model is not None

    }


@app.get("/metrics")
def metrics():

    return Response(

        generate_latest(),

        media_type=CONTENT_TYPE_LATEST

    )


@app.post("/predict")
async def predict(

    file: UploadFile = File(...)

):

    if model is None:

        raise HTTPException(

            status_code=503,

            detail=(
                "Model not found. "
                "Train the model first."
            )

        )


    REQUEST_COUNT.inc()

    start_time = time.perf_counter()


    file_data = await file.read()


    image_array = np.frombuffer(

        file_data,

        dtype=np.uint8

    )


    frame = cv2.imdecode(

        image_array,

        cv2.IMREAD_COLOR

    )


    if frame is None:

        raise HTTPException(

            status_code=400,

            detail="Invalid image file."

        )


    results = model(

        frame,

        conf=0.25,

        imgsz=640,

        verbose=False

    )


    result = results[0]

    detections = []


    if result.boxes is not None:

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0].cpu()
            )

            class_id = int(
                box.cls[0].cpu()
            )

            x1, y1, x2, y2 = coordinates

            class_name = (
                CLASS_NAMES[class_id]
            )


            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 0, 255),

                2

            )


            cv2.putText(

                frame,

                (
                    f"{class_name} "
                    f"{confidence:.2f}"
                ),

                (
                    x1,
                    max(25, y1 - 8)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 0, 255),

                2

            )


            detections.append({

                "class":
                    class_name,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "bbox": [

                    int(x1),

                    int(y1),

                    int(x2),

                    int(y2)

                ]

            })


    DEFECT_COUNT.inc(
        len(detections)
    )


    REQUEST_LATENCY.observe(

        time.perf_counter()
        - start_time

    )


    success, encoded = cv2.imencode(

        ".jpg",

        frame

    )


    if not success:

        raise HTTPException(

            status_code=500,

            detail="Could not encode result."

        )


    return {

        "filename":
            file.filename,

        "defect_count":
            len(detections),

        "detections":
            detections,

        "annotated_image_base64":
            base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")

    }