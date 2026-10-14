
from pathlib import Path
from threading import Lock

import numpy as np
from ultralytics import YOLO


class DefectDetector:
    def __init__(
        self,
        model_path,
        confidence=0.25,
        image_size=640
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.lock = Lock()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

        names = self.model.names
        self.class_names = {
            int(k): str(v) for k, v in names.items()
        }

    def predict(self, image):
        if image is None or image.size == 0:
            raise ValueError("Invalid input image.")

        with self.lock:
            result = self.model.predict(
                source=image,
                imgsz=self.image_size,
                conf=self.confidence,
                verbose=False
            )[0]

        height, width = image.shape[:2]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                coords = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                x1, y1, x2, y2 = coords
                class_id = int(
                    box.cls[0].detach().cpu().item()
                )
                confidence = float(
                    box.conf[0].detach().cpu().item()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": self.class_names.get(
                        class_id,
                        f"class_{class_id}"
                    ),
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": {
                        "x1": round(max(0, min(width, x1)), 2),
                        "y1": round(max(0, min(height, y1)), 2),
                        "x2": round(max(0, min(width, x2)), 2),
                        "y2": round(max(0, min(height, y2)), 2)
                    }
                })

        return {
            "image_width": width,
            "image_height": height,
            "detection_count": len(detections),
            "detections": detections
        }
