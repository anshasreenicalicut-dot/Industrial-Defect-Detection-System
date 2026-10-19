# Industrial Defect Detection Using YOLOv8

## 1. Project Overview

Industrial Defect Detection is a computer vision project that uses a YOLOv8-based object detection model to identify defects in industrial surface images.

The system provides a REST API built with FastAPI, allowing users to upload images and receive detection results. Docker is used to containerize the application, while Prometheus collects application metrics for monitoring.

## 2. Objectives

- Detect defects in industrial surface images using a trained YOLOv8 model.
- Provide an API for image upload and defect prediction.
- Validate uploaded image formats and file sizes.
- Return structured JSON responses containing detection results.
- Containerize the API using Docker.
- Monitor API performance and application metrics using Prometheus.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Object detection |
| Ultralytics | YOLO model framework |
| OpenCV | Image decoding and processing |
| NumPy | Image array processing |
| FastAPI | REST API development |
| Uvicorn | ASGI server |
| Docker | Application containerization |
| Docker Compose | Multi-container orchestration |
| Prometheus | Metrics collection and monitoring |
| pytest | Automated testing |

## 4. Project Structure

```text
Industrial Detection/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── detector.py
│
├── models/
│   └── best.pt
│
├── monitoring/
│   └── prometheus.yml
│
├── tests/
│   └── test_api.py
│
├── dataset/
│   └── data.yaml
│
├── runs/
│   └── industrial_defect/
│
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
└── README.md
```

*Note: The structure above represents the intended organization. Some folders, such as `dataset/`, `runs/`, or `tests/`, may differ depending on your current project files.*

## 5. Features

- **Image-based defect detection:** Processes uploaded industrial surface images.
- **YOLOv8 inference:** Uses trained model weights to predict defects.
- **REST API:** Exposes endpoints through FastAPI.
- **Input validation:** Supports JPG, JPEG, and PNG uploads and rejects unsupported formats.
- **File size validation:** Limits uploaded images to 10 MB.
- **Lazy model loading:** Loads the detection model when a prediction is first requested.
- **Health monitoring:** Provides an endpoint for checking API and model-file status.
- **Containerization:** Runs the application inside Docker.
- **Metrics monitoring:** Exposes Prometheus metrics through `/metrics`.

## 6. Installation and Setup

### Prerequisites

Install the following software:

- Python
- Visual Studio Code
- Docker Desktop
- Git (optional, for version control)

### Step 1: Clone the repository

If the project is hosted on GitHub:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Industrial Detection"
```

Alternatively, open your existing project directory in VS Code.

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` contains the dependencies required by your implementation, including:

```text
ultralytics
fastapi
uvicorn[standard]
opencv-python-headless
numpy
python-multipart
prometheus-fastapi-instrumentator
```

Add testing dependencies such as `pytest` if your test suite requires them.

### Step 4: Add the trained model

Place your trained YOLOv8 model at:

```text
models/best.pt
```

The API expects the model at this location by default. If you have not trained the model yet, prepare the annotated dataset and train YOLOv8 before testing real predictions.

## 7. Running the API Locally

From the project root, activate the virtual environment and run:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API documentation:

**Swagger UI:** http://localhost:8000/docs

Other useful endpoints:

- Home: http://localhost:8000/
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Displays API information |
| GET | `/health` | Checks API and model status |
| POST | `/predict` | Accepts an image and returns defect predictions |
| GET | `/metrics` | Exposes metrics for Prometheus |

### Prediction request

1. Open `/docs`.
2. Expand `POST /predict`.
3. Click **Try it out**.
4. Upload a JPG, JPEG, or PNG image.
5. Click **Execute**.
6. Inspect the JSON response.

The prediction response contains the status, uploaded filename, and detection information returned by the detector implementation.

## 9. Docker Deployment

Make sure Docker Desktop is running.

Build and start the application and monitoring services:

```powershell
docker compose up --build -d
```

Check the running containers:

```powershell
docker compose ps
```

View API logs:

```powershell
docker compose logs --tail=100 api
```

View Prometheus logs:

```powershell
docker compose logs --tail=100 prometheus
```

Stop the services:

```powershell
docker compose down
```

### Docker services

| Service | Local URL | Purpose |
|---|---|---|
| API | http://localhost:8000 | Image prediction API |
| Swagger UI | http://localhost:8000/docs | Interactive API testing |
| Prometheus | http://localhost:9090 | Monitoring dashboard |

The Docker Compose configuration should mount `monitoring/prometheus.yml` into the Prometheus container and configure the API target using the Docker Compose service name, such as `api:8000`.

## 10. Prometheus Monitoring

Prometheus collects metrics exposed by the FastAPI application.

Example `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["prometheus:9090"]

  - job_name: "industrial-defect-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

After starting Docker Compose, open:

http://localhost:9090/targets

The `industrial-defect-api` target should show **UP** when Prometheus can successfully scrape the API.

Example Prometheus query:

```promql
up{job="industrial-defect-api"}
```

A value of `1` indicates a successful scrape; `0` indicates that the target is configured but the scrape failed.

## 11. Testing

Run automated tests from the project root:

```powershell
python -m pytest -v
```

Test the health endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Test the metrics endpoint:

```powershell
Invoke-WebRequest http://localhost:8000/metrics -UseBasicParsing
```

Test predictions through Swagger UI at http://localhost:8000/docs.

A successful metrics request should return HTTP `200 OK`. Prediction tests should use a valid image and a compatible trained model.

## 12. Error Handling

The API is designed to handle common errors, including:

- Unsupported image formats: HTTP `415`.
- Empty or invalid images: HTTP `400`.
- Images larger than 10 MB: HTTP `413`.
- Missing or unavailable model: HTTP `503`.
- Unexpected prediction failures: HTTP `500`.

If the API container repeatedly restarts, inspect its logs:

```powershell
docker compose logs --tail=100 api
```

If Prometheus cannot scrape the API, verify the configuration file, target service name, `/metrics` endpoint, and container logs.

## 13. Limitations

- Detection quality depends on the training dataset, annotations, and model weights.
- The model can detect only the defect classes it was trained to recognize.
- Performance depends on image quality and available CPU/GPU resources.
- A successful API health check does not necessarily mean the model has already been loaded.
- Prometheus monitoring requires a valid scrape configuration and reachable metrics endpoint.
- The project requires evaluation on representative industrial images before real-world deployment.

## 14. Future Enhancements

- Add a graphical user interface for image uploads and visualization.
- Display predicted defect classes, confidence scores, and bounding boxes.
- Store prediction history and inspection reports.
- Add model evaluation metrics and defect statistics.
- Introduce authentication and access control.
- Add dashboards and alerts for API availability and performance.
- Optimize inference speed for industrial inspection workloads.

## 15. Conclusion

This project demonstrates an end-to-end approach to industrial surface defect detection using computer vision and modern API infrastructure. YOLOv8 provides the detection capability, FastAPI exposes prediction services, Docker simplifies deployment, and Prometheus supports operational monitoring.

The system provides a foundation for developing automated industrial inspection solutions that can be extended with additional defect classes, improved datasets, evaluation tools, and production monitoring.

## Author

**Project:** Industrial Defect Detection Using YOLOv8  
**Domain:** Machine Learning and Computer Vision  
**Tools:** Python, YOLOv8, FastAPI, Docker, Prometheus

