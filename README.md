# Industrial-Defect-Detection-System

A Computer Vision-based Industrial Defect Detection System designed to automatically detect and classify surface defects on metal/steel surfaces using YOLOv8. The project is developed as a foundation for real-time industrial quality inspection and future edge deployment.

📌 Project Overview

In manufacturing industries, surface defects such as scratches, cracks, inclusions, and pits can affect product quality. Manual inspection is time-consuming and can be inconsistent.

This project uses deep learning and computer vision to automatically:

Detect surface defects
Localize defects using bounding boxes
Classify different defect types
Display confidence scores
Evaluate detection performance
Prepare the model for real-time inference
Support future ONNX/TensorRT edge optimization
🎯 Objectives

The main objectives of the project are:

Collect and prepare an industrial surface-defect dataset.
Convert defect annotations into YOLO format.
Apply image augmentation to improve model generalization.
Train a YOLOv8 object detection model.
Evaluate the model using standard object-detection metrics.
Analyze false positives and false negatives.
Prepare the trained model for real-time and edge deployment.
📊 Dataset

The project uses the NEU Metal Surface Defects Dataset.

The dataset contains six major categories of steel surface defects:

Class ID	Defect Class
0	Crazing
1	Inclusion
2	Patches
3	Pitted Surface
4	Rolled-in Scale
5	Scratches
Defect Description
Crazing – Fine crack-like patterns on the metal surface.
Inclusion – Foreign material or impurities embedded in the metal.
Patches – Irregular regions with abnormal surface appearance.
Pitted Surface – Small holes or depressions on the surface.
Rolled-in Scale – Oxide/scale material pressed into the surface during rolling.
Scratches – Linear marks or surface damage.

Note: The dataset itself is not included in this repository because of its size. Download/prepare the dataset separately and place it in the required directory structure.

🧠 Technology Stack
Programming Language
Python
Machine Learning / Deep Learning
PyTorch
Ultralytics YOLOv8
Computer Vision
OpenCV
Albumentations
Data Processing
NumPy
Pandas
Visualization
Matplotlib
Deployment / Optimization
ONNX
ONNX Runtime
TensorRT (for compatible NVIDIA hardware)
Development Environment
Visual Studio Code
Python Virtual Environment
