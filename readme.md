# Family Face Recognition System (Academic Version)

## Project Overview

This project implements a **family-level face recognition system** using modern computer vision techniques. The goal is to recognize known individuals (family members) from a webcam stream or a static image, while correctly labeling unknown people as *Unknown*. In addition, the system records **presence intervals**, i.e., which person was present and during which time period.

The project is designed for **educational purposes** and demonstrates how pretrained deep learning models can be combined with classical machine learning methods to build an effective recognition pipeline without training a deep neural network from scratch.

---

## Learning Objectives

By studying or reproducing this project, one can learn:

* How face recognition systems are structured end-to-end
* The difference between **face detection**, **face embedding**, and **face classification**
* How pretrained deep learning models (ArcFace) are used in practice
* How to handle **small datasets** using classical classifiers (SVM)
* How to implement **Unknown person detection** using confidence thresholds
* How to log temporal information (who was present and when)

---

## System Architecture

The system follows a modular pipeline:

1. **Face Detection**
   Faces are detected in images or video frames using InsightFace.

2. **Face Embedding Extraction**
   Each detected face is converted into a fixed-length numerical vector (512 dimensions) using the ArcFace model. These embeddings capture identity-related facial features.

3. **Classification**
   A Support Vector Machine (SVM) is trained on embeddings of known individuals to perform identity recognition.

4. **Decision Logic**
   If the classification confidence is below a predefined threshold, the face is labeled as *Unknown*.

5. **Visualization and Logging**
   Bounding boxes and labels are drawn on images or video frames, and presence intervals are logged to a file.

High-level workflow:

```
Image / Webcam Frame
        ↓
Face Detection (InsightFace)
        ↓
Face Embedding (ArcFace, 512-D)
        ↓
SVM Classifier
        ↓
Known Person / Unknown
        ↓
Visualization + Presence Log
```

---

## Technologies Used

* **Python 3.9+** – Main programming language
* **InsightFace (ArcFace)** – Face detection and embedding extraction
* **OpenCV** – Image and video processing
* **scikit-learn** – SVM classifier
* **NumPy** – Numerical operations
* **ONNX Runtime** – Efficient model inference

---

## Dataset Preparation

The dataset consists of face images of family members. All images were collected with explicit consent and are kept private.

Recommended guidelines:

* **50–70 images per person** for stable recognition
* Include variation in:

  * lighting conditions
  * head pose (frontal and side views)
  * facial expressions

Dataset structure:

```
dataset_raw/
  PersonName/
    img001.jpg
    img002.jpg
```

The dataset is excluded from the public repository for privacy reasons.

---

## Implementation Details

### Embedding Extraction

The script `build_embeddings.py`:

* Reads images from the dataset
* Detects the largest face in each image
* Extracts and normalizes ArcFace embeddings
* Saves embeddings and labels for training

### Model Training

The script `train_svm.py`:

* Loads extracted embeddings
* Splits data into training and testing sets
* Trains a linear SVM classifier
* Evaluates accuracy and saves the trained model

### Real-Time Recognition and Logging

The script `realtime.py`:

* Captures frames from a webcam
* Performs face recognition in real time
* Displays bounding boxes and labels
* Logs presence intervals (start time, end time, duration) for each person

### Static Image Annotation

The script `annotate_image.py`:

* Takes a static image as input
* Detects and recognizes all faces
* Outputs a new image with labeled bounding boxes

---

## Unknown Person Detection

To avoid incorrect identification, a confidence threshold is used:

```python
THRESH = 0.60
```

* If confidence ≥ threshold → known person
* If confidence < threshold → labeled as *Unknown*

This mechanism is essential for real-world scenarios where non-enrolled individuals may appear.

---

## Ethical Considerations

* All subjects provided informed consent
* Data is used strictly for educational purposes
* No biometric data is shared publicly
* The project is not intended for surveillance or commercial use

---

## Possible Extensions

* Attendance-style reporting and analytics
* Support for larger datasets
* Face recognition across multiple cameras
* Exporting models to mobile-friendly formats (ONNX / TFLite)
* Graphical or web-based user interface

---

## Author

**Seljan Karimli**
Computer Engineering / Artificial Intelligence
GitHub: [https://github.com/SelKarimli](https://github.com/SelKarimli)

---

This project serves as a practical and educational example of applying face recognition techniques in a small-scale, privacy-aware setting.
