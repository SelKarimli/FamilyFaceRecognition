# 👨‍👩‍👧‍👦 Family Face Recognition System (Python)

A **Python-based face recognition system** that identifies family members (Father, Mother, Sister, Me) in real time using a webcam.
The system is built using **InsightFace (ArcFace embeddings)** and **SVM classification**, and includes **unknown person detection**.

> 🎓 Educational / personal project  
> ⚠️ The dataset is private and **not included** in this repository

---

## ✨ Features

- ✅ Real-time face recognition via webcam
- ✅ Supports multiple known identities (family members)
- ✅ **Unknown person detection** using a confidence threshold
- ✅ Robust to different angles, lighting, and facial expressions
- ✅ Uses **pretrained ArcFace** (no heavy training required)
- ✅ Clean, readable, and modular Python code

---

## 🧠 Technologies Used

- **Python 3.9+**
- **InsightFace** (ArcFace + face detection)
- **OpenCV**
- **scikit-learn (SVM)**
- **NumPy**
- **ONNX Runtime**

---

## 📁 Project Structure

```
family-face-recognition/
│
├── dataset_raw/          # Private dataset (NOT uploaded)
│   ├── Sel/
│   ├── Ata/
│   ├── Ana/
│   └── Baci/
│
├── build_embeddings.py   # Face detection + embedding extraction
├── train_svm.py          # Train SVM classifier
├── realtime.py           # Real-time recognition with webcam
│
├── X.npy                 # Face embeddings (generated)
├── y.npy                 # Labels (generated)
├── names.npy             # Class names (generated)
├── svm_face.pkl          # Trained model
│
└── README.md
```

---

## 📸 Dataset Preparation

- Each person should have **~50–70 images**
- Images should include:
  - frontal and side angles
  - different lighting conditions
  - slight facial expressions
- Images can be extracted **automatically from video** (recommended)

📌 Dataset structure:
```
dataset_raw/
  PersonName/
    img001.jpg
    img002.jpg
```

---

## ⚙️ Installation

```bash
pip install insightface onnxruntime opencv-python scikit-learn numpy tqdm joblib
```

> If GPU is not available, the system will automatically fall back to CPU.

---

## 🚀 Usage

### 1️⃣ Extract Face Embeddings
```bash
python build_embeddings.py
```

### 2️⃣ Train the Classifier
```bash
python train_svm.py
```

### 3️⃣ Run Real-Time Face Recognition
```bash
python realtime.py
```

Press **ESC** to exit.

---

## 🧪 Unknown Person Detection

The system uses a confidence threshold to avoid false identification.

```python
THRESH = 0.60
```

- Increase threshold → fewer false positives
- Decrease threshold → fewer false "Unknown" results

---

## 📊 Model Choice

- **ArcFace (pretrained)** for high-quality face embeddings
- **SVM (linear kernel)** for small datasets with high accuracy
- No end-to-end deep learning training required

---

## 🔒 Privacy & Ethics

- All images belong to family members
- Explicit consent was obtained
- Dataset is **not shared publicly**
- Project is for **educational and personal use only**

---

## 📌 Future Improvements

- Add face recognition from video files
- Improve unknown detection with distance-based metrics
- Convert model to ONNX / TensorFlow Lite
- Build a Flutter or Web frontend

---

## 👤 Author

**Seljan Karimli**  
Computer Science / AzTU Student  
GitHub: https://github.com/SelKarimli

---

⭐ If you find this project useful, feel free to star the repository!

