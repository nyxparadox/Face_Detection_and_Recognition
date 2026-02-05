# 🎯 Face Recognition System (MTCNN + FaceNet)

A real-time face recognition system built with **Python**, using **MTCNN for face detection** and **FaceNet embeddings + SVM for recognition**.

This project demonstrates a complete face recognition pipeline — from dataset loading and embedding generation to image and webcam recognition — using **free and open-source tools only**.

---

## 🚀 Features

- Face detection using **MTCNN**
- Face recognition using **FaceNet embeddings**
- SVM classifier for identity prediction
- Image-based face recognition
- Real-time webcam recognition
- Automatic handling of missing detections
- Modular and clean training pipeline
- Fully offline & free

---

## 🧠 How It Works

The pipeline follows these steps:

1. Detect faces using **MTCNN**
2. Extract **FaceNet embeddings**
3. Train an **SVM classifier**
4. Save trained artifacts
5. Recognize faces from image or webcam

**Image/Webcam → MTCNN → FaceNet → SVM → Identity**


---

## 📸 Demo

#

![Screenshot Image_01](https://res.cloudinary.com/dytuvjwqu/image/upload/v1770295669/6920c917-0b6b-410c-ab69-428f796c2a74-1_all_16993_vdgzma.jpg)

#

![Screenshot Image_02](https://res.cloudinary.com/dytuvjwqu/image/upload/v1770295871/6920c917-0b6b-410c-ab69-428f796c2a74-1_all_17001_jg5lri.jpg)



---

## 📂 Project Structure

Face-Recognition\
│\
├── face_train.py\
├── face_recognizer(Image).py\
├── face_recognizer(Video).py\
├── images_dataset\
│ └──  person_1\
| &ensp;└──  images.png\
│ └── person_2\
| &ensp;└──  images.png\
└── README.md\


> Dataset and trained artifacts are excluded using `.gitignore`.

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/nyxparadox/Face_Detection_and_Recognition.git

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt


python face_train.py        #This generates trained model artifacts locally.

python face_recognizer(Image).py    #Image recognition

python face_recognizer(Video).py    #Webcam recognition
```

---
## 📦 Requirements

- Python 3.9+
- OpenCV
- NumPy
- MTCNN
- keras-facenet
- scikit-learn
- TensorFlow

---
## ⚠️  Note

**This repository does not include face datasets or trained model artifacts.**

To use the system:

Add your own images to the dataset folder

Run training locally
