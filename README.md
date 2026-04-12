# 😊 Emotion Detection — Deep Learning

> Real-time facial emotion recognition system trained on the FER-2013 dataset using convolutional neural networks and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-FER--2013-green?style=flat-square)
![License](https://img.shields.io/github/license/walicard56/Emotion_Detection_IA?style=flat-square)

---

## Overview

This project implements a **facial emotion recognition** system capable of detecting 7 emotions in real time from a webcam feed or static images. It uses a **Convolutional Neural Network (CNN)** trained on the industry-standard **FER-2013** dataset and leverages **OpenCV** for face detection and frame processing.

**Detectable emotions:**
`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## How It Works

```
Webcam / Image
      ↓
Face Detection (OpenCV Haar Cascade)
      ↓
Crop & Grayscale (48×48 px)
      ↓
CNN Model Inference
      ↓
Emotion Label + Confidence Score
```

---

## Project Structure

```
├── cria_ia.py      # Model architecture definition and training pipeline
├── ia.py           # Real-time inference — webcam or image input
├── data/           # FER-2013 dataset (not included — see setup below)
└── .gitignore
```

---

## Requirements

```
tensorflow>=2.10
keras
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

---

## Installation

```bash
git clone https://github.com/walicard56/Emotion_Detection_IA.git
cd Emotion_Detection_IA
pip install -r requirements.txt
```

---

## Dataset Setup

1. Download the **FER-2013** dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Extract and place it in the `data/` directory:

```
data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

---

## Training the Model

```bash
python cria_ia.py
```

Training parameters (configurable inside `cria_ia.py`):

```python
EPOCHS      = 50
BATCH_SIZE  = 64
IMG_SIZE    = (48, 48)
NUM_CLASSES = 7
```

The trained model is saved automatically after training.

---

## Running Real-Time Detection

```bash
python ia.py
```

- Press `Q` to quit the webcam feed
- Detected emotion and confidence score are displayed in real time on screen

---

## Model Architecture

| Layer | Details |
|---|---|
| Input | 48×48 grayscale image |
| Conv2D × 4 | Feature extraction with ReLU + MaxPooling |
| Dropout | 0.25 / 0.5 — regularization |
| Dense | 512 units + ReLU |
| Output | 7-class Softmax |

---

## Author

**Walisson Jose** · [GitHub](https://github.com/walicard56) · [Portfolio](https://walicard56.github.io/Portifolio_wali)
