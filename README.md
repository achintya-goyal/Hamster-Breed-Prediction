# 🐹 Hamster Breed Classification System

## 📌 Overview
This project uses deep learning to classify hamster breeds (Dwarf, Syrian, Rats) using MobileNetV2 with transfer learning.

## 🚀 Features
- Image classification (~95% accuracy)
- Real-world testing
- Grad-CAM visualization for explainability
- GPU acceleration (DirectML)

## 🧠 Model
- Architecture: MobileNetV2
- Transfer Learning
- Fine-tuning

## 📊 Results
- Training Accuracy: ~97%
- Validation Accuracy: ~95%
- Real-world Test: 9/10 correct

## 🔍 Explainability
Grad-CAM is used to visualize important regions influencing predictions.

## ⚠️ Limitations
- Difficulty distinguishing visually similar classes (e.g., baby rats vs hamsters)

## 🛠️ Tech Stack
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## ▶️ How to Run
```bash
pip install -r requirements.txt
python train.py
python predict.py
python gradcam.py
