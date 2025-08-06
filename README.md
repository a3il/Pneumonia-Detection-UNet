# Pneumonia Detection and Classification using Chest X-Ray Images

## 📌 Overview
This project applies machine learning and deep learning techniques to detect and classify pneumonia from chest X-ray images. A custom Convolutional Neural Network (CNN) was developed for both binary and multi-class classification, alongside a Support Vector Machine (SVM) model for performance comparison. The CNN achieved up to **96% accuracy** in binary classification and over **92%** in multi-class classification tasks.

---

## 🧠 Problem Statement
Pneumonia is a potentially life-threatening infection that causes inflammation in the lungs. Early diagnosis is critical, especially in regions with limited access to radiologists. This project aims to provide an automated tool that classifies chest X-ray images as:
- Normal
- Viral Pneumonia
- Bacterial Pneumonia
- Covid-19
---

## 🛠️ Tools & Technologies
- **Programming Language**: Python  
- **Libraries & Frameworks**:  
  - TensorFlow / Keras  
  - OpenCV  
  - Scikit-learn  
  - Matplotlib, Seaborn  
- **Development Environment**: Jupyter Notebook

---

## 📁 Dataset
The dataset used is publicly available from [Kaggle: Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), containing 5,863 labeled chest X-ray images divided into:
- Normal
- Pneumonia (Bacterial)
- Pneumonia (Viral)

---

## 📊 Model Architectures
### 🧬 CNN Model
- Built from scratch using Keras
- Trained on grayscale chest X-ray images
- Binary and multi-class configurations
- Included dropout and batch normalization layers for regularization

### ⚙️ SVM Model
- Used image features extracted with OpenCV
- Flattened and scaled grayscale images
- Served as a classical ML baseline

---

## 📈 Results Summary

| Model        | Classification Type | Accuracy |
|--------------|---------------------|----------|
| CNN (Custom) | Binary              | 96%      |
| CNN (Custom) | Multi-class         | 92%      |
| SVM          | Binary              | 89%      |

> Additional evaluation: Confusion matrix, precision, recall, F1-score

---

## 📷 Sample Outputs
- Visualization of X-ray input images
- Model accuracy/loss plots over training epochs
- Confusion matrix heatmaps
- Sample predictions (Correct vs Misclassified)

---

## 🚀 How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/a3il/pneumonia-xray-classification.git
   cd pneumonia-xray-classification
