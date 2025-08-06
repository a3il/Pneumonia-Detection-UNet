# Pneumonia Detection and Classification using Chest X-Ray Images

[Research Paper PDF](https://github.com/user-attachments/files/21613574/Pneumonia_Detection.pdf)

## 📌 Overview
This project applies machine learning and deep learning techniques to detect and classify pneumonia from chest X-ray images. A custom Convolutional Neural Network (CNN) was developed for both binary and multi-class classification, alongside a Support Vector Machine (SVM) model for performance comparison. The CNN achieved up to **96% accuracy** in binary classification and over **92%** in multi-class classification tasks.
<img width="269" height="299" alt="arch" src="https://github.com/user-attachments/assets/fd0a8717-e1f9-4eeb-8db5-e5ddda9d4857" />

---

## 🧠 Problem Statement
Pneumonia is a potentially life-threatening infection that causes inflammation in the lungs. Early diagnosis is critical, especially in regions with limited access to radiologists. This project aims to provide an automated tool that classifies chest X-ray images as:
- Normal
- Viral Pneumonia
- Bacterial Pneumonia
- Covid-19
---
<img width="266" height="262" alt="classes" src="https://github.com/user-attachments/assets/d1e5b3fd-798f-4a0c-bbaf-e01ff68d0e04" />

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

---<img width="272" height="216" alt="dataset" src="https://github.com/user-attachments/assets/8e60f632-1233-4d93-ab73-25a6954df8a6" />

<img width="266" height="271" alt="preprocessing" src="https://github.com/user-attachments/assets/425d9105-4472-4280-aa55-9d4abfe0de1d" />

## 📊 Model Architectures
### 🧬 CNN Model
- Built from scratch using Keras
- Trained on grayscale chest X-ray images
- Binary and multi-class configurations
- Included dropout and batch normalization layers for regularization
<img width="138" height="224" alt="unetarch" src="https://github.com/user-attachments/assets/58ccd82a-b7e3-448e-bb66-5b0a8a82f88a" />
<img width="262" height="255" alt="cnnarch" src="https://github.com/user-attachments/assets/47e14672-ed39-419a-922e-cb6e8d078bd7" />

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
  

---<img width="535" height="233" alt="confusion matrix" src="https://github.com/user-attachments/assets/90d80c05-a6f6-422a-b8f5-f14ffaf918c3" />
<img width="157" height="140" alt="binary confusion matrix" src="https://github.com/user-attachments/assets/abb9c925-234d-4b0f-a663-cdbefc68348b" />
<img width="241" height="205" alt="roccurve" src="https://github.com/user-attachments/assets/261fbb2f-ad62-44b8-ac2e-eaa7fee01fb7" />


## 🚀 How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/a3il/Pneumonia-Detection-UNet.git
   cd pneumonia-xray-classification
