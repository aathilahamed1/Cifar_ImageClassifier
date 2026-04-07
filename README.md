# 🚀 CIFAR-10 Image Classifier

An end-to-end Image Classification system built using **PyTorch**, **CNN**, and **Transfer Learning (ResNet)**, deployed as a live web application using **Streamlit** on Hugging Face Spaces.

---

## 🌐 Live Demo
👉 https://huggingface.co/spaces/aathilahamed/cifar-image-classifier

---

## 📸 Demo Screenshots

### 🔹 UI Interface
![UI](images/ui.png)

### 🔹 Prediction Example
![Prediction](images/prediction.png)

### 🔹 Training Progress
![Training](images/training.png)

### 🔹 Architecture Diagram
![Architecture](images/architecture.png)

---

## ⚙️ Features

- Image upload & real-time prediction
- CNN baseline model
- Transfer Learning using ResNet
- Data augmentation & preprocessing
- Deployed web app (Streamlit)

---

## 🧠 Model Details

- Dataset: **CIFAR-10**
- Input Size: **32x32**
- Models Used:
  - CNN (Baseline)
  - ResNet (Transfer Learning)

---

## 📊 Results

- Final Accuracy: **71.43%**
- Training Loss decreased consistently across epochs

---

## 🔄 Pipeline

Image → Preprocessing → CNN / ResNet → Prediction

---

## ⚠️ Limitations

- Model is trained on **CIFAR-10 (low-resolution images)**
- Performance may vary on real-world images

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- Hugging Face Spaces

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/aathilahamed1/Cifar_ImageClassifier.git
cd Cifar_ImageClassifier

pip install -r requirements.txt
streamlit run app.py
