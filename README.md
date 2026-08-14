# **Waste Segregation using Computer Vision**

## **Description**

This project implements a computer vision-based waste segregation system using a MobileNetV2 model to classify waste into categories such as battery, cardboard, clothes, glass, metal, paper, and plastic. It provides real-time predictions through a webcam and also identifies items as biodegradable or non-biodegradable to assist in proper waste disposal.

---

## **Problem Statement**

Improper waste segregation is a major issue in everyday environments such as homes, campuses, and public spaces. Manual sorting is inefficient and often inaccurate. This project aims to automate the classification of waste using computer vision to support better waste management practices.

---

## **Objectives**

* Classify waste into multiple categories using image classification
* Enable real-time detection using a webcam
* Identify waste as biodegradable or non-biodegradable
* Build a simple, usable system that demonstrates practical application of computer vision

---

## **Project Structure**

```
waste-segregation-project/
├── models/
    ├──waste_model.h5
├── results/
    ├── battery.png
    ├── cardboard.png
    ├── clothes.png
    ├── glass.png
    ├── metal.png
    ├── paper.png
    ├── plastic.png
    ├── plastic2.png
├── README.md
├── prepare_data.py
├── requirements.txt
├── train.py
├── webcam_classifier.py
```

---

## **Dataset**

* Waste classification dataset (multi-class)
* Classes used:

  * Battery
  * Cardboard
  * Clothes
  * Glass
  * Metal
  * Paper
  * Plastic

**Note:** Dataset is not included due to size.
Download from Kaggle and place inside:

```
Data/Garbage_Classification/
```

---

## **Technologies Used**

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

---

## **Installation & Setup**

### **1. Clone the Repository**

```
git clone <your-repo-link>
cd <repo-folder>
```

### **2. Install Dependencies**

```
pip install -r requirements.txt
```

---

## **How to Run**

### **Step 1: Prepare Dataset**

Split dataset into training and validation:

```
python prepare_data.py
```

### **Step 2: Train Model**

```
python train.py
```

### **Step 3: Run Real-Time Classification**

```
python webcam_classifier.py
```

---

## **Features**

* Real-time waste classification using webcam
* Multi-class classification (7 categories)
* Biodegradable vs Non-biodegradable identification
* Prediction smoothing for stable output
* Bounding box guidance for better accuracy

---

## **Results**

The model successfully classifies waste items in real-time with reasonable accuracy.
Screenshots of predictions are available in the `results/` folder.

---

## **Challenges Faced**

* Limited computational resources (CPU-only training)
* Dataset variability and class imbalance
* Noisy predictions in real-time webcam feed
* Need for stabilization (solved using smoothing technique)

---

## **Future Improvements**

* Use object detection (YOLO) for better localization
* Improve accuracy with larger dataset
* Deploy as a web or mobile application
* Integrate with smart bins for automated sorting

---

## **Conclusion**

This project demonstrates how computer vision can be applied to solve a real-world problem like waste segregation. It highlights the practical use of deep learning models in building efficient and scalable environmental solutions.

---

If you want next, I can create your **full project report (high scoring, structured)** — that’s actually the most important part for marks.

