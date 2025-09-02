# 🛢️ AI SpillGuard: Oil Spill Detection using Deep Learning

## 📌 Overview

AI SpillGuard is a deep learning pipeline for detecting **oil spills vs. no-oil** in satellite and aerial imagery.
The system leverages **Convolutional Neural Networks (CNNs)** to classify images, while ensuring **explainability** with **Grad-CAM heatmaps**.

✅ **Binary Classification**: Oil vs. No Oil
✅ **Preprocessing**: Resizing, normalization, augmentations
✅ **Evaluation**: Confusion matrix, ROC-AUC, Precision-Recall curves
✅ **Explainability**: Grad-CAM overlays for model interpretability

---

## 📂 Project Structure

```
AI_SpillGuard/
│── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│── notebooks/
│   └── oil_spill_classification.ipynb
│── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── gradcam_overlay.png
│── README.md
│── requirements.txt
```

---

## ⚙️ Preprocessing

* **Resizing**: All images resized to **512×512**.
* **Normalization**: Pixel values scaled to **\[0,1]**.
* **Label Mapping**: Masks converted into **binary labels** (`oil=1`, `no-oil=0`).
* **Data Augmentation**: Random flips, rotations, zooms, and brightness adjustments to improve generalization.

---

## 🚀 Training Pipeline

1. Load train/validation/test datasets using TensorFlow Datasets (`tf.data`).
2. Apply preprocessing (resize, normalize, augment).
3. Train CNN classifier (3 conv layers + dense layers).
4. Evaluate using **accuracy, AUC, confusion matrix, ROC curve**.
5. Generate **Grad-CAM heatmaps** for interpretability.

---

## 📊 Results

* **Validation Accuracy**: \~93%
* **AUC (ROC)**: 0.96
* **Confusion Matrix**: Very few false negatives (missed spills).
* **Grad-CAM**: Model attends to water surfaces where oil films appear.

**Example ROC Curve:**
![ROC Curve](results/roc_curve.png)

**Example Grad-CAM Overlay:**
![Grad-CAM](results/gradcam_overlay.png)

---

## 🛠️ Installation & Usage

### 1. Clone repository

```bash
git clone https://github.com/your-username/AI_SpillGuard.git
cd AI_SpillGuard
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Notebook

Open Jupyter Notebook and execute:

```
notebooks/oil_spill_classification.ipynb
```

---

## 📌 Future Work

* Extend to **multiclass segmentation** (Background, Oil, Water, Others).
* Deploy as a **Streamlit web app** for real-time monitoring.
* Experiment with **transfer learning** (ResNet, EfficientNet).

---

## 👨‍💻 Author

Developed as part of an **internship project** on deep learning for environmental monitoring.


