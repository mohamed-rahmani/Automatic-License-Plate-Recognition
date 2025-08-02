# 🚗 Automatic License Plate Recognition (ALPR)

This project implements an Automatic License Plate Recognition (ALPR) system using a supervised classification model (SVM) trained on isolated characters (letters and digits). The system includes character segmentation, classification, and full plate reconstruction.

---

## ⚙️ Features

- 🔍 **Automatic character segmentation** from the license plate image.
- 🤖 **Character classification** (digits and letters) using a Support Vector Machine (SVM).
- 🧠 **Plate reconstruction** by sorting characters based on their horizontal position.
- ⚠️ **Low-confidence warning** for characters that may need manual verification.

---

## 📦 Dependencies

- Python ≥ 3.6
- `scikit-learn`
- `scikit-image`
- `joblib`
- `matplotlib`
