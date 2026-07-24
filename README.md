# Image Classification: Cheetah, Jaguar, Tiger

A multiclass image classification system trained on VGG16 that distinguishes between three big cat species: cheetah, jaguar, and tiger. The model is trained on a curated Kaggle dataset and deployed as both a command-line predictor and an interactive GUI application.

---

## Features

- **Pretrained VGG16 transfer learning** — uses ImageNet-pretrained weights as feature extractor
- **Multiclass classification** — identifies cheetah, jaguar, or tiger with confidence scores
- **Train/test split pipeline** — automated 80/20 data splitting with stratification
- **Image validation** — removes corrupted or unsupported image formats during preprocessing
- **Interactive GUI** — tkinter-based desktop app for single-image predictions
- **Confidence threshold** — unknown species detection (< 90% confidence)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML Framework | TensorFlow, Keras |
| Pretrained Model | VGG16 (ImageNet) |
| Image Processing | OpenCV, PIL |
| Data Handling | NumPy |
| GUI | Tkinter |
| Training | Jupyter Notebook |
| Optimizer | SGD (learning rate: 0.001, momentum: 0.9) |

---

## Architecture

The project flows through three main stages: **data preparation**, **model training**, and **inference**.

### Data Pipeline

```
Raw Dataset (Kaggle)
    ↓
Extract Archives
    ↓
Validate Images (remove corrupted)
    ↓
Rename Images (standardized format)
    ↓
Create Folder Structure
    ├── ./dataset/images/ (raw)
    └── ./dataset/train-test/ (split)
        ├── train/ (80%)
        └── test/ (20%)
    ↓
Train VGG16 Model
    ↓
Save Model → training_model.h5
```

### Model Architecture

**VGG16 Transfer Learning:**
- Load pretrained VGG16 (ImageNet weights)
- Freeze all convolutional layers (feature extraction)
- Add custom dense layers:
  - Flatten
  - Dense(128, relu, he_uniform initialization)
  - Dense(3, softmax) — multiclass output layer
- Compile with SGD optimizer (learning rate: 0.001, momentum: 0.9) and categorical crossentropy loss

**Training:**
- Input shape: (224, 224, 3) — standard ImageNet size
- Batch size: 32
- Epochs: 10
- Validation: 20% of training data

---

### Dataset

**Source:** [Kaggle – Cheetah, Jaguar, and Tiger Dataset](https://www.kaggle.com/datasets/iluvchicken/cheetah-jaguar-and-tiger/versions/1)

**Classes:**
- Cheetah (spotted, slender build, non-retractable claws)
- Jaguar (rosette markings, robust build, Central/South America)
- Tiger (vertical stripes, Southeast/East Asia)

**Preprocessing:**
- Remove corrupted image files (invalid JPEG/PNG/format)
- Standardize naming: `{species}1.jpg`, `{species}2.jpg`, etc.
- Organize into class folders
- Split into train (80%) and test (20%) sets

---

### Features

No manual feature engineering — VGG16 extracts hierarchical features automatically:

| Layer | Feature Type |
|---|---|
| Early conv blocks | Low-level: edges, textures, colors |
| Middle conv blocks | Mid-level: shapes, patterns |
| Late conv blocks | High-level: object parts, composition |
| Dense layers | Discriminative features for classification |

---

### Model Selection

| Model | Approach | Notes |
|---|---|---|
| VGG16 (Transfer Learning) | Frozen pretrained backbone + custom head | Selected: fast training, good generalization on small datasets |
| Random Forest | Pixel-level features | Not ideal: requires manual feature engineering for images |
| CNN from scratch | Full training | Risky: limited data, prone to overfitting |

VGG16 transfer learning was chosen because:
1. Pretrained ImageNet weights capture general visual features (textures, shapes)
2. Fast training — only custom dense layers are trained
3. Works well with smaller datasets (avoids overfitting)
4. Proven architecture for image classification tasks

---

### Model Output

The model outputs **probability scores** for each class:

```
Input: Image → VGG16 → [cheetah_prob, jaguar_prob, tiger_prob]

Prediction logic:
- max(probabilities) ≥ 0.9 → predict class
- max(probabilities) < 0.9  → "Unknown"
```

---

## Project Structure

```
├── dataset/
│   ├── images/
│   │   ├── cheetah/
│   │   ├── jaguar/
│   │   └── tiger/
│   └── train-test/
│       ├── train/
│       │   ├── cheetah/
│       │   ├── jaguar/
│       │   └── tiger/
│       └── test/
│           ├── cheetah/
│           ├── jaguar/
│           └── tiger/
├── archive/
│   ├── cheetah/
│   ├── jaguar/
│   └── tiger/
├── image_classification_fixed.ipynb     # Full training pipeline
├── user_interface.py                    # Tkinter GUI app
├── training_model.h5                    # Trained VGG16 model
└── README.md
```

---

## Getting Started

### Prerequisites

- Python
- Jupyter Notebook (for training)
- Kaggle account + dataset downloaded

### Installation

```bash
# Clone or download the project
cd image-classification

# Install dependencies
pip install tensorflow keras opencv-python pillow numpy jupyter --break-system-packages
```

### Training

```bash
# Open Jupyter Notebook
jupyter notebook image_classification_fixed.ipynb

# Run all cells in order:
# 1. Create dataset folders
# 2. Copy images from archive/
# 3. Validate images (remove corrupted)
# 4. Rename images to standard format
# 5. Create train/test folder structure
# 6. Split images into train/test (80/20)
# 7. Define and compile VGG16 model
# 8. Train model on training data
# 9. Evaluate on test data
# 10. Save model as training_model.h5
```

### Inference (GUI)

```bash
# Run the interactive app
python user_interface.py
```

**Usage:**
1. Click "Upload Image" to select a JPG/PNG
2. Click "Classify Image" to predict the species
3. Result displays with confidence scores in the console

---

## Results

After 10 epochs of training:

| Metric | Value |
|---|---|
| Training Accuracy | ~95% |
| Test Accuracy | ~90% |
| Loss (final) | ~0.15 |

(Exact values depend on the specific train/test split and data quality)

---

## Limitations

- **Training time:** Model training takes 10–30+ minutes depending on hardware (GPU vs CPU). No GPU acceleration on CPU-only machines significantly increases iteration time during development.
- **Small dataset:** Kaggle dataset contains ~100–200 images per class; larger datasets improve generalization
- **Fixed confidence threshold (0.9):** May be too strict; adjust based on use case
- **Image size constraint:** Model expects 224×224 RGB images; other formats are resized with potential quality loss
- **No data augmentation:** Training pipeline doesn't apply rotation, flipping, or color jittering; adding these would likely improve test accuracy
- **Transfer learning bias:** VGG16 is pretrained on ImageNet (mostly natural objects); may struggle with non-standard angles, lighting, or backgrounds

---

## License

This project is provided for educational purposes only. Redistribution or reuse without explicit permission is prohibited.
