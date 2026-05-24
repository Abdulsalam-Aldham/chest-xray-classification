# Chest X-Ray Classification 🫁

Deep learning project for classifying chest X-ray images as **Normal** or **Pneumonia** using two approaches: a custom **CNN** built from scratch and **MobileNetV2** with transfer learning.

## 📊 Results

| Model | Test Accuracy |
|-------|--------------|
| Custom CNN | 87.18% |
| MobileNetV2 | 86.38% |

## 📁 Dataset

- **Source:** [Labeled Chest X-Ray Images (Kaggle)](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)
- **Classes:** NORMAL, PNEUMONIA
- **Training images:** 5,232 (1,349 Normal + 3,883 Pneumonia)
- **Test images:** 624 (234 Normal + 390 Pneumonia)

## 🛠️ Tech Stack

- Python 3.12
- TensorFlow / Keras 2.20
- NumPy, Matplotlib
- Scikit-learn
- Google Colab

## 🏗️ Project Steps

1. Setup Kaggle API & download dataset
2. Data exploration and visualization
3. Image preprocessing & augmentation
4. Build & train custom CNN
5. Build & train MobileNetV2 (transfer learning)
6. Evaluate both models
7. Compare results

## 🔬 Model Architectures

### Custom CNN
- 3 × Conv2D + MaxPooling layers (32 → 64 → 128 filters)
- Flatten + Dropout (0.5)
- Dense (128) + Output (sigmoid)
- **Total params:** ~11.1M

### MobileNetV2
- Pre-trained ImageNet weights (frozen)
- GlobalAveragePooling2D
- Dropout (0.3) + Dense (sigmoid)

## ⚙️ Training Configuration

- **Input size:** 224 × 224
- **Batch size:** 32
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 10 (with EarlyStop
