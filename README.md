# 🔢 MNIST Digit Classifier — Logistic & Softmax Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-eab308?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange?style=flat-square)

> A **pure NumPy** implementation of Logistic Regression (binary) and Softmax Regression (multiclass) with **HOG feature extraction** for classification of handwritten digits from the MNIST dataset — built from mathematical first principles, without any high-level ML estimators.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Mathematical Foundation](#-mathematical-foundation)
3. [Project Structure](#-project-structure)
4. [Quick Start](#-quick-start)
5. [Implementation Notes](#-implementation-notes)
6. [Results](#-results)
7. [Future Improvements](#-future-improvements)

---

## 🧠 Project Overview

This project implements both **binary Logistic Regression** and **multiclass Softmax Regression** classifiers entirely from scratch using NumPy. The binary model identifies the digit **`8`** from MNIST, while the multiclass model classifies all 10 digits (0-9).

The goal is not to achieve state-of-the-art accuracy, but to **deeply understand the mechanics** of fundamental machine learning algorithms — the Sigmoid activation, Softmax function, Cross-Entropy loss, and Gradient Descent update rule — by building them explicitly, line by line.

### Highlights

- ✅ Logistic Regression (binary) with Gradient Descent — **zero `sklearn` estimators used**
- ✅ Softmax Regression (multiclass) with L1/L2 regularization and cross-validation
- ✅ HOG feature extraction captures edge patterns better than raw pixels
- ✅ Full preprocessing pipeline: normalization + HOG + 3-way split (train/val/test)
- ✅ Model persistence: save and reload trained weights (`.npy` format)
- ✅ Comprehensive evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ Clean Matplotlib-based visualizations

---

## 📐 Mathematical Foundation

### 1. The Sigmoid Function

The Sigmoid function squashes any real-valued input into the range `(0, 1)`, making it suitable for outputting a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where the linear combination $z$ is computed as:

$$z = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

In code, this is implemented directly in NumPy:

```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```

### 2. Binary Cross-Entropy Loss

The model is optimized by minimizing the **Binary Cross-Entropy (Log Loss)** over all training examples:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where:
- $m$ is the number of training samples
- $y^{(i)}$ is the true binary label (`1` for digit `8`, `0` otherwise)
- $\hat{y}^{(i)} = \sigma(\theta^T x^{(i)})$ is the predicted probability

In code, this is implemented directly in NumPy:

```python
def compute_loss(y, y_hat):
    m = y.size
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
```

### 3. Gradient Descent Update Rule

At each iteration, the parameters $\theta$ are updated by moving in the direction of the **negative gradient** of the loss:

$$\theta := \theta - \alpha \cdot \nabla_\theta J(\theta)$$

The gradient of the loss with respect to $\theta$ has the elegant closed form:

$$\nabla_\theta J(\theta) = \frac{1}{m} X^T (\hat{y} - y)$$

Where:
- $\alpha$ is the **learning rate** (controls the step size)
- $X$ is the feature matrix of shape $(m \times n)$
- $(\hat{y} - y)$ is the vector of prediction errors

The update is applied for a fixed number of **iterations** until convergence.

---

## 📁 Project Structure

```
mnist-logistic-regression/
│
├── src/
│   ├── __init__.py             # Makes src a proper package
│   ├── data_loader.py          # Fetches MNIST via OpenML, extracts HOG features, splits into train/val/test
│   ├── logistic_regression.py  # Core model: Sigmoid, Gradient Descent, Save/Load
│   └── utils.py                # Metrics computation & Confusion matrix plotting
│
├── notebooks/
│   └── modelTrial.ipynb        # Exploratory data analysis, hyperparameter tuning, HOG experiments
│
├── models/
│   └── *.npy                   # Saved model weights (theta parameters)
│
├── train_final.py              # Main script: train, evaluate, and save the model
├── requirements.txt
└── README.md
```

### Module Responsibilities

| File | Responsibility |
|---|---|
| `src/data_loader.py` | Fetches MNIST from OpenML, normalizes pixels, extracts HOG features, encodes binary/multiclass labels, performs 3-way train/val/test split |
| `src/logistic_regression.py` | Binary classifier with `fit()`, `predict()`, `save_model()`, `load_model()` methods |
| `src/softmax_regression.py` | Multiclass classifier with L1/L2 regularization, cross-validation, learning curves, bias-variance diagnosis |
| `src/utils.py` | Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix from scratch (no sklearn); saves plots to files |
| `train_final.py` | Orchestrates end-to-end pipeline for both binary and multiclass models |
| `notebooks/modelTrial.ipynb` | Sandbox for visual EDA, hyperparameter experiments, HOG param tuning, and prototype runs |

---

## 🚀 Quick Start

### Prerequisites

- Python `3.8` or higher
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/F-Shabrawy/ml-image-classifiers.git
cd ml-image-classifiers
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes:**

```
numpy
matplotlib
scikit-learn
scikit-image
```

> **Note:** `scikit-learn` is used **only** for fetching the MNIST dataset (`fetch_openml`). `scikit-image` provides the HOG feature extractor. All evaluation metrics and the model itself are implemented entirely in NumPy.

### 3. Train the Model

```bash
python train_final.py
```

This will:
1. Fetch the MNIST dataset via OpenML (cached after first download)
2. Extract HOG features from each image (orientations=9, 4×4 cells, 3×3 blocks)
3. Split data into train/val/test (80/10/10)
4. Train the Logistic Regression model for **2000 iterations** at **learning rate 1.3**
5. Print train, val, and test evaluation metrics to the console
6. Save the trained `theta` weights to the `models/` directory
7. Save Confusion Matrix and loss curves to `results/plots/`

### 4. Expected Console Output

```
training complete and model saved.
iterations: 2000
Final Train Loss: 0.0419
Final Val Loss: 0.0477
Train Accuracy: 0.9860
Train Precision: 0.9347
Train Recall: 0.9208
Train F1 Score: 0.9277
Val Accuracy: 0.9830
Val Precision: 0.9220
Val Recall: 0.9104
Val F1 Score: 0.9161
Test Accuracy: 0.9847
Test Precision: 0.9268
Test Recall: 0.9051
Test F1 Score: 0.9158
```

---

## 🔧 Implementation Notes

### HOG Feature Extraction

Instead of feeding raw pixel values into the classifier, images are first transformed using **Histogram of Oriented Gradients (HOG)**:

- **orientations=9**: Captures edges in 8 directions + 1
- **pixels_per_cell=(4,4)**: Each 4×4 cell produces 1 gradient histogram
- **cells_per_block=(3,3)**: Normalizes across 3×3 blocks of cells

This reduces noise and emphasizes structural edge patterns, improving generalization.

### Pure NumPy — No `sklearn` Estimators

This implementation deliberately avoids `sklearn.linear_model.LogisticRegression` or any equivalent high-level estimator. Every component of the algorithm is built manually:

| Component | Implementation |
|---|---|
| **Sigmoid function** | `1 / (1 + np.exp(-z))` — applied element-wise via NumPy broadcasting |
| **Gradient computation** | `(1/m) * X.T @ (y_hat - y)` — vectorized matrix operation |
| **Parameter update** | `theta -= learning_rate * gradient` — in-place NumPy update |
| **Loss tracking** | Cross-entropy computed each iteration for convergence monitoring |
| **Accuracy** | `mean(y_pred == y_true)` — pure NumPy |
| **Precision, Recall, F1** | Built from per-class confusion matrix counts — no sklearn |
| **Confusion Matrix** | `zeros((n,n))` populated by iterating over predictions — no sklearn |

No evaluation metrics from sklearn are used. Everything is computed from scratch for transparency.

### Model Persistence

Trained parameters are saved and loaded using NumPy's native binary format:

```python
# Save
np.save('models/hog_lr_13_2000.npy', self.theta)

# Load
self.theta = np.load('models/hog_lr_13_2000.npy')
```

This allows the trained model to be reloaded instantly for inference without retraining.

### Train/Validation/Test Split

Data is split into three sets for better hyperparameter tuning and bias-variance diagnosis:

- **Training (80%)**: Model fitting and gradient descent
- **Validation (10%)**: Held-out during training — used to track overfitting
- **Test (10%)**: Final untouched evaluation set

Validation loss is tracked alongside training loss to detect divergence early.

### Binary Classification Setup

Although MNIST contains 10 classes (`0`–`9`), this project frames it as a **one-vs-all binary problem**:

- **Positive class (`y = 1`):** Images of the digit `8`
- **Negative class (`y = 0`):** All other digits

Labels are re-encoded accordingly before training.

---

## 📊 Results

Training configuration: **2000 iterations**, **learning rate α = 1.3**, **HOG features (or=9, ppc=(4,4), cpb=(3,3))**

### Performance Metrics

| Metric | Train | Val | Test |
|---|---|---|---|
| **Accuracy** | 98.60% | 98.30% | 98.47% |
| **Precision** | 93.47% | 92.20% | 92.68% |
| **Recall** | 92.08% | 91.04% | 90.51% |
| **F1-Score** | 92.77% | 91.61% | 91.58% |

### Interpretation

HOG features dramatically improved performance over raw pixels. The train/val gap is under 1%, meaning the model generalizes well with no overfitting.

- **Recall jumped from ~75% (raw pixels) to ~90%** — HOG's edge orientation features help the model recognize digit `8` much more reliably.
- **F1-Score at ~91.6%** — a 14-point improvement over raw pixels, showing HOG captures structural patterns better than flat intensities.
- The model converges in **2000 iterations** vs 15000 needed for raw pixels — HOG features are more informative per dimension.

### Confusion Matrix

The confusion matrix is saved to `results/plots/cm.png` after training:

```
                 Predicted: 0    Predicted: 1
Actual: 0          12,402             241
Actual: 1             335             1022
```

*(Values are illustrative; exact counts depend on the train/test split seed.)*

---

### Multi-class Softmax Regression (10-class)

Extended the binary logistic regression to handle all 10 digits (0-9) using **softmax regression** with the following enhancements:

- **L2 Regularization**: `λ = 0.01` to prevent overfitting
- **K-fold Cross-validation**: 5-fold CV for hyperparameter tuning
- **Learning Curves**: Diagnose bias-variance tradeoff

| Metric | Train | Val | Test |
|---|---|---|---|
| **Accuracy** | 98.01% | 97.24% | 97.41% |
| **Precision (macro)** | 97.89% | 97.03% | 97.12% |
| **Recall (macro)** | 97.87% | 97.00% | 97.08% |
| **F1-Score (macro)** | 97.87% | 97.01% | 97.09% |

The model achieves **97.41% test accuracy** on 10-class classification, demonstrating that the gradient descent framework generalizes well from binary to multiclass problems via softmax.

---

## 🔭 Future Improvements

| Idea | Description | Status |
|---|---|---|
| **Class weighting** | Weight the loss function to penalize false negatives more heavily and handle imbalance | Pending |
| **Multi-class extension** | Extend to full 10-class classification using Softmax regression | ✅ Completed |
| **Regularization (L1/L2)** | Add Ridge or Lasso regularization to the cost function to reduce overfitting | ✅ Completed |
| **Cross-validation** | Use k-fold CV to tune hyperparameters (alpha, HOG params) | ✅ Completed |

---

<div align="center">
  <sub>Built from scratch with NumPy & mathematical curiosity.</sub>
</div>