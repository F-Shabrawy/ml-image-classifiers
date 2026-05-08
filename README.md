# 🔢 MNIST Digit Classifier — Logistic Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-eab308?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange?style=flat-square)

> A **pure NumPy** implementation of Logistic Regression with **HOG feature extraction** for binary classification of handwritten digits from the MNIST dataset — built from mathematical first principles, without any high-level ML estimators.

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

This project implements a **binary Logistic Regression classifier** entirely from scratch using NumPy to identify the digit **`8`** from the MNIST handwritten digits dataset.

The goal is not to achieve state-of-the-art accuracy, but to **deeply understand the mechanics** of a fundamental machine learning algorithm — the Sigmoid activation, the Cross-Entropy loss, and the Gradient Descent update rule — by building them explicitly, line by line.

### Highlights

- ✅ Logistic Regression with Gradient Descent — **zero `sklearn` estimators used**
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
| `src/data_loader.py` | Fetches MNIST from OpenML, normalizes pixels, extracts HOG features, encodes binary labels, performs 3-way train/val/test split |
| `src/logistic_regression.py` | Contains the `LogisticRegression` class with `fit()`, `predict()`, `sigmoid()`, `save()`, and `load()` methods |
| `src/utils.py` | Computes Accuracy, Precision, Recall, F1-Score; renders the Confusion Matrix & loss curves |
| `train_final.py` | Orchestrates the end-to-end pipeline: load data → train → evaluate → save weights |
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

> **Note:** `scikit-learn` is used **only** for fetching the MNIST dataset (`fetch_openml`) and computing evaluation metrics. `scikit-image` provides the HOG feature extractor. The Logistic Regression model itself is implemented entirely in NumPy.

### 3. Train the Model

```bash
python train_final.py
```

This will:
1. Fetch the MNIST dataset via OpenML (cached after first download)
2. Extract HOG features from each image (orientations=9, 4×4 cells, 3×3 blocks)
3. Split data into train/val/test (80/10/10)
4. Train the Logistic Regression model for **15000 iterations** at **learning rate 1.3**
5. Print train, val, and test evaluation metrics to the console
6. Save the trained `theta` weights to the `models/` directory
7. Display the Confusion Matrix and loss curves

### 4. Expected Console Output

```
training complete and model saved.
iterations: 15000
Final Train Loss: 0.0752
Final Val Loss: 0.1432
Train Accuracy:  0.9782
Val Accuracy:  0.9600
Test Accuracy:  0.9620
Train Precision:  0.9142
Val Precision:  0.8610
Test Precision:  0.8683
Train Recall:  0.8253
Val Recall:  0.7950
Test Recall:  0.7801
Train F1 Score:  0.8675
Val F1 Score:  0.8267
Test F1 Score:  0.8220
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

This design makes every mathematical step explicit and auditable.

### Model Persistence

Trained parameters are saved and loaded using NumPy's native binary format:

```python
# Save
np.save('models/theta.npy', self.theta)

# Load
self.theta = np.load('models/theta.npy')
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

Training configuration: **15000 iterations**, **learning rate α = 1.3**, **HOG features (or=9, ppc=(4,4), cpb=(3,3))**

### Performance Metrics

| Metric | Train | Val | Test |
|---|---|---|---|
| **Accuracy** | ~97.82% | ~96.00% | ~96.20% |
| **Precision** | ~91.42% | ~86.10% | ~86.83% |
| **Recall** | ~82.53% | ~79.50% | ~78.01% |
| **F1-Score** | ~86.75% | ~82.67% | ~82.20% |

### Interpretation

HOG features significantly improved recall (+3%) and F1 (+4%) over raw pixel features. The train/val gap stays within ~2%, suggesting the model generalizes well without severe overfitting.

- **Precision (~86.83%):** When the model predicts a digit is `8`, it is correct about 87% of the time.
- **Recall (~78.01%):** The model identifies about 78% of all actual `8`s — a notable improvement from ~75% with raw pixels.
- **F1-Score (~82.20%):** Up from ~78%, showing HOG's structural edge features help.

### Confusion Matrix

The confusion matrix is generated automatically after training:

```
                 Predicted: 0    Predicted: 1
Actual: 0          12,402             241
Actual: 1             335             1022
```

*(Values are illustrative; exact counts depend on the train/test split seed.)*

---

## 🔭 Future Improvements

| Idea | Description |
|---|---|---|
| **Class weighting** | Weight the loss function to penalize false negatives more heavily and handle imbalance |
| **Multi-class extension** | Extend to full 10-class classification using Softmax regression |
| **Regularization (L1/L2)** | Add Ridge or Lasso regularization to the cost function to reduce overfitting |
| **Cross-validation** | Use k-fold CV to tune hyperparameters (alpha, HOG params) |

---

<div align="center">
  <sub>Built from scratch with NumPy & mathematical curiosity.</sub>
</div>