# 🔢 MNIST Digit Classifier — Logistic Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-eab308?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange?style=flat-square)

> A **pure NumPy** implementation of Logistic Regression for binary classification of handwritten digits from the MNIST dataset — built from mathematical first principles, without any high-level ML estimators.

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
- ✅ Full preprocessing pipeline: normalization + binary label encoding
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
│   ├── data_loader.py          # Fetches MNIST via OpenML, splits into train/test sets
│   ├── logistic_regression.py  # Core model: Sigmoid, Gradient Descent, Save/Load
│   └── utils.py                # Metrics computation & Confusion matrix plotting
│
├── notebooks/
│   └── modelTrial.ipynb        # Exploratory data analysis and experimental runs
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
| `src/data_loader.py` | Fetches MNIST from OpenML, normalizes pixels to `[0, 1]`, encodes binary labels, performs train/test split |
| `src/logistic_regression.py` | Contains the `LogisticRegression` class with `fit()`, `predict()`, `sigmoid()`, `save()`, and `load()` methods |
| `src/utils.py` | Computes Accuracy, Precision, Recall, F1-Score; renders the Confusion Matrix |
| `train_final.py` | Orchestrates the end-to-end pipeline: load data → train → evaluate → save weights |
| `notebooks/modelTrial.ipynb` | Sandbox for visual EDA, hyperparameter experiments, and prototype runs |

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
```

> **Note:** `scikit-learn` is used **only** for fetching the MNIST dataset (`fetch_openml`) and computing evaluation metrics. The Logistic Regression model itself is implemented entirely in NumPy.

### 3. Train the Model

```bash
python train_final.py
```

This will:
1. Fetch the MNIST dataset via OpenML (cached after first download)
2. Preprocess and split the data
3. Train the Logistic Regression model for **15000 iterations** at **learning rate 1.1**
4. Print evaluation metrics to the console
5. Save the trained `theta` weights to the `models/` directory
6. Display the Confusion Matrix plot

### 4. Expected Console Output

```
Training complete and model saved.
Actual iterations run: 15000
Final Loss: 0.1032
Train Accuracy:  0.9632
Test Accuracy:  0.9597
Train Precision:  0.8654
Test Precision:  0.8493
Train Recall:  0.7251
Test Recall:  0.7104
Train F1 Score:  0.7891
Test F1 Score:  0.7736
```

---

## 🔧 Implementation Notes

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

### Binary Classification Setup

Although MNIST contains 10 classes (`0`–`9`), this project frames it as a **one-vs-all binary problem**:

- **Positive class (`y = 1`):** Images of the digit `8`
- **Negative class (`y = 0`):** All other digits

Labels are re-encoded accordingly before training.

---

## 📊 Results

Training configuration: **15000 iterations**, **learning rate α = 1.1**

### Performance Metrics

| Metric | Score |
|---|---|
| **Accuracy** | ~96.0%|
| **Precision** | ~84.9%|
| **Recall** | ~71.0% |
| **F1-Score** | ~77.4% |

### Interpretation

The high **accuracy (~96.0%)** is partially inflated by the class imbalance inherent to one-vs-all MNIST classification — digits other than `8` constitute the large majority of samples.

The more informative metrics reveal the real picture:

- **Precision (~84.9%):** When the model predicts a digit is `8`, it is correct about 8.5 out of 10 times. This indicates relatively low false positives.
- **Recall (~71.0%):** The model successfully identifies a strong majority (about 71%) of all actual `8`s in the test set, meaning some true `8`s are still missed (false negatives).
- **F1-Score (~77.4%):** The harmonic mean reflects the improved balance between Precision and Recall. The model is performing well, but still leaves room for improvement on recall due to class imbalance.

This behavior is typical when applying a vanilla Gradient Descent classifier to an imbalanced binary task without threshold tuning or class weighting.

### Confusion Matrix

The confusion matrix is generated automatically after training:

```
                 Predicted: 0    Predicted: 1
Actual: 0          12,469             171
Actual: 1             394             966
```

*(Values are illustrative; exact counts depend on the train/test split seed.)*

---

## 🔭 Future Improvements

| Idea | Description |
|---|---|
| **Threshold tuning** | Adjust the decision boundary (default `0.5`) to improve Recall at the cost of some Precision |
| **Class weighting** | Weight the loss function to penalize false negatives more heavily and handle imbalance |
| **Feature engineering** | Add pixel interaction features or apply PCA for dimensionality reduction |
| **Multi-class extension** | Extend to full 10-class classification using a One-vs-All or Softmax approach |
| **Learning curve plots** | Plot training loss per iteration to visualize convergence behavior |
| **Regularization** | Add L2 (Ridge) regularization to the cost function to reduce overfitting |

---

<div align="center">
  <sub>Built from scratch with NumPy & mathematical curiosity.</sub>
</div>