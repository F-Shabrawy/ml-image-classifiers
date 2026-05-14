<div align="center">

# MNIST Digit Classification from Scratch

**Logistic Regression · Softmax Regression · HOG Features · Pure NumPy**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)](.)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST%2070k-F59E0B?style=for-the-badge)](http://yann.lecun.com/exdb/mnist/)

*A from-scratch implementation of binary and multiclass classifiers for handwritten digit recognition — built on mathematical first principles with zero high-level ML estimators.*

---

[Overview](#overview) · [Mathematical Foundation](#mathematical-foundation) · [Getting Started](#getting-started) · [Results](#results) · [Architecture](#architecture)

</div>

---

## Overview

This project implements **binary Logistic Regression** and **multiclass Softmax Regression** classifiers entirely from scratch using NumPy. Rather than relying on high-level estimators such as `sklearn.linear_model`, every component — from the sigmoid activation through gradient descent to evaluation metrics — is built explicitly to provide full transparency into how these algorithms work.

### Key Features

| | Feature | Detail |
|---|---|---|
| 🧮 | **From-scratch algorithms** | Sigmoid, softmax, cross-entropy loss, and gradient descent — all implemented in pure NumPy |
| 🔍 | **HOG feature extraction** | Edge-oriented histogram features replace raw pixels for stronger generalization |
| 📊 | **Complete evaluation suite** | Accuracy, precision, recall, F1-score, and confusion matrices — no `sklearn` metrics used |
| 🔄 | **Regularization & CV** | L1/L2 regularization and k-fold cross-validation for robust hyperparameter tuning |
| 💾 | **Model persistence** | Save and reload trained weights in `.npy` format for instant inference |
| 📈 | **Convergence diagnostics** | Loss curves and learning-rate analysis across train/val splits |

---

## Mathematical Foundation

### Sigmoid Function

The sigmoid maps any real value to the interval $(0,\;1)$, producing a probability estimate:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \theta^\top x$$

```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```

### Binary Cross-Entropy Loss

The cost function penalizes confident wrong predictions via log-loss:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \bigl[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \bigr]$$

where $\hat{y}^{(i)} = \sigma(\theta^\top x^{(i)})$ and $m$ is the number of training examples.

```python
def compute_loss(y, y_hat):
    m = y.size
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
```

### Gradient Descent

Parameters are updated iteratively along the negative gradient:

$$\theta \leftarrow \theta - \alpha \cdot \frac{1}{m} X^\top (\hat{y} - y)$$

where $\alpha$ is the learning rate and $(\ \hat{y} - y\ )$ is the vector of prediction residuals.

### Softmax Extension

For the 10-class problem the sigmoid is replaced by the **softmax** function and the loss generalizes to categorical cross-entropy. L2 regularization ($\lambda = 0.01$) is added to the cost to prevent overfitting.

---

## Getting Started

### Prerequisites

- Python ≥ 3.8
- pip

### Installation

```bash
git clone https://github.com/F-Shabrawy/ml-image-classifiers.git
cd ml-image-classifiers
pip install -r requirements.txt
```

<details>
<summary><b>Dependencies</b></summary>

| Package | Purpose |
|---|---|
| `numpy >=1.21` | Core linear algebra and array operations |
| `matplotlib >=3.4` | Plotting loss curves, confusion matrices |
| `scikit-learn >=0.24` | MNIST download via `fetch_openml` **only** |
| `scikit-image >=0.19` | HOG feature extraction |
| `pandas >=1.3` | Data handling utilities |
| `jupyter >=1.0` | Interactive notebook exploration |

> `scikit-learn` is **not** used for any model fitting or evaluation — only for dataset retrieval.

</details>

### Training

```bash
python train_final.py
```

The script will:

1. Fetch MNIST via OpenML (cached after the first download)
2. Extract HOG features (orientations=9, 4×4 cells, 3×3 blocks)
3. Split into train / validation / test (80 / 10 / 10)
4. Train **binary** logistic regression (digit `8` vs. rest)
5. Train **multiclass** softmax regression (all 10 digits)
6. Print evaluation metrics and save weights to `models/`

---

## Results

### Binary Classification — Digit `8` vs. Rest

> Training config: 500 iterations · α = 1.3 · HOG features (or=9, ppc=4×4, cpb=3×3)

| Metric | Train | Validation | Test |
|:---|:---:|:---:|:---:|
| Accuracy | 98.16 % | 97.94 % | **98.10 %** |
| Precision | 91.77 % | 90.25 % | 90.48 % |
| Recall | 89.12 % | 89.50 % | 88.65 % |
| F1-Score | 90.42 % | 89.87 % | 89.55 % |

### Multiclass Softmax — 10-Class

> Training config: 500 iterations · α = 1.3 · L2 λ = 0.01

| Metric | Train | Validation | Test |
|:---|:---:|:---:|:---:|
| Accuracy | 97.14 % | 96.61 % | **96.87 %** |
| Precision (macro) | 97.13 % | 96.60 % | 96.85 % |
| Recall (macro) | 97.13 % | 96.61 % | 96.84 % |
| F1-Score (macro) | 97.12 % | 96.60 % | 96.84 % |

### Key Observations

- **HOG features boosted recall from ~75 % (raw pixels) to >88 %**, demonstrating that edge-orientation histograms capture digit structure far more effectively than flat pixel intensities.
- The train–validation gap stays **under 1 %** across all metrics, indicating strong generalization with no overfitting.
- Convergence is reached in **500 iterations** with HOG versus 15 000 iterations required with raw pixels — a significant speedup.
- The softmax model achieves **96.87 % test accuracy** on the full 10-class task, validating that the gradient descent framework scales cleanly from binary to multiclass problems.

---

## Architecture

### Project Structure

```
ml-image-classifiers/
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # MNIST fetch, HOG extraction, train/val/test split
│   ├── logistic_regression.py     # Binary classifier: sigmoid, gradient descent, save/load
│   ├── softmax_regression.py      # Multiclass classifier: softmax, L1/L2, cross-validation
│   └── utils.py                   # Metrics (accuracy, precision, recall, F1, confusion matrix)
├── notebooks/
│   ├── modelTrial.ipynb           # Exploratory analysis & hyperparameter experiments
│   ├── logistic_regression.ipynb  # Binary model deep-dive
│   ├── softmax.ipynb              # Multiclass model deep-dive
│   └── final_training.ipynb       # End-to-end training notebook
├── models/                        # Saved weight files (.npy)
├── results/plots/                 # Generated confusion matrices & loss curves
├── reports/                       # Supporting mathematical documentation & figures
├── tests/                         # Unit test directory
├── train_final.py                 # CLI entry point — trains & evaluates both models
├── requirements.txt
└── README.md
```

### Design Decisions

| Decision | Rationale |
|---|---|
| **No `sklearn` estimators** | Forces explicit implementation of every algorithmic step for educational clarity |
| **HOG over raw pixels** | Captures directional edge patterns; dramatically improves convergence speed and recall |
| **3-way data split** | Validation set enables early-stopping diagnostics without contaminating the test set |
| **`.npy` serialization** | NumPy's native binary format — fast, zero-dependency, minimal disk footprint |
| **Threshold tuning (binary)** | Decision threshold set to 0.4 instead of 0.5 to improve recall on the minority class |

---

## Roadmap

| Feature | Status |
|---|---|
| Multiclass softmax regression | ✅ Complete |
| L1 / L2 regularization | ✅ Complete |
| K-fold cross-validation | ✅ Complete |

---

<div align="center">

Built from scratch with **NumPy** and mathematical curiosity.

</div>