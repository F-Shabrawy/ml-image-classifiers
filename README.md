# MNIST Binary Image Classifier - Logistic Regression

## Project Overview

Implementation of **Logistic Regression from scratch** for binary image classification on MNIST dataset. This project emphasizes mathematical understanding and proper experimentation methodology.

**Course**: CSE382: Introduction to Machine Learning  
**Phase 1 Deadline**: Week 13  
**Phase 2 Deadline**: Week 15

## Phase 1: Binary Classification

### Objective
Build a complete binary image classification pipeline implementing logistic regression to distinguish between two selected MNIST classes.

### Project Structure
```
ml-image-classifiers/
├── src/                          # Core implementations
│   ├── __init__.py
│   ├── data_loader.py           # MNIST loading & preprocessing
│   ├── logistic_regression.py   # Model implementation (from scratch)
│   ├── utils.py                 # Helpers (metrics, normalization)
│   └── visualizations.py        # Plotting functions
├── notebooks/
│   └── logistic_regression_exploration.ipynb   # Experimentation
├── data/
│   ├── raw/                     # Original MNIST data
│   └── processed/               # Train/val/test splits
├── models/                      # Saved checkpoints
├── results/                     # Metrics & visualizations
├── tests/                       # Unit tests
└── reports/                     # Technical documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Load and preprocess MNIST
python -m src.data_loader

# Run experiments
jupyter notebook notebooks/logistic_regression_exploration.ipynb
```

## Model Details

| Component | Details |
|-----------|---------|
| **Algorithm** | Logistic Regression (Binary Classification) |
| **Loss Function** | Binary Cross-Entropy: $-[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| **Optimization** | Gradient Descent (Batch & SGD) |
| **Feature Extraction** | Image flattening + normalization |
| **Classes** | 2 selected MNIST digits |

## Mathematical Foundation

### Logistic Function (Sigmoid)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Prediction
$$\hat{y} = \sigma(w^T x + b)$$

### Gradient Update
$$w := w - \alpha \frac{1}{m}X^T(\hat{y} - y)$$

where $\alpha$ is learning rate and $m$ is batch size.

## Phase 1 Results

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

See `results/` directory for detailed analysis and confusion matrices.

## Phase 2: Multi-class Enhancement

Coming soon: Extension to 10-class classification with advanced techniques (hyperparameter tuning, regularization, ensemble methods).

## Implementation Notes

- ✅ All models implemented from scratch using NumPy
- ✅ scikit-learn used only for metrics & data preprocessing, not modeling
- ✅ Code includes mathematical documentation
- ✅ Comprehensive testing and validation splits

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- Binary Cross-Entropy: https://en.wikipedia.org/wiki/Cross_entropy
- Gradient Descent: https://en.wikipedia.org/wiki/Gradient_descent
