import os
from src.data_loader import load_mnist, load_mnist_multiclass
from src.logistic_regression import LogisticRegression
from src.softmax_regression import SoftmaxRegression
from src.utils import print_metrics

def train_binary():
    print("=" * 50)
    print("Training Logistic Regression (Binary: 8 vs. rest)")
    print("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()

    model = LogisticRegression(alpha=1.3, num_iter=2000)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    os.makedirs('models', exist_ok=True)
    model.save_model('models/logistic_binary.npy')

    print(f"iterations: {model.actual_iter}")
    if model.losses:
        print(f"Final Train Loss: {model.losses[-1]:.4f}")
    if model.val_losses:
        print(f"Final Val Loss: {model.val_losses[-1]:.4f}")

    y_pred_train = model.predict(X_train, threshold=0.4)
    y_pred_val = model.predict(X_val, threshold=0.4)
    y_pred_test = model.predict(X_test, threshold=0.4)

    print("\n--- Binary Classification Results ---")
    print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model)

def train_multiclass():
    print("\n" + "=" * 50)
    print("Training Softmax Regression (10-class)")
    print("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_multiclass()

    model = SoftmaxRegression(alpha=1.3, num_iter=2000, lambda_=0.01, reg_type='l2')
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    os.makedirs('models', exist_ok=True)
    model.save_model('models/softmax_multiclass.npy')

    print(f"iterations: {model.actual_iter}")
    if model.losses:
        print(f"Final Train Loss: {model.losses[-1]:.4f}")
    if model.val_losses:
        print(f"Final Val Loss: {model.val_losses[-1]:.4f}")

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print("\n--- Multi-class Classification Results ---")
    print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model)

if __name__ == "__main__":
    train_binary()
    train_multiclass()
    print("\n" + "=" * 50)
    print("Training complete! Models saved to models/")
    print("=" * 50)