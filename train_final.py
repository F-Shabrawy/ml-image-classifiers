import os
from src.data_loader import load_mnist
from src.logistic_regression import LogisticRegression
from src.utils import print_metrics

def train_and_save():
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist()

    # hog features converge faster, fewer iters needed vs raw pixels
    model = LogisticRegression(alpha=1.3, num_iter=2000)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    os.makedirs('models', exist_ok=True)
    model_path = 'models/hog_lr_13_2000.npy'
    model.save_model(model_path)

    print("training complete and model saved.")
    print(f"iterations: {model.actual_iter}")
    if model.losses:
        print(f"Final Train Loss: {model.losses[-1]:.4f}")
    if model.val_losses:
        print(f"Final Val Loss: {model.val_losses[-1]:.4f}")

    y_pred_train = model.predict(X_train, threshold=0.4)
    y_pred_val = model.predict(X_val, threshold=0.4)
    y_pred_test = model.predict(X_test, threshold=0.4)
    print_metrics(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, model)

if __name__ == "__main__":
    train_and_save()
