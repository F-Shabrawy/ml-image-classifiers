import os
from src.data_loader import load_mnist
from src.logistic_regression import LogisticRegression
from src.utils import print_metrics, plot_confusion_matrix

def train_and_save():
    X_train, X_test, y_train, y_test = load_mnist()
    model = LogisticRegression(alpha=1.1, num_iter=15000)
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/final_logistic_regression_11_15000.npy'
    model.save_model(model_path)
    
    print("Training complete and model saved.")
    print(f"Actual iterations run: {model.actual_iter}")
    if model.losses:
        print(f"Final Loss: {model.losses[-1]:.4f}")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print_metrics(y_train, y_pred_train, y_test, y_pred_test)
    plot_confusion_matrix(y_test, y_pred_test, "Confusion Matrix (Logistic Regression)")

if __name__ == "__main__":
    train_and_save()
