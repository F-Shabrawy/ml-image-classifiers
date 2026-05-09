import numpy as np
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def compute_loss(y, y_hat):
    m = y.shape[0]
    log_likelihood = -np.log(y_hat[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss
def calculate_gradient(X, y, y_hat):
    m = y.size
    y_one_hot = np.eye(y_hat.shape[1])[y]
    return X.T @ (y_hat - y_one_hot) / m
class SoftmaxRegression:
    def __init__(self, alpha=0.1, num_iter=200, tolerance=1e-4):
        self.alpha = alpha
        self.num_iter = num_iter
        self.tolerance = tolerance
        self.theta = None
        self.losses = []
        self.val_losses = []
        self.actual_iter = 0
        self.num_classes = None
    def fit(self, X, y, X_val=None, y_val=None):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.num_classes = len(np.unique(y))
        self.theta = np.zeros((X_b.shape[1], self.num_classes))
        self.losses = []
        self.val_losses = []
        for i in range(self.num_iter):
            y_hat = softmax(X_b @ self.theta)
            loss = compute_loss(y, y_hat)
            self.losses.append(loss)
            self.actual_iter = i + 1
            if X_val is not None and y_val is not None:
                X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
                y_val_hat = softmax(X_val_b @ self.theta)
                val_loss = compute_loss(y_val, y_val_hat)
                self.val_losses.append(val_loss)
            grad = calculate_gradient(X_b, y, y_hat)
            self.theta -= self.alpha * grad
            if np.linalg.norm(grad) < self.tolerance:
                break
        return self.theta
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_hat = softmax(X_b @ self.theta)
        return np.argmax(y_hat, axis=1)
    def save_model(self, file_path):
        np.save(file_path, self.theta)
    def load_model(self, file_path):
        self.theta = np.load(file_path)