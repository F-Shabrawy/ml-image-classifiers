import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size
    return X.T @ (sigmoid(X @ theta) - y) / m

def compute_loss(y, y_hat):
    m = y.size
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def predict_prob(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

class LogisticRegression:
    def __init__(self, alpha=0.1, num_iter=100, tolerance=1e-7):
        self.alpha = alpha
        self.num_iter = num_iter
        self.tolerance = tolerance
        self.theta = None
        self.losses = []
        self.val_losses = []
        self.actual_iter = 0

    def fit(self, X, y, X_val=None, y_val=None):
        # stick a 1 column on for bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X_b.shape[1])

        self.losses = []
        self.val_losses = []

        for i in range(self.num_iter):
            y_hat = sigmoid(X_b @ self.theta)
            loss = compute_loss(y, y_hat)
            self.losses.append(loss)
            self.actual_iter = i + 1

            # track validation loss too, helps spot overfitting
            if X_val is not None and y_val is not None:
                X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]
                y_val_hat = sigmoid(X_val_b @ self.theta)
                val_loss = compute_loss(y_val, y_val_hat)
                self.val_losses.append(val_loss)

            grad = calculate_gradient(self.theta, X_b, y)
            self.theta -= self.alpha * grad

            # stop if gradient gets flat enough
            if np.linalg.norm(grad) < self.tolerance:
                break
        return self.theta

    def predict(self, X, threshold=0.5):
        return (predict_prob(X, self.theta) >= threshold).astype(int)

    def save_model(self, file_path):
        np.save(file_path, self.theta)

    def load_model(self, file_path):
        self.theta = np.load(file_path)
