import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
import os
import sys
sys.path.append('..')
from data_preprocessing import preprocess_inference


def add_bias(X: np.ndarray) -> np.ndarray:
    """Add bias column of ones to design matrix."""
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


class LinearRegressionGD:
    """Linear regression implemented with gradient descent."""
    def __init__(self, lr=0.001, n_iter=3000, clip=1000.0):
        self.lr = lr          # Learning rate for gradient descent
        self.n_iter = n_iter  # Maximum number of iterations
        self.clip = clip      # Gradient clipping threshold
        self.w = None         # Weight vector (including bias)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model using gradient descent."""
        Xb = add_bias(X)  # Add bias term (column of ones)
        self.w = np.zeros(Xb.shape[1])  # Initialize weights to zeros
        n = len(Xb)

        for _ in range(self.n_iter):
            preds = np.dot(Xb, self.w)  # Make predictions
            res = preds - y      # Calculate residuals
            grad = (2 / n) * np.dot(Xb.T, res)  # Compute gradient

            # Clip gradient to prevent exploding gradients
            g_norm = np.linalg.norm(grad)
            if g_norm > self.clip:
                grad = grad * (self.clip / g_norm)

            # Update weights
            new_w = self.w - self.lr * grad
            if not np.all(np.isfinite(new_w)):
                break
            self.w = new_w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return np.dot(add_bias(X), self.w)


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Generate polynomial features up to given degree."""
    if degree <= 1:
        return X
    feats = [X]
    for d in range(2, degree + 1):
        feats.append(X ** d)
    return np.concatenate(feats, axis=1)


class PolynomialRegressionGD:
    def __init__(self, degree=2, lr=0.001, n_iter=4000, clip=1000.0):
        self.degree = degree
        self.lr = lr
        self.n_iter = n_iter
        self.clip = clip
        self.model = LinearRegressionGD(lr=lr, n_iter=n_iter, clip=clip)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_poly = polynomial_features(X, self.degree)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = polynomial_features(X, self.degree)
        return self.model.predict(X_poly)


class RidgeRegressionGD:
    """Ridge regression (L2 regularization) implemented with gradient descent."""
    def __init__(self, lr=0.001, n_iter=3000, alpha=0.5):
        self.lr = lr          # Learning rate for gradient descent
        self.n_iter = n_iter  # Maximum number of iterations
        self.alpha = alpha    # L2 regularization strength
        self.w = None         # Weight vector (including bias)

    def fit(self, X, y):
        """Train the model using gradient descent with L2 regularization."""
        Xb = add_bias(X)  # Add bias term (column of ones)
        self.w = np.zeros(Xb.shape[1])  # Initialize weights to zeros
        n = len(Xb)

        for _ in range(self.n_iter):
            preds = np.dot(Xb, self.w)  # Make predictions
            # L2 regularization (don't penalize bias term)
            reg = self.alpha * np.concatenate([[0.0], self.w[1:]])
            grad = (2 / n) * np.dot(Xb.T, (preds - y)) + reg  # Compute gradient with regularization

            # Update weights
            new_w = self.w - self.lr * grad
            if not np.all(np.isfinite(new_w)):
                break
            self.w = new_w
        return self

    def predict(self, X):
        """Make predictions on new data."""
        return np.dot(add_bias(X), self.w)


class LassoRegressionGD:
    """Lasso regression (L1 regularization) implemented with gradient descent."""
    def __init__(self, lr=0.001, n_iter=4000, alpha=0.01):
        self.lr = lr          # Learning rate for gradient descent
        self.n_iter = n_iter  # Maximum number of iterations
        self.alpha = alpha    # L1 regularization strength
        self.w = None         # Weight vector (including bias)

    def fit(self, X, y):
        """Train the model using gradient descent with L1 regularization."""
        Xb = add_bias(X)  # Add bias term (column of ones)
        self.w = np.zeros(Xb.shape[1])  # Initialize weights to zeros
        n = len(Xb)

        for _ in range(self.n_iter):
            preds = np.dot(Xb, self.w)  # Make predictions
            grad = (2 / n) * np.dot(Xb.T, (preds - y))  # Base gradient
            # L1 regularization (don't penalize bias term)
            l1 = self.alpha * np.concatenate([[0.0], np.sign(self.w[1:])])
            full_grad = grad + l1  # Add L1 penalty to gradient

            # Update weights
            new_w = self.w - self.lr * full_grad
            if not np.all(np.isfinite(new_w)):
                break
            self.w = new_w
        return self

    def predict(self, X):
        """Make predictions on new data."""
        return np.dot(add_bias(X), self.w)


# Metrics

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)


# Helpers

def load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metrics_output_path", required=True)
    parser.add_argument("--predictions_output_path", required=True)
    parser.add_argument("--target", default="Price")
    args = parser.parse_args()

# for testing
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. Train a model first with train_model.py."
        )
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data file not found: {args.data_path}. Provide a valid CSV path with --data_path."
        )

    # Load trained model and preprocessing objects
    bundle = load_bundle(args.model_path)
    model = bundle["model"]
    processors = bundle["processors"]

    # Load test data and apply same preprocessing as training
    df = pd.read_csv(args.data_path)
    X_df, y = preprocess_inference(df, processors)
    X = X_df.to_numpy(dtype=float)

    # Generate predictions
    preds = model.predict(X)

    # Save predictions to file (one per line, no header)
    os.makedirs(os.path.dirname(args.predictions_output_path), exist_ok=True)
    with open(args.predictions_output_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(f"{p}\n")

    # Calculate and save metrics (use NaN if no true labels available)
    os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
    if y is not None:
        m, r, r2v = mse(y, preds), rmse(y, preds), r2_score(y, preds)
    else:
        m = r = r2v = float("nan")

    with open(args.metrics_output_path, "w", encoding="utf-8") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {m:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {r:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2v:.2f}\n")

    print("Predictions written:", args.predictions_output_path)
    print("Metrics written:", args.metrics_output_path)


if __name__ == "__main__":
    main()

