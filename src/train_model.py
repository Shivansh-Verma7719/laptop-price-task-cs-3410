
import argparse
import os
import pickle
from typing import Dict, Any
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from data_preprocessing import preprocess_training

def add_bias(X):
    # Add bias column of ones to design matrix
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

def train_valid_split(X, y, valid_ratio=0.2, seed=42):
    # Random train/validation split
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - valid_ratio))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]
    return X_train, y_train, X_val, y_val

class LinearRegressionGD:
    # Linear regression implemented with gradient descent
    def __init__(self, lr=0.001, n_iter=3000, clip=1000.0):
        self.lr = lr          # Learning rate for gradient descent
        self.n_iter = n_iter  # Maximum number of iterations
        self.clip = clip      # Gradient clipping threshold
        self.w = None         # Weight vector (including bias)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Train the model using gradient descent
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
                print("Stopping linear GD early due to non-finite weights")
                break
            self.w = new_w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Make predictions on new data
        return np.dot(add_bias(X), self.w)

class RidgeRegressionGD:
    # Ridge regression (L2 regularization) implemented with gradient descent
    def __init__(self, lr=0.001, n_iter=3000, alpha=0.5, clip=1000.0):
        self.lr = lr          # Learning rate for gradient descent
        self.n_iter = n_iter  # Maximum number of iterations
        self.alpha = alpha    # L2 regularization strength
        self.clip = clip      # Gradient clipping threshold
        self.w = None         # Weight vector (including bias)

    def fit(self, X, y):
        # Train the model using gradient descent with L2 regularization
        Xb = add_bias(X)  # Add bias term (column of ones)
        self.w = np.zeros(Xb.shape[1])  # Initialize weights to zeros
        n = len(Xb)

        for _ in range(self.n_iter):
            preds = np.dot(Xb, self.w)  # Make predictions
            # L2 regularization (don't penalize bias term)
            reg = self.alpha * np.concatenate([[0.0], self.w[1:]])
            grad = (2 / n) * np.dot(Xb.T, (preds - y)) + reg  # Compute gradient with regularization

            # Clip gradient to prevent exploding gradients
            g_norm = np.linalg.norm(grad)
            if g_norm > self.clip:
                grad = grad * (self.clip / g_norm)

            # Update weights
            new_w = self.w - self.lr * grad
            if not np.all(np.isfinite(new_w)):
                print("Stopping ridge GD early due to non-finite weights")
                break
            self.w = new_w
        return self

    def predict(self, X):
        # Make predictions on new data
        return np.dot(add_bias(X), self.w)

class LassoRegressionGD:
    # Lasso regression (L1 regularization) implemented with gradient descent
    def __init__(self, lr=0.001, n_iter=4000, alpha=0.01, clip=1000.0):
        self.lr = lr          # Learning rate for gradient descent
        self.n_iter = n_iter  # Maximum number of iterations
        self.alpha = alpha    # L1 regularization strength
        self.clip = clip      # Gradient clipping threshold
        self.w = None         # Weight vector (including bias)

    def fit(self, X, y):
        # Train the model using gradient descent with L1 regularization
        Xb = add_bias(X)  # Add bias term (column of ones)
        self.w = np.zeros(Xb.shape[1])  # Initialize weights to zeros
        n = len(Xb)

        for _ in range(self.n_iter):
            preds = np.dot(Xb, self.w)  # Make predictions
            grad = (2 / n) * np.dot(Xb.T, (preds - y))  # Base gradient
            # L1 regularization (don't penalize bias term)
            l1 = self.alpha * np.concatenate([[0.0], np.sign(self.w[1:])])
            full_grad = grad + l1  # Add L1 penalty to gradient

            # Clip gradient to prevent exploding gradients
            g_norm = np.linalg.norm(full_grad)
            if g_norm > self.clip:
                full_grad = full_grad * (self.clip / g_norm)

            # Update weights
            new_w = self.w - self.lr * full_grad
            if not np.all(np.isfinite(new_w)):
                print("Stopping lasso GD early due to non-finite weights")
                break
            self.w = new_w
        return self

    def predict(self, X):
        # Make predictions on new data
        return np.dot(add_bias(X), self.w)

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    # Generate polynomial features up to given degree
    if degree <= 1:
        return X
    feats = [X]
    for d in range(2, degree + 1):
        feats.append(X ** d)
    return np.concatenate(feats, axis=1)

class PolynomialRegressionGD:
    # Polynomial regression implemented using linear regression on polynomial features
    def __init__(self, degree=2, lr=0.001, n_iter=4000, clip=1000.0):
        self.degree = degree    # Degree of polynomial features
        self.lr = lr           # Learning rate for gradient descent
        self.n_iter = n_iter   # Maximum number of iterations
        self.clip = clip       # Gradient clipping threshold
        self.model = LinearRegressionGD(lr=lr, n_iter=n_iter, clip=clip)  # Underlying linear model

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Train the model by transforming features to polynomial and fitting linear model
        X_poly = polynomial_features(X, self.degree)  # Generate polynomial features
        self.model.fit(X_poly, y)  # Fit linear model on polynomial features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Make predictions by transforming features to polynomial and using linear model
        X_poly = polynomial_features(X, self.degree)  # Generate polynomial features
        return self.model.predict(X_poly)

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot

def train_and_select(X_train, y_train, X_val, y_val):
    """Train all models and return best by validation MSE."""
    models: Dict[str, Any] = {
        "linear": LinearRegressionGD(lr=0.01, n_iter=5000),
        "ridge": RidgeRegressionGD(lr=0.01, n_iter=5000, alpha=0.1),
        "lasso": LassoRegressionGD(lr=0.01, n_iter=5000, alpha=0.001),
        "poly_d2": PolynomialRegressionGD(degree=2, lr=0.01, n_iter=5000),
    }
    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        results[name] = {"mse": mse(y_val, preds), "r2": r2_score(y_val, preds)}
        trained[name] = model
    best = min(results, key=lambda k: results[k]["mse"])
    return {"results": results, "trained": trained, "best": best}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--target", default="Price")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--save_prefix", default="regression_model")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    raw = pd.read_csv(args.data_path)
    X_df, y, processors = preprocess_training(raw, target=args.target)
    X = X_df.to_numpy(dtype=float)

    # Split
    X_tr, y_tr, X_val, y_val = train_valid_split(X, y, valid_ratio=0.2, seed=42)

    # Train and pick best
    outcome = train_and_select(X_tr, y_tr, X_val, y_val)

    # Save models
    os.makedirs(args.models_dir, exist_ok=True)
    idx = 1
    for name, model in outcome["trained"].items():
        if name == outcome["best"]:
            continue
        path = os.path.join(args.models_dir, f"{args.save_prefix}{idx}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": model, "processors": processors}, f)
        idx += 1

    final_path = os.path.join(args.models_dir, f"{args.save_prefix}_final.pkl")
    with open(final_path, "wb") as f:
        pickle.dump({"model": outcome["trained"][outcome["best"]], "processors": processors}, f)

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/train_model_selection.csv", "w", encoding="utf-8") as f:
        f.write("model,mse_val,r2_val\n")
        for name, m in outcome["results"].items():
            f.write(f"{name},{m['mse']:.4f},{m['r2']:.4f}\n")

    print("Best model:", outcome["best"], "->", final_path)

if __name__ == "__main__":
    main()
