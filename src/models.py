from typing import Tuple

import numpy as np


class SoftmaxMath:
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
        correct_class_probs = probs[np.arange(y_true.shape[0]), y_true]
        return -np.mean(np.log(correct_class_probs + 1e-15))


class SoftmaxRegressionClassifier:
    def __init__(self, input_dim: int, n_classes: int) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.W = np.zeros((input_dim, n_classes))
        self.b = np.zeros((1, n_classes))

    def initialize(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((self.input_dim, self.n_classes)) * 0.1
        self.b = np.zeros((1, self.n_classes))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.W) + self.b
        return SoftmaxMath.softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        probs = self.predict_proba(X)
        loss = SoftmaxMath.cross_entropy(probs, y)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        return loss, acc


class HiddenLayerClassifier:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = np.zeros((input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.zeros((hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

    def initialize(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * np.sqrt(1.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = rng.standard_normal((self.hidden_dim, self.output_dim)) * np.sqrt(1.0 / self.hidden_dim)
        self.b2 = np.zeros((1, self.output_dim))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = np.dot(X, self.W1) + self.b1
        h = np.tanh(z1)
        logits = np.dot(h, self.W2) + self.b2
        return z1, h, logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, logits = self.forward(X)
        return SoftmaxMath.softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        probs = self.predict_proba(X)
        loss = SoftmaxMath.cross_entropy(probs, y)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        return loss, acc
