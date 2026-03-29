from typing import Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    correct_class_probs = probs[np.arange(y_true.shape[0]), y_true]
    return -np.mean(np.log(correct_class_probs + 1e-15))


def predict_softmax(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def predict_hidden_layer(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    z1 = np.dot(X, W1) + b1
    h = np.tanh(z1)
    logits = np.dot(h, W2) + b2
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def evaluate_softmax(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float]:
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    loss = cross_entropy(probs, y)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)
    return loss, acc


def evaluate_hidden_layer(
    X: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[float, float]:
    z1 = np.dot(X, W1) + b1
    h = np.tanh(z1)
    logits = np.dot(h, W2) + b2
    probs = softmax(logits)
    loss = cross_entropy(probs, y)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)
    return loss, acc
