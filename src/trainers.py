from typing import List, Tuple

import numpy as np

from .models import cross_entropy, softmax


History = List[List[float]]


def train_softmax(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    lambda_reg: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, History]:
    np.random.seed(seed)
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = int(np.max(y_train)) + 1

    W = np.random.randn(n_features, n_classes) * 0.1
    b = np.zeros((1, n_classes))
    history: History = []

    for epoch in range(epochs):
        logits = np.dot(X_train, W) + b
        probs = softmax(logits)

        train_ce = cross_entropy(probs, y_train)
        train_loss = train_ce + 0.5 * lambda_reg * np.sum(np.square(W))

        dZ = probs.copy()
        dZ[np.arange(n_samples), y_train] -= 1
        dZ /= n_samples

        dW = np.dot(X_train.T, dZ) + lambda_reg * W
        db = np.sum(dZ, axis=0, keepdims=True)

        W -= learning_rate * dW
        b -= learning_rate * db

        logits_val = np.dot(X_val, W) + b
        probs_val = softmax(logits_val)
        val_loss = cross_entropy(probs_val, y_val)
        val_preds = np.argmax(probs_val, axis=1)
        val_acc = np.mean(val_preds == y_val)
        history.append([epoch, train_loss, val_loss, val_acc])

    return W, b, history


def train_hidden_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    lambda_reg: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, History]:
    np.random.seed(seed)
    n_samples = X_train.shape[0]
    input_dim = X_train.shape[1]
    output_dim = int(np.max(y_train)) + 1

    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1.0 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
    b2 = np.zeros((1, output_dim))

    history: History = []

    for epoch in range(epochs):
        z1 = np.dot(X_train, W1) + b1
        h = np.tanh(z1)

        logits = np.dot(h, W2) + b2
        probs = softmax(logits)

        train_ce = cross_entropy(probs, y_train)
        reg_term = 0.5 * lambda_reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        train_loss = train_ce + reg_term

        dZ2 = probs.copy()
        dZ2[np.arange(n_samples), y_train] -= 1
        dZ2 /= n_samples

        dW2 = np.dot(h.T, dZ2) + lambda_reg * W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dh = np.dot(dZ2, W2.T)
        dZ1 = dh * (1 - np.power(h, 2))

        dW1 = np.dot(X_train.T, dZ1) + lambda_reg * W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        z1_val = np.dot(X_val, W1) + b1
        h_val = np.tanh(z1_val)
        logits_val = np.dot(h_val, W2) + b2
        probs_val = softmax(logits_val)
        val_loss = cross_entropy(probs_val, y_val)
        val_preds = np.argmax(probs_val, axis=1)
        val_acc = np.mean(val_preds == y_val)
        history.append([epoch, train_loss, val_loss, val_acc])

    return W1, b1, W2, b2, history
