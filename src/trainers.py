from typing import List, Tuple

import numpy as np

from .models import HiddenLayerClassifier, SoftmaxMath, SoftmaxRegressionClassifier


History = List[List[float]]


class SoftmaxTrainer:
    def __init__(self, epochs: int, learning_rate: float, lambda_reg: float) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def fit(
        self,
        model: SoftmaxRegressionClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> History:
        n_samples = X_train.shape[0]
        history: History = []

        for epoch in range(self.epochs):
            probs = model.predict_proba(X_train)
            train_ce = SoftmaxMath.cross_entropy(probs, y_train)
            train_loss = train_ce + 0.5 * self.lambda_reg * np.sum(np.square(model.W))

            dZ = probs.copy()
            dZ[np.arange(n_samples), y_train] -= 1
            dZ /= n_samples

            dW = np.dot(X_train.T, dZ) + self.lambda_reg * model.W
            db = np.sum(dZ, axis=0, keepdims=True)

            model.W -= self.learning_rate * dW
            model.b -= self.learning_rate * db

            val_loss, val_acc = model.evaluate(X_val, y_val)
            history.append([epoch, train_loss, val_loss, val_acc])

        return history


class HiddenLayerTrainer:
    def __init__(self, epochs: int, learning_rate: float, lambda_reg: float) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

    def fit(
        self,
        model: HiddenLayerClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> History:
        n_samples = X_train.shape[0]
        history: History = []

        for epoch in range(self.epochs):
            _, h, logits = model.forward(X_train)
            probs = SoftmaxMath.softmax(logits)

            train_ce = SoftmaxMath.cross_entropy(probs, y_train)
            reg_term = 0.5 * self.lambda_reg * (
                np.sum(np.square(model.W1)) + np.sum(np.square(model.W2))
            )
            train_loss = train_ce + reg_term

            dZ2 = probs.copy()
            dZ2[np.arange(n_samples), y_train] -= 1
            dZ2 /= n_samples

            dW2 = np.dot(h.T, dZ2) + self.lambda_reg * model.W2
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dh = np.dot(dZ2, model.W2.T)
            dZ1 = dh * (1 - np.power(h, 2))

            dW1 = np.dot(X_train.T, dZ1) + self.lambda_reg * model.W1
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            model.W1 -= self.learning_rate * dW1
            model.b1 -= self.learning_rate * db1
            model.W2 -= self.learning_rate * dW2
            model.b2 -= self.learning_rate * db2

            val_loss, val_acc = model.evaluate(X_val, y_val)
            history.append([epoch, train_loss, val_loss, val_acc])

        return history
