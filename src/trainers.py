"""Training loops for linear and one-hidden-layer softmax models."""

from typing import List

import numpy as np

from .config import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    DEFAULT_SEED,
    HIDDEN_OPTIMIZER_DEFAULT,
    MOMENTUM_BETA,
)
from .models import HiddenLayerClassifier, SoftmaxMath, SoftmaxRegressionClassifier


History = List[List[float]]


class SoftmaxTrainer:
    """Mini-batch trainer for softmax regression with L2 regularization."""

    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        lambda_reg: float,
        batch_size: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.seed = seed

    def fit(
        self,
        model: SoftmaxRegressionClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> History:
        """Train model and return epoch history rows.

        History row format: [epoch, train_loss, val_loss, val_accuracy].
        """
        n_samples = X_train.shape[0]
        batch_size = self.batch_size or n_samples
        rng = np.random.default_rng(self.seed)
        history: History = []

        for epoch in range(self.epochs):
            # Deterministic per-epoch shuffling controlled by trainer seed.
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                probs_batch = model.predict_proba(X_batch)
                dZ = probs_batch.copy()
                dZ[np.arange(y_batch.shape[0]), y_batch] -= 1
                dZ /= y_batch.shape[0]

                dW = np.dot(X_batch.T, dZ) + self.lambda_reg * model.W
                db = np.sum(dZ, axis=0, keepdims=True)

                model.W -= self.learning_rate * dW
                model.b -= self.learning_rate * db

            probs = model.predict_proba(X_train)
            train_ce = SoftmaxMath.cross_entropy(probs, y_train)
            train_loss = train_ce + 0.5 * self.lambda_reg * np.sum(np.square(model.W))

            val_loss, val_acc = model.evaluate(X_val, y_val)
            history.append([epoch, train_loss, val_loss, val_acc])

        return history


class HiddenLayerTrainer:
    """Mini-batch trainer for one-hidden-layer tanh network."""

    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        lambda_reg: float,
        batch_size: int | None = None,
        seed: int = DEFAULT_SEED,
        optimizer: str = HIDDEN_OPTIMIZER_DEFAULT,
        momentum_beta: float = MOMENTUM_BETA,
        adam_beta1: float = ADAM_BETA1,
        adam_beta2: float = ADAM_BETA2,
        adam_epsilon: float = ADAM_EPSILON,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.seed = seed
        self.optimizer = optimizer
        self.momentum_beta = momentum_beta
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def fit(
        self,
        model: HiddenLayerClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> History:
        """Train model and return epoch history rows.

        History row format: [epoch, train_loss, val_loss, val_accuracy].
        """
        n_samples = X_train.shape[0]
        batch_size = self.batch_size or n_samples
        rng = np.random.default_rng(self.seed)
        history: History = []

        v_W1 = np.zeros_like(model.W1)
        v_b1 = np.zeros_like(model.b1)
        v_W2 = np.zeros_like(model.W2)
        v_b2 = np.zeros_like(model.b2)

        m_W1 = np.zeros_like(model.W1)
        m_b1 = np.zeros_like(model.b1)
        m_W2 = np.zeros_like(model.W2)
        m_b2 = np.zeros_like(model.b2)

        s_W1 = np.zeros_like(model.W1)
        s_b1 = np.zeros_like(model.b1)
        s_W2 = np.zeros_like(model.W2)
        s_b2 = np.zeros_like(model.b2)
        adam_step = 0

        for epoch in range(self.epochs):
            # Deterministic per-epoch shuffling controlled by trainer seed.
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                _, h_batch, logits_batch = model.forward(X_batch)
                probs_batch = SoftmaxMath.softmax(logits_batch)

                dZ2 = probs_batch.copy()
                dZ2[np.arange(y_batch.shape[0]), y_batch] -= 1
                dZ2 /= y_batch.shape[0]

                dW2 = np.dot(h_batch.T, dZ2) + self.lambda_reg * model.W2
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dh = np.dot(dZ2, model.W2.T)
                dZ1 = dh * (1 - np.power(h_batch, 2))

                dW1 = np.dot(X_batch.T, dZ1) + self.lambda_reg * model.W1
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                if self.optimizer == "sgd":
                    model.W1 -= self.learning_rate * dW1
                    model.b1 -= self.learning_rate * db1
                    model.W2 -= self.learning_rate * dW2
                    model.b2 -= self.learning_rate * db2
                elif self.optimizer == "momentum":
                    v_W1 = self.momentum_beta * v_W1 + dW1
                    v_b1 = self.momentum_beta * v_b1 + db1
                    v_W2 = self.momentum_beta * v_W2 + dW2
                    v_b2 = self.momentum_beta * v_b2 + db2

                    model.W1 -= self.learning_rate * v_W1
                    model.b1 -= self.learning_rate * v_b1
                    model.W2 -= self.learning_rate * v_W2
                    model.b2 -= self.learning_rate * v_b2
                elif self.optimizer == "adam":
                    adam_step += 1

                    m_W1 = self.adam_beta1 * m_W1 + (1.0 - self.adam_beta1) * dW1
                    m_b1 = self.adam_beta1 * m_b1 + (1.0 - self.adam_beta1) * db1
                    m_W2 = self.adam_beta1 * m_W2 + (1.0 - self.adam_beta1) * dW2
                    m_b2 = self.adam_beta1 * m_b2 + (1.0 - self.adam_beta1) * db2

                    s_W1 = self.adam_beta2 * s_W1 + (1.0 - self.adam_beta2) * np.square(dW1)
                    s_b1 = self.adam_beta2 * s_b1 + (1.0 - self.adam_beta2) * np.square(db1)
                    s_W2 = self.adam_beta2 * s_W2 + (1.0 - self.adam_beta2) * np.square(dW2)
                    s_b2 = self.adam_beta2 * s_b2 + (1.0 - self.adam_beta2) * np.square(db2)

                    mhat_W1 = m_W1 / (1.0 - self.adam_beta1**adam_step)
                    mhat_b1 = m_b1 / (1.0 - self.adam_beta1**adam_step)
                    mhat_W2 = m_W2 / (1.0 - self.adam_beta1**adam_step)
                    mhat_b2 = m_b2 / (1.0 - self.adam_beta1**adam_step)

                    shat_W1 = s_W1 / (1.0 - self.adam_beta2**adam_step)
                    shat_b1 = s_b1 / (1.0 - self.adam_beta2**adam_step)
                    shat_W2 = s_W2 / (1.0 - self.adam_beta2**adam_step)
                    shat_b2 = s_b2 / (1.0 - self.adam_beta2**adam_step)

                    model.W1 -= self.learning_rate * mhat_W1 / (np.sqrt(shat_W1) + self.adam_epsilon)
                    model.b1 -= self.learning_rate * mhat_b1 / (np.sqrt(shat_b1) + self.adam_epsilon)
                    model.W2 -= self.learning_rate * mhat_W2 / (np.sqrt(shat_W2) + self.adam_epsilon)
                    model.b2 -= self.learning_rate * mhat_b2 / (np.sqrt(shat_b2) + self.adam_epsilon)
                else:
                    raise ValueError(f"Unsupported optimizer for hidden layer: {self.optimizer}")

            _, h, logits = model.forward(X_train)
            probs = SoftmaxMath.softmax(logits)
            train_ce = SoftmaxMath.cross_entropy(probs, y_train)
            reg_term = 0.5 * self.lambda_reg * (
                np.sum(np.square(model.W1)) + np.sum(np.square(model.W2))
            )
            train_loss = train_ce + reg_term

            val_loss, val_acc = model.evaluate(X_val, y_val)
            history.append([epoch, train_loss, val_loss, val_acc])

        return history
