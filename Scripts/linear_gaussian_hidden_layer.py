from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Starter-datasets-and-scripts" / "data" / "linear_gaussian.npz"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    correct_class_probs = probs[np.arange(y_true.shape[0]), y_true]
    return -np.mean(np.log(correct_class_probs + 1e-15))


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    out_path: Path,
) -> None:
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    z1 = np.dot(grid, W1) + b1
    h = np.tanh(z1)
    logits = np.dot(h, W2) + b2
    probs = softmax(logits)
    preds_grid = np.argmax(probs, axis=1).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds_grid, levels=2, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=18, edgecolors="k", linewidths=0.2, cmap="coolwarm")
    plt.title("Linear Gaussian: 1-Hidden-Layer Decision Boundary")
    plt.xlabel("x1 (standardized)")
    plt.ylabel("x2 (standardized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


with np.load(DATA_PATH) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std = np.where(std < 1e-12, 1.0, std)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

n_samples = X_train.shape[0]
input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = 2
lambda_reg = 0.01

W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1.0 / input_dim)
b1 = np.zeros((1, hidden_dim))

W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
b2 = np.zeros((1, output_dim))

eta = 0.3
epochs = 300

history = []

for epoch in range(epochs):
    z1 = np.dot(X_train, W1) + b1
    h = np.tanh(z1)

    logits = np.dot(h, W2) + b2
    probabilities = softmax(logits)

    train_ce = cross_entropy(probabilities, y_train)
    loss_reg = 0.5 * lambda_reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    train_loss = train_ce + loss_reg

    dZ2 = probabilities.copy()
    dZ2[np.arange(n_samples), y_train] -= 1
    dZ2 /= n_samples

    dW2 = np.dot(h.T, dZ2) + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dh = np.dot(dZ2, W2.T)
    dZ1 = dh * (1 - np.power(h, 2))

    dW1 = np.dot(X_train.T, dZ1) + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 -= eta * dW1
    b1 -= eta * db1
    W2 -= eta * dW2
    b2 -= eta * db2

    z1_val = np.dot(X_val, W1) + b1
    h_val = np.tanh(z1_val)
    logits_val = np.dot(h_val, W2) + b2
    probabilities_val = softmax(logits_val)

    val_loss = cross_entropy(probabilities_val, y_val)
    val_preds = np.argmax(probabilities_val, axis=1)
    val_acc = np.mean(val_preds == y_val)
    history.append([epoch, train_loss, val_loss, val_acc])

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc*100:.2f}%")

z1_test = np.dot(X_test, W1) + b1
h_test = np.tanh(z1_test)
logits_test = np.dot(h_test, W2) + b2
probabilities_test = softmax(logits_test)

test_loss = cross_entropy(probabilities_test, y_test)
preds_test = np.argmax(probabilities_test, axis=1)
test_acc = np.mean(preds_test == y_test)

metrics_path = RESULTS_DIR / "linear_gaussian_hidden_layer_metrics.csv"
np.savetxt(
    metrics_path,
    np.array(history),
    delimiter=",",
    header="epoch,train_loss,val_loss,val_accuracy",
    comments="",
)

plot_path = FIGURES_DIR / "linear_gaussian_hidden_layer_boundary.png"
plot_decision_boundary(X_test, y_test, W1, b1, W2, b2, plot_path)

print("-" * 40)
print("FINAL TEST RESULTS (LINEAR GAUSSIAN + 1-HIDDEN-LAYER)")
print(f"Test Cross-Entropy: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Saved metrics: {metrics_path}")
print(f"Saved figure: {plot_path}")
print("-" * 40)
