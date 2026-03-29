from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "raw_data_and_scripts" / "data" / "linear_gaussian.npz"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    correct_class_probs = probs[np.arange(y_true.shape[0]), y_true]
    return -np.mean(np.log(correct_class_probs + 1e-15))


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, out_path: Path) -> None:
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    logits_grid = np.dot(grid, W) + b
    probs_grid = softmax(logits_grid)
    preds_grid = np.argmax(probs_grid, axis=1).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds_grid, levels=2, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=18, edgecolors="k", linewidths=0.2, cmap="coolwarm")
    plt.title("Linear Gaussian: Softmax Decision Boundary")
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
n_features = X_train.shape[1]
n_classes = 2
W = np.random.randn(n_features, n_classes) * 0.1
b = np.zeros((1, n_classes))

eta = 0.1
epochs = 200
lambda_reg = 0.01

history = []

for epoch in range(epochs):
    logits = np.dot(X_train, W) + b
    probabilities = softmax(logits)

    train_ce = cross_entropy(probabilities, y_train)
    loss_reg = 0.5 * lambda_reg * np.sum(np.square(W))
    train_loss = train_ce + loss_reg

    dZ = probabilities.copy()
    dZ[np.arange(n_samples), y_train] -= 1
    dZ /= n_samples

    dW = np.dot(X_train.T, dZ) + lambda_reg * W
    db = np.sum(dZ, axis=0, keepdims=True)

    W -= eta * dW
    b -= eta * db

    logits_val = np.dot(X_val, W) + b
    probabilities_val = softmax(logits_val)
    val_loss = cross_entropy(probabilities_val, y_val)
    val_preds = np.argmax(probabilities_val, axis=1)
    val_acc = np.mean(val_preds == y_val)
    history.append([epoch, train_loss, val_loss, val_acc])

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc*100:.2f}%")

logits_test = np.dot(X_test, W) + b
probabilities_test = softmax(logits_test)
test_loss = cross_entropy(probabilities_test, y_test)
preds_test = np.argmax(probabilities_test, axis=1)
test_acc = np.mean(preds_test == y_test)

metrics_path = RESULTS_DIR / "linear_gaussian_softmax_metrics.csv"
np.savetxt(
    metrics_path,
    np.array(history),
    delimiter=",",
    header="epoch,train_loss,val_loss,val_accuracy",
    comments="",
)

plot_path = FIGURES_DIR / "linear_gaussian_softmax_boundary.png"
plot_decision_boundary(X_test, y_test, W, b, plot_path)

print("-" * 40)
print("FINAL TEST RESULTS (LINEAR GAUSSIAN + SOFTMAX)")
print(f"Test Cross-Entropy: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Saved metrics: {metrics_path}")
print(f"Saved figure: {plot_path}")
print("-" * 40)

