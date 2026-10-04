"""
Step 5: Exoplanet Detection Project — Train the Astronet-style CNN (PyTorch)

Loads exoplanet_dataset.npz (built in Step 4), trains a two-branch CNN
(global view + local view, matching the Shallue & Vanderburg 2018
architecture), and evaluates it with accuracy/precision/recall — not
just accuracy, since class imbalance and false-positive cost matter here.

Run this locally:
    pip install torch numpy scikit-learn matplotlib
    python step5_train_cnn.py

Note: with only ~40 examples (from a first Step 4 run), this is a proof
that the full pipeline works end-to-end, not a publishable result. Once
this runs cleanly, go back to step4_build_dataset.py, raise N_PER_CLASS
to 100+, rerun it (will take longer), and re-run this script on the
bigger file for a real result worth putting in a portfolio writeup.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

DATA_FILE = "exoplanet_dataset_large.npz"
EPOCHS = 40           # upper limit — early stopping will usually halt sooner
PATIENCE = 8          # stop if validation loss hasn't improved in this many epochs
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2
SEED = 42


class LightCurveDataset(Dataset):
    """Wraps the global/local view arrays + labels for PyTorch's DataLoader."""

    def __init__(self, X_global, X_local, y):
        # Conv1d expects shape (batch, channels, length) — add a channel dim
        self.X_global = torch.tensor(X_global, dtype=torch.float32).unsqueeze(1)
        self.X_local = torch.tensor(X_local, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_global[idx], self.X_local[idx], self.y[idx]


class ConvBranch(nn.Module):
    """
    One convolutional column (used for both the global and local view,
    with different pooling since the inputs are different lengths).
    Mirrors Astronet's idea of stacked CONV + MAXPOOL blocks, scaled down
    to suit a small dataset (the original paper used far more data).
    """

    def __init__(self, input_length, out_features=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        # Figure out the flattened size dynamically so this works for both
        # the 2001-length global view and the 201-length local view
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            flat_size = self.conv(dummy).numel()
        self.fc = nn.Sequential(
            nn.Linear(flat_size, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class AstronetStyleCNN(nn.Module):
    """Two branches (global + local) merged into a final classifier head."""

    def __init__(self, global_length, local_length):
        super().__init__()
        self.global_branch = ConvBranch(global_length, out_features=32)
        self.local_branch = ConvBranch(local_length, out_features=32)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            # No sigmoid here — BCEWithLogitsLoss applies it internally (more stable)
        )

    def forward(self, global_view, local_view):
        g = self.global_branch(global_view)
        l = self.local_branch(local_view)
        combined = torch.cat([g, l], dim=1)
        return self.classifier(combined).squeeze(1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for global_view, local_view, y in loader:
        global_view, local_view, y = global_view.to(device), local_view.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(global_view, local_view)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device):
    """Validation loss, tracked each epoch so it can be plotted alongside
    training loss — this is what reveals overfitting (train loss keeps
    dropping while val loss flattens or rises)."""
    model.eval()
    total_loss = 0.0
    for global_view, local_view, y in loader:
        global_view, local_view, y = global_view.to(device), local_view.to(device), y.to(device)
        logits = model(global_view, local_view)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for global_view, local_view, y in loader:
        global_view, local_view = global_view.to(device), local_view.to(device)
        logits = model(global_view, local_view)
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    X_global, X_local, y = data["X_global"], data["X_local"], data["y"]
    print(f"Loaded {len(y)} examples ({y.sum()} confirmed, {len(y) - y.sum()} false positives)")

    dataset = LightCurveDataset(X_global, X_local, y)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train examples: {n_train}, Validation examples: {n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = AstronetStyleCNN(global_length=X_global.shape[1], local_length=X_local.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = compute_val_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Checkpoint whenever validation loss improves — this protects
        # against overfitting by keeping the model from the point where it
        # generalized best, not just whatever it looks like after the last
        # epoch (which is often already overfit on a small dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "exoplanet_cnn_best.pt")
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} — train loss: {train_loss:.4f}  "
                  f"val loss: {val_loss:.4f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — validation loss hasn't "
                  f"improved in {PATIENCE} epochs.")
            break

    total_epochs_run = epoch
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch} "
          f"(saved to exoplanet_cnn_best.pt)")

    # Reload the best checkpoint for final evaluation, rather than
    # evaluating whatever the model happens to look like after the last
    # epoch
    model.load_state_dict(torch.load("exoplanet_cnn_best.pt"))

    print("\nEvaluating best checkpoint on validation set...")
    y_true, y_pred = evaluate(model, val_loader, device)

    acc = accuracy_score(y_true, y_pred)
    # zero_division=0 avoids a crash if a class is never predicted (common with tiny datasets)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n--- Validation Results ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of predicted planets, how many were real)")
    print(f"Recall:    {rec:.3f}  (of real planets, how many were caught)")
    print(f"Confusion matrix (rows=true, cols=predicted, order=[false_pos, confirmed]):")
    print(cm)

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, total_epochs_run + 1), train_losses, label="Train loss")
    axes[0].plot(range(1, total_epochs_run + 1), val_losses, label="Validation loss")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                     label=f"Best epoch ({best_epoch})")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["False Pos", "Confirmed"])
    axes[1].set_yticklabels(["False Pos", "Confirmed"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("\nSaved plot to training_results.png")
    print(f"Best model weights already saved to exoplanet_cnn_best.pt "
          f"(from epoch {best_epoch})")


if __name__ == "__main__":
    main()
