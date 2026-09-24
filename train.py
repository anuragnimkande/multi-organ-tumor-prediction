"""
Training Script -- Hybrid Quantum-Classical Brain Tumor Detection
================================================================
Trains the HybridModel on the brain tumor MRI dataset and saves
the best checkpoint based on validation accuracy.

Differential Learning Rates:
    CNN backbone  : lr * 0.1   (protect pretrained weights)
    Classical MLP : lr         (standard)
    Quantum layer : lr * 10    (overcome barren-plateau vanishing gradients)
    Classifier    : lr         (standard)

Usage:
    python train.py [--epochs N] [--batch_size N] [--lr LR] [--data PATH]

Outputs:
    checkpoints/best_model.pth          - best validation model
    checkpoints/last_model.pth          - final epoch model
    logs/training_history.json          - per-epoch metrics
    logs/training_curves.png            - accuracy + loss plots
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "backend"))

from modules.brain.dataset_loader import get_dataloaders
from modules.brain.model.hybrid import HybridModel, save_model
from modules.brain.model.cnn import ClassicalCNNClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Quantum-Classical Brain Tumor Detector"
    )
    parser.add_argument("--data",       type=str, default=str(ROOT / "datasets" / "brain"),
                        help="Path to dataset root (contains yes/ and no/)")
    parser.add_argument("--epochs",     type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Adam learning rate")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Train CNN from scratch (not recommended)")
    parser.add_argument("--classical",   action="store_true",
                        help="Train the classical CNN baseline instead")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute accuracy, precision, recall, F1."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training and evaluation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns avg loss + metrics dict."""
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)   # (batch, 1)

        optimizer.zero_grad()
        outputs = model(imgs)                      # (batch, 1)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = (outputs.detach().cpu().squeeze() > 0.5).long().tolist()
        truths = labels.detach().cpu().squeeze().long().tolist()
        if isinstance(preds, int):
            preds  = [preds]
            truths = [truths]
        all_preds.extend(preds)
        all_labels.extend(truths)

    avg_loss = running_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run evaluation (val or test). Returns avg loss + metrics dict."""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)

        preds  = (outputs.cpu().squeeze() > 0.5).long().tolist()
        truths = labels.cpu().squeeze().long().tolist()
        if isinstance(preds, int):
            preds  = [preds]
            truths = [truths]
        all_preds.extend(preds)
        all_labels.extend(truths)

    avg_loss = running_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, save_path: str):
    """Plot and save accuracy + loss training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — Hybrid Quantum-Classical Model", fontsize=14)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved training curves → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Hybrid Quantum-Classical Brain Tumor Detector — Training")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Dataset      : {args.data}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mode         : {'Classical CNN' if args.classical else 'Hybrid Quantum-Classical'}")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.data, batch_size=args.batch_size
    )

    # ── Model ────────────────────────────────────────────────────────────
    if args.classical:
        model = ClassicalCNNClassifier(pretrained=not args.no_pretrain)
        model_name = "classical_cnn"
    else:
        model = HybridModel(n_quantum_layers=3, pretrained=not args.no_pretrain)
        model_name = "hybrid_quantum"

    model = model.to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # -- Optimizer & Loss (differential LRs per component) ----------------
    criterion = nn.BCELoss()
    if args.classical:
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = Adam([
            {"params": model.cnn.parameters(),         "lr": args.lr * 0.1},
            {"params": model.pre_quantum.parameters(), "lr": args.lr},
            {"params": model.quantum_layer.parameters(), "lr": args.lr * 10.0},
            {"params": model.classifier.parameters(),  "lr": args.lr},
        ])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # ── Output directories ──────────────────────────────────────────────
    ckpt_dir = ROOT / "checkpoints"
    log_dir  = ROOT / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    best_path = str(ckpt_dir / f"best_{model_name}.pth")
    last_path = str(ckpt_dir / f"last_{model_name}.pth")

    # ── Training loop ────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
    }
    best_val_acc = 0.0
    start_time   = time.time()

    current_lr = args.lr
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  [lr={current_lr:.2e}]")

        train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_m   = evaluate(model, val_loader, criterion, device)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["accuracy"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  [LR] Reduced: {prev_lr:.2e} → {new_lr:.2e}")
            current_lr = new_lr

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])

        print(f"  Train → loss={train_loss:.4f}  acc={train_m['accuracy']:.4f}  "
              f"f1={train_m['f1']:.4f}")
        print(f"  Val   → loss={val_loss:.4f}  acc={val_m['accuracy']:.4f}  "
              f"f1={val_m['f1']:.4f}  "
              f"prec={val_m['precision']:.4f}  rec={val_m['recall']:.4f}")

        # Save best checkpoint
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            save_model(model, best_path, {
                "epoch": epoch,
                "val_accuracy": best_val_acc,
                "val_f1": val_m["f1"],
            })
            print(f"  ★ New best val accuracy: {best_val_acc:.4f}")

    # Save final checkpoint
    save_model(model, last_path, {"epoch": args.epochs})

    # ── Test evaluation ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Final Test Evaluation (best checkpoint):")
    from model.hybrid import load_model as load_hybrid
    from model.cnn import ClassicalCNNClassifier as Classical

    if args.classical:
        best_model = Classical(pretrained=False).to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        best_model.load_state_dict(ckpt["model_state_dict"])
    else:
        best_model = load_hybrid(best_path, device=device, n_quantum_layers=3)
    best_model.eval()

    test_loss, test_m = evaluate(best_model, test_loader, criterion, device)
    print(f"  Test → loss={test_loss:.4f}  acc={test_m['accuracy']:.4f}  "
          f"prec={test_m['precision']:.4f}  rec={test_m['recall']:.4f}  "
          f"f1={test_m['f1']:.4f}")

    # ── Save history & plots ─────────────────────────────────────────────
    history_path = str(log_dir / f"training_history_{model_name}.json")
    with open(history_path, "w") as f:
        json.dump({**history, "test_metrics": test_m}, f, indent=2)
    print(f"[Log] History saved → {history_path}")

    plot_curves(history, str(log_dir / f"training_curves_{model_name}.png"))

    elapsed = time.time() - start_time
    print(f"\n[Done] Training complete in {elapsed/60:.1f} min")
    print(f"       Best val accuracy : {best_val_acc:.4f}")
    print(f"       Test accuracy     : {test_m['accuracy']:.4f}")
    print(f"       Test F1           : {test_m['f1']:.4f}\n")


if __name__ == "__main__":
    main()
