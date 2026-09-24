"""
Fast Training via Pre-Cached Tensors
=====================================
Step 1: Preprocess all images once -> save as .pt tensor files in cache/
Step 2: Train from cached tensors (no OpenCV overhead per batch)

This reduces training time from ~5s/batch -> <0.5s/batch on CPU.

Differential Learning Rates:
    CNN backbone  : lr * 0.1   (protect pretrained weights)
    Classical MLP : lr         (standard)
    Quantum layer : lr * 10    (overcome barren-plateau vanishing gradients)
    Classifier    : lr         (standard)

Usage:
    python train_fast.py [--epochs N] [--batch_size N] [--classical]
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "backend"))

from modules.brain.preprocessing import load_and_preprocess
from modules.brain.model.hybrid import HybridModel, save_model, load_model
from modules.brain.model.cnn import ClassicalCNNClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fast training via pre-cached tensors")
    parser.add_argument("--data",       type=str,   default=str(ROOT / "datasets" / "brain"))
    parser.add_argument("--cache",      type=str,   default=str(ROOT / "cache"))
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--classical",  action="store_true")
    parser.add_argument("--no_pretrain",action="store_true")
    parser.add_argument("--skip_cache", action="store_true", help="Skip cache rebuild if cache exists")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Cache builder
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def build_cache(data_dir: str, cache_dir: str):
    """
    Pre-process all MRI images and save as .pt tensor files.
    Only rebuilds if cache is missing or stale.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    manifest_file = cache_path / "manifest.json"
    yes_dir = Path(data_dir) / "yes"
    no_dir  = Path(data_dir) / "no"

    all_samples = []
    for fname in sorted(os.listdir(yes_dir)):
        if Path(fname).suffix.lower() in VALID_EXT:
            all_samples.append((str(yes_dir / fname), 1))
    for fname in sorted(os.listdir(no_dir)):
        if Path(fname).suffix.lower() in VALID_EXT:
            all_samples.append((str(no_dir / fname), 0))

    print(f"\n[Cache] Building cache for {len(all_samples)} images → {cache_path}")
    print(f"[Cache] This runs once and makes training ~10x faster.\n")

    cached = []
    for i, (img_path, label) in enumerate(tqdm(all_samples, desc="Preprocessing")):
        tensor_file = cache_path / f"sample_{i:05d}.pt"
        if not tensor_file.exists():
            try:
                tensor = load_and_preprocess(img_path)
                torch.save({"tensor": tensor, "label": label, "path": img_path},
                           str(tensor_file))
            except Exception as e:
                print(f"\n[Cache] WARN: Skipping {img_path}: {e}")
                continue
        cached.append(str(tensor_file))

    # Save manifest
    with open(manifest_file, "w") as f:
        json.dump(cached, f)

    print(f"[Cache] Done. {len(cached)} tensors cached.")
    return cached


# ──────────────────────────────────────────────────────────────────────────────
# Cached Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CachedMRIDataset(Dataset):
    """Loads pre-processed tensors from cache — no OpenCV overhead at train time."""

    def __init__(self, tensor_files: list, augment: bool = False):
        self.files   = tensor_files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item  = torch.load(self.files[idx], weights_only=False)
        tensor = item["tensor"]   # (1, 224, 224)
        label  = item["label"]    # int

        if self.augment:
            tensor = self._augment(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)

    def _augment(self, t: torch.Tensor) -> torch.Tensor:
        img = t.squeeze(0).numpy()

        if random.random() < 0.5:
            img = np.fliplr(img).copy()
        if random.random() < 0.3:
            img = np.flipud(img).copy()
        if random.random() < 0.5:
            import cv2
            angle = random.uniform(-15, 15)
            h, w  = img.shape
            M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.5:
            img = np.clip(img * (1.0 + random.uniform(-0.10, 0.10)), 0, 1)
        if random.random() < 0.3:
            img = np.clip(img + np.random.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)

        return torch.from_numpy(img.astype(np.float32)).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds  = (out.detach().cpu().squeeze() > 0.5).long().tolist()
        truths = labels.detach().cpu().squeeze().long().tolist()
        if isinstance(preds, int):
            preds, truths = [preds], [truths]
        all_preds.extend(preds)
        all_labels.extend(truths)

    return running_loss / len(loader.dataset), compute_metrics(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False, ncols=80):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        out  = model(imgs)
        loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)

        probs  = out.cpu().squeeze()
        preds  = (probs > 0.5).long().tolist()
        truths = labels.cpu().squeeze().long().tolist()
        if isinstance(preds, int):
            preds, truths = [preds], [truths]
            probs = [probs.item()]
        all_preds.extend(preds)
        all_labels.extend(truths)
        all_probs.extend(probs if isinstance(probs, list) else probs.tolist())

    return running_loss / len(loader.dataset), compute_metrics(all_labels, all_preds)


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(history, save_path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=5)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=5)
    axes[0].set(xlabel="Epoch", ylabel="BCE Loss", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=5)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=5)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode   = "Classical CNN" if args.classical else "Hybrid Quantum-Classical"

    print(f"\n{'='*60}")
    print(f"  QuantumBrain — Fast Training ({mode})")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"{'='*60}\n")

    # ── Step 1: Build / load cache ────────────────────────────────────────
    manifest = Path(args.cache) / "manifest.json"
    if manifest.exists() and args.skip_cache:
        print("[Cache] Loading existing cache...")
        with open(manifest) as f:
            all_files = json.load(f)
        print(f"[Cache] {len(all_files)} tensors found.")
    else:
        all_files = build_cache(args.data, args.cache)

    if len(all_files) == 0:
        print("[Error] No cached tensors found!"); return

    # ── Step 2: Split ────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(all_files)
    total      = len(all_files)
    train_size = int(total * 0.70)
    val_size   = int(total * 0.15)

    train_files = all_files[:train_size]
    val_files   = all_files[train_size:train_size + val_size]
    test_files  = all_files[train_size + val_size:]

    print(f"[Splits] train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")

    train_ds = CachedMRIDataset(train_files, augment=True)
    val_ds   = CachedMRIDataset(val_files,   augment=False)
    test_ds  = CachedMRIDataset(test_files,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Step 3: Model ─────────────────────────────────────────────────────
    if args.classical:
        model = ClassicalCNNClassifier(pretrained=not args.no_pretrain)
        model_name = "classical_cnn"
    else:
        model = HybridModel(n_quantum_layers=3, pretrained=not args.no_pretrain)
        model_name = "hybrid_quantum"

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_name} | {total_params:,} parameters\n")

    # ── Step 4: Training ──────────────────────────────────────────────────
    criterion = nn.BCELoss()
    if args.classical:
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = Adam([
            {"params": model.cnn.parameters(),           "lr": args.lr * 0.1},
            {"params": model.pre_quantum.parameters(),   "lr": args.lr},
            {"params": model.quantum_layer.parameters(), "lr": args.lr * 10.0},
            {"params": model.classifier.parameters(),    "lr": args.lr},
        ])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    ckpt_dir = ROOT / "checkpoints"
    log_dir  = ROOT / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    best_path = str(ckpt_dir / f"best_{model_name}.pth")
    last_path = str(ckpt_dir / f"last_{model_name}.pth")

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                    "train_f1": [], "val_f1": []}
    best_val_acc = 0.0
    current_lr   = args.lr
    start_time   = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs}  [lr={current_lr:.2e}]")

        train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_m   = evaluate(model, val_loader, criterion, device)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["accuracy"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  [LR] {prev_lr:.2e} → {new_lr:.2e}")
            current_lr = new_lr

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])

        epoch_time = time.time() - epoch_start
        print(f"  Train → loss={train_loss:.4f}  acc={train_m['accuracy']:.4f}  f1={train_m['f1']:.4f}")
        print(f"  Val   → loss={val_loss:.4f}  acc={val_m['accuracy']:.4f}  "
              f"f1={val_m['f1']:.4f}  prec={val_m['precision']:.4f}  rec={val_m['recall']:.4f}  "
              f"[{epoch_time:.0f}s]")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            save_model(model, best_path, {
                "epoch": epoch,
                "val_accuracy": best_val_acc,
                "val_f1": val_m["f1"],
            })
            print(f"  ★ Best checkpoint saved  (val_acc={best_val_acc:.4f})")

    save_model(model, last_path, {"epoch": args.epochs})

    # ── Step 5: Test evaluation ───────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Final Test Evaluation (best checkpoint):")

    if args.classical:
        best_model = ClassicalCNNClassifier(pretrained=False).to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        best_model.load_state_dict(ckpt["model_state_dict"])
    else:
        best_model = load_model(best_path, device=device, n_quantum_layers=3)
    best_model.eval()

    test_loss, test_m = evaluate(best_model, test_loader, criterion, device)
    print(f"  Test → loss={test_loss:.4f}  acc={test_m['accuracy']:.4f}  "
          f"prec={test_m['precision']:.4f}  rec={test_m['recall']:.4f}  f1={test_m['f1']:.4f}")

    # ── Save logs ─────────────────────────────────────────────────────────
    history_path = str(log_dir / f"training_history_{model_name}.json")
    with open(history_path, "w") as f:
        json.dump({**history, "test_metrics": test_m}, f, indent=2)
    print(f"\n[Log] {history_path}")

    plot_curves(history, str(log_dir / f"training_curves_{model_name}.png"),
                f"Training Curves — {mode}")

    elapsed = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Test accuracy     : {test_m['accuracy']:.4f}")
    print(f"  Test F1           : {test_m['f1']:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
