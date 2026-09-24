"""
Evaluation Script -- Hybrid Quantum-Classical Brain Tumor Detection
==================================================================
Runs comprehensive evaluation on the test split:
    - Confusion matrix (seaborn)
    - Accuracy / Loss curves (from training history)
    - Grad-CAM heatmap visualization
    - Classical CNN vs Hybrid comparison table
    - PDF report generation (fpdf2)

Usage:
    python evaluate.py [--data PATH] [--checkpoint PATH] [--classical_ckpt PATH]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc,
)
from tqdm import tqdm

# -- Path setup ---------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "backend"))

from modules.brain.dataset_loader import get_dataloaders
from modules.brain.model.hybrid import HybridModel, load_model
from modules.brain.model.cnn import ClassicalCNNClassifier


# ----------------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Detection Models")
    parser.add_argument("--data",            type=str,
                        default=str(ROOT / "datasets" / "brain"))
    parser.add_argument("--checkpoint",      type=str,
                        default=str(ROOT / "checkpoints" / "best_hybrid_quantum.pth"),
                        help="Path to hybrid model checkpoint")
    parser.add_argument("--classical_ckpt",  type=str,
                        default=str(ROOT / "checkpoints" / "best_classical_cnn.pth"),
                        help="Path to classical CNN checkpoint (optional)")
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--out",             type=str,
                        default=str(ROOT / "logs"),
                        help="Output directory for plots and PDF")
    parser.add_argument("--n_layers",        type=int, default=3,
                        help="Number of quantum layers in checkpoint (default: 3)")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model, loader, device):
    """Collect all predictions, probabilities and true labels from a loader."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in tqdm(loader, desc="Inferring"):
        imgs  = imgs.to(device)
        probs = model(imgs).cpu().squeeze()
        preds = (probs > 0.5).long()

        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
            preds = preds.unsqueeze(0)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for ResNet18 backbone inside HybridModel.
    Hooks into the last conv layer of the CNN backbone.
    """

    def __init__(self, model: HybridModel):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # Hook onto layer4 (last residual block) of ResNet18
        target_layer = list(model.cnn.backbone.children())[-3][-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor, device: str) -> np.ndarray:
        self.model.eval()
        img_tensor = img_tensor.to(device).requires_grad_(True)
        output     = self.model(img_tensor)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cv2.resize(cam, (224, 224))

    def overlay(self, img_tensor: torch.Tensor, cam: np.ndarray) -> np.ndarray:
        img_np  = img_tensor.squeeze().cpu().detach().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = np.stack([img_np] * 3, axis=-1)
        return np.clip(0.5 * img_rgb + 0.5 * heatmap, 0, 1)


# ----------------------------------------------------------------------------
# Plot functions
# ----------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path, title="Hybrid Model"):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Tumor", "Tumor"],
                yticklabels=["No Tumor", "Tumor"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix -> {save_path}")


def plot_roc_curve(y_true, y_probs, save_path, label="Hybrid"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax     = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#7c3aed", lw=2,
            label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC curve -> {save_path}")


def plot_grad_cam_samples(model, dataset, device, save_path, n_samples=6):
    """Visualize Grad-CAM heatmaps for random samples from the dataset."""
    cam_gen = GradCAM(model)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    fig.suptitle("Grad-CAM Heatmaps -- CNN Feature Attention", fontsize=13)

    for col, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_input = img_tensor.unsqueeze(0)
        cam       = cam_gen.generate(img_input, device)
        overlay   = cam_gen.overlay(img_input, cam)
        label_str = "Tumor" if label.item() == 1 else "No Tumor"

        axes[0, col].imshow(img_tensor.squeeze().numpy(), cmap="gray")
        axes[0, col].set_title(f"Original\n({label_str})", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title("Grad-CAM", fontsize=8)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Grad-CAM samples -> {save_path}")


def print_comparison_table(hybrid_m: dict, classical_m: dict):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("  Model Comparison: Classical CNN vs Hybrid Quantum-Classical")
    print("=" * 60)
    rows = [
        ("Accuracy",  classical_m["accuracy"],  hybrid_m["accuracy"]),
        ("Precision", classical_m["precision"], hybrid_m["precision"]),
        ("Recall",    classical_m["recall"],    hybrid_m["recall"]),
        ("F1 Score",  classical_m["f1"],        hybrid_m["f1"]),
    ]
    print(f"  {'Metric':<12} {'Classical CNN':>16} {'Hybrid Quantum':>16}")
    print("  " + "-" * 46)
    for name, cls_val, hyb_val in rows:
        better = "*" if hyb_val >= cls_val else " "
        print(f"  {name:<12} {cls_val:>15.4f} {hyb_val:>15.4f} {better}")
    print("=" * 60 + "\n")


# ----------------------------------------------------------------------------
# PDF Report Generation
# ----------------------------------------------------------------------------

def generate_pdf_report(
    hybrid_metrics:    dict,
    classical_metrics: dict | None,
    confusion_img:     str,
    roc_img:           str,
    gradcam_img:       str,
    training_history:  dict | None,
    save_path:         str,
):
    """
    Generate a professional PDF evaluation report using fpdf2.

    Args:
        hybrid_metrics:    dict with accuracy/precision/recall/f1 for hybrid model
        classical_metrics: same for classical model (or None if not available)
        confusion_img:     path to confusion matrix PNG
        roc_img:           path to ROC curve PNG
        gradcam_img:       path to Grad-CAM PNG
        training_history:  dict with train/val metrics lists (or None)
        save_path:         output .pdf path
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("[PDF] fpdf2 not installed. Run: pip install fpdf2")
        return

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)       # dark blue-gray bg
            self.rect(0, 0, 210, 28, "F")
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(0, 212, 255)       # cyan
            self.set_y(7)
            self.cell(0, 8, "Brain Tumor Detection -- Evaluation Report", align="C")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(180, 180, 200)
            self.set_y(17)
            self.cell(0, 5, "Hybrid Quantum-Classical Model (ResNet18 + 4-qubit VQC)", align="C")
            self.ln(16)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(130, 130, 160)
            self.cell(0, 5, f"Page {self.page_no()} | Quantum Brain Tumor Detection System", align="C")

        def section_title(self, title: str):
            self.set_fill_color(30, 41, 59)
            self.set_text_color(0, 212, 255)
            self.set_font("Helvetica", "B", 12)
            self.ln(4)
            self.set_x(10)
            self.cell(190, 9, f"  {title}", fill=True, ln=True)
            self.ln(3)
            self.set_text_color(30, 30, 50)

        def metric_row(self, name, value, highlight=False):
            self.set_font("Helvetica", "", 10)
            self.set_x(18)
            if highlight:
                self.set_fill_color(220, 252, 231)
                self.set_text_color(20, 120, 50)
            else:
                self.set_fill_color(248, 250, 252)
                self.set_text_color(50, 50, 80)
            self.cell(70, 7, name, fill=True, border="LTB")
            self.cell(50, 7, f"{value:.4f}" if isinstance(value, float) else str(value),
                      fill=True, border="RTB", align="C")
            self.ln(1)

        def two_col_metric_row(self, name, cls_val, hyb_val):
            self.set_font("Helvetica", "", 10)
            self.set_x(18)
            better = hyb_val >= cls_val
            self.set_fill_color(248, 250, 252)
            self.set_text_color(50, 50, 80)
            self.cell(60, 7, name, fill=True, border="LTB")
            # Classical
            self.set_fill_color(248, 250, 252)
            self.cell(55, 7, f"{cls_val:.4f}", fill=True, border="TB", align="C")
            # Hybrid
            if better:
                self.set_fill_color(220, 252, 231)
                self.set_text_color(20, 120, 50)
            else:
                self.set_fill_color(254, 226, 226)
                self.set_text_color(180, 30, 30)
            self.cell(55, 7, f"{hyb_val:.4f}" + (" *" if better else ""),
                      fill=True, border="RTB", align="C")
            self.set_text_color(50, 50, 80)
            self.ln(1)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ------------------------------------------------------------------ Summary
    pdf.section_title("1.  Executive Summary")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 60, 80)
    pdf.set_x(15)
    pdf.multi_cell(
        180, 6,
        "This report presents the evaluation results of the Hybrid Quantum-Classical "
        "brain tumor detection model. The architecture combines a pretrained ResNet-18 "
        "CNN feature extractor with a 4-qubit variational quantum circuit (VQC) using "
        "dense angle embedding and StronglyEntanglingLayers. All metrics are reported "
        "on the held-out test split.",
    )
    pdf.ln(3)

    # -------------------------------------------------------- Hybrid Model Metrics
    pdf.section_title("2.  Hybrid Quantum-Classical Model Metrics")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(18)
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(70, 8, "  Metric", fill=True, border=1)
    pdf.cell(50, 8, "Value", fill=True, border=1, align="C")
    pdf.ln(1)

    metrics_list = [
        ("Accuracy",  hybrid_metrics["accuracy"],  hybrid_metrics["accuracy"] >= 0.95),
        ("Precision", hybrid_metrics["precision"], hybrid_metrics["precision"] >= 0.95),
        ("Recall",    hybrid_metrics["recall"],    hybrid_metrics["recall"] >= 0.95),
        ("F1 Score",  hybrid_metrics["f1"],        hybrid_metrics["f1"] >= 0.95),
    ]
    for name, val, hi in metrics_list:
        pdf.metric_row(name, val, highlight=hi)
    pdf.ln(4)

    # ------------------------------------------------- Classical vs Hybrid Table
    if classical_metrics:
        pdf.section_title("3.  Classical CNN vs Hybrid Quantum-Classical Comparison")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(18)
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(60, 8, "  Metric", fill=True, border=1)
        pdf.cell(55, 8, "Classical CNN", fill=True, border=1, align="C")
        pdf.cell(55, 8, "Hybrid Quantum (*=better)", fill=True, border=1, align="C")
        pdf.ln(1)

        compare_rows = [
            ("Accuracy",  classical_metrics["accuracy"],  hybrid_metrics["accuracy"]),
            ("Precision", classical_metrics["precision"], hybrid_metrics["precision"]),
            ("Recall",    classical_metrics["recall"],    hybrid_metrics["recall"]),
            ("F1 Score",  classical_metrics["f1"],        hybrid_metrics["f1"]),
        ]
        for name, cv, hv in compare_rows:
            pdf.two_col_metric_row(name, cv, hv)
        pdf.ln(4)

    # ------------------------------------------------------------- Confusion Matrix
    if os.path.isfile(confusion_img):
        sec_num = 4 if classical_metrics else 3
        pdf.section_title(f"{sec_num}.  Confusion Matrix")
        pdf.image(confusion_img, x=30, w=120)
        pdf.ln(3)

    # ----------------------------------------------------------------- ROC Curve
    if os.path.isfile(roc_img):
        sec_num = 5 if classical_metrics else 4
        # Check if we need a new page
        if pdf.get_y() > 180:
            pdf.add_page()
        pdf.section_title(f"{sec_num}.  ROC Curve")
        pdf.image(roc_img, x=30, w=120)
        pdf.ln(3)

    # --------------------------------------------------------------- Grad-CAM
    if os.path.isfile(gradcam_img):
        sec_num = 6 if classical_metrics else 5
        pdf.add_page()
        pdf.section_title(f"{sec_num}.  Grad-CAM Feature Attention Maps")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(15)
        pdf.set_text_color(80, 80, 100)
        pdf.multi_cell(180, 5,
            "Grad-CAM highlights the regions of the MRI scan that most influenced "
            "the model's prediction. Hot colors (red/yellow) indicate high activation.")
        pdf.ln(2)
        pdf.image(gradcam_img, x=10, w=190)
        pdf.ln(3)

    # --------------------------------------------------------- Training History
    if training_history:
        sec_num = 7 if classical_metrics else 6
        pdf.add_page()
        pdf.section_title(f"{sec_num}.  Training History")
        train_acc = training_history.get("train_acc", [])
        val_acc   = training_history.get("val_acc",   [])
        train_loss = training_history.get("train_loss", [])
        val_loss   = training_history.get("val_loss",   [])

        if train_acc:
            # Draw a mini training summary table
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(18)
            pdf.set_fill_color(15, 23, 42)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(25, 8, "Epoch", fill=True, border=1, align="C")
            pdf.cell(40, 8, "Train Loss", fill=True, border=1, align="C")
            pdf.cell(40, 8, "Val Loss",   fill=True, border=1, align="C")
            pdf.cell(35, 8, "Train Acc",  fill=True, border=1, align="C")
            pdf.cell(35, 8, "Val Acc",    fill=True, border=1, align="C")
            pdf.ln(8)

            # Show every epoch (up to 25 rows to avoid overflow)
            n_epochs = min(len(train_acc), 25)
            for ep in range(n_epochs):
                pdf.set_font("Helvetica", "", 9)
                pdf.set_x(18)
                if ep % 2 == 0:
                    pdf.set_fill_color(240, 245, 255)
                else:
                    pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(40, 40, 70)
                pdf.cell(25, 6, str(ep + 1),             fill=True, border="LRB", align="C")
                pdf.cell(40, 6, f"{train_loss[ep]:.4f}" if ep < len(train_loss) else "-",
                         fill=True, border="RB", align="C")
                pdf.cell(40, 6, f"{val_loss[ep]:.4f}"   if ep < len(val_loss)   else "-",
                         fill=True, border="RB", align="C")
                pdf.cell(35, 6, f"{train_acc[ep]:.4f}"  if ep < len(train_acc)  else "-",
                         fill=True, border="RB", align="C")
                pdf.cell(35, 6, f"{val_acc[ep]:.4f}"    if ep < len(val_acc)    else "-",
                         fill=True, border="RB", align="C")
                pdf.ln(0)
            pdf.ln(4)

    # ---------------------------------------------------------------- Architecture
    last_sec = (8 if classical_metrics else 7)
    pdf.add_page()
    pdf.section_title(f"{last_sec}.  Model Architecture")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 60, 80)
    arch_lines = [
        "Input MRI: (1, 224, 224) normalized grayscale tensor",
        "",
        "Stage 1 -- CNN Feature Extractor (ResNet-18 pretrained)",
        "         Conv2d(1, 64) [grayscale adapted] + BatchNorm + MaxPool",
        "         Layer1..Layer4 (BasicBlocks)",
        "         AdaptiveAvgPool -> Flatten -> Linear(512, 128) + ReLU + Dropout(0.3)",
        "         Output: (batch, 128)",
        "",
        "Stage 2 -- Non-Linear Quantum Input Compression (MLP)",
        "         Linear(128, 32) + ReLU + Dropout(0.2)",
        "         Linear(32, 8) + Tanh",
        "         Output: (batch, 8)  [features in [-1, 1]]",
        "",
        "Stage 3 -- Variational Quantum Circuit (PennyLane TorchLayer)",
        "         Dense AngleEmbedding: RY(features[0:4]) + RX(features[4:8])",
        "         StronglyEntanglingLayers x 3  [full SU(2) + multi-range CNOT]",
        "         Measurement: PauliZ expectation on all 4 qubits",
        "         Output: (batch, 4)  [values in [-1, 1]]",
        "",
        "Stage 4 -- Classifier",
        "         Linear(4, 1) + Sigmoid",
        "         Output: (batch, 1)  [tumor probability in [0, 1]]",
        "",
        "Optimizer: Adam with differential learning rates",
        "  CNN backbone    : lr = 1e-5  (fine-tuning, protects pretrained weights)",
        "  Classical MLP   : lr = 1e-4  (standard)",
        "  Quantum VQC     : lr = 1e-3  (10x, overcomes barren-plateau gradients)",
        "  Classifier head : lr = 1e-4  (standard)",
    ]
    for line in arch_lines:
        pdf.set_x(15)
        if line.startswith("Stage") or line.startswith("Input") or line.startswith("Optimizer"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 80, 160)
        elif line == "":
            pdf.ln(2)
            continue
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 70, 90)
        pdf.cell(180, 6, line)
        pdf.ln(0)

    pdf.output(save_path)
    print(f"[PDF] Report saved -> {save_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print(f"\n[Evaluate] Device: {device}")

    # -- Load data ------------------------------------------------------------
    _, _, test_loader, dataset = get_dataloaders(args.data, batch_size=args.batch_size)

    # -- Load hybrid model ----------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("        Run `python train_fast.py` first to generate a checkpoint.")
        sys.exit(1)

    hybrid_model = load_model(args.checkpoint, device=device,
                              n_quantum_layers=args.n_layers)
    y_true, y_pred, y_prob = get_predictions(hybrid_model, test_loader, device)

    hybrid_metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }

    print("\n[Hybrid Model] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))

    confusion_img = str(out_dir / "confusion_matrix_hybrid.png")
    roc_img       = str(out_dir / "roc_curve_hybrid.png")
    gradcam_img   = str(out_dir / "gradcam_samples.png")

    plot_confusion_matrix(y_true, y_pred, confusion_img, "Hybrid Model")
    plot_roc_curve(y_true, y_prob, roc_img, "Hybrid Quantum-Classical")

    # -- Grad-CAM -------------------------------------------------------------
    plot_grad_cam_samples(hybrid_model, dataset, device, gradcam_img, n_samples=6)

    # -- Classical baseline (if checkpoint exists) ----------------------------
    classical_metrics = None
    if os.path.isfile(args.classical_ckpt):
        cls_model = ClassicalCNNClassifier(pretrained=False).to(device)
        ckpt = torch.load(args.classical_ckpt, map_location=device, weights_only=False)
        cls_model.load_state_dict(ckpt["model_state_dict"])
        cls_model.eval()

        cy_true, cy_pred, cy_prob = get_predictions(cls_model, test_loader, device)
        classical_metrics = {
            "accuracy":  accuracy_score(cy_true, cy_pred),
            "precision": precision_score(cy_true, cy_pred, zero_division=0),
            "recall":    recall_score(cy_true, cy_pred, zero_division=0),
            "f1":        f1_score(cy_true, cy_pred, zero_division=0),
        }

        plot_confusion_matrix(cy_true, cy_pred,
                              str(out_dir / "confusion_matrix_classical.png"), "Classical CNN")
        plot_roc_curve(cy_true, cy_prob,
                       str(out_dir / "roc_curve_classical.png"), "Classical CNN")
        print_comparison_table(hybrid_metrics, classical_metrics)
    else:
        print("[Info] Classical checkpoint not found -- skipping comparison.")

    # -- Load training history (if available) --------------------------------
    history_path = out_dir / "training_history_hybrid_quantum.json"
    training_history = None
    if history_path.exists():
        with open(history_path) as f:
            training_history = json.load(f)

    # -- Save JSON metrics (legacy) -------------------------------------------
    report = {"hybrid": hybrid_metrics, "classical": classical_metrics}
    json_path = str(out_dir / "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[JSON] Report saved -> {json_path}")

    # -- Generate PDF Report --------------------------------------------------
    pdf_path = str(out_dir / "evaluation_report.pdf")
    generate_pdf_report(
        hybrid_metrics    = hybrid_metrics,
        classical_metrics = classical_metrics,
        confusion_img     = confusion_img,
        roc_img           = roc_img,
        gradcam_img       = gradcam_img,
        training_history  = training_history,
        save_path         = pdf_path,
    )

    print(f"\n[Done] Evaluation complete.")
    print(f"       PDF report  -> {pdf_path}")
    print(f"       JSON report -> {json_path}")


if __name__ == "__main__":
    main()
