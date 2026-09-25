"""
Multi-Organ Model Manager & Prediction Engine
=============================================
Provides model loading, caching, inference, and prediction
for supported organ modalities.

Supported organs with trained checkpoints
-----------------------------------------
  brain  : best_hybrid_quantum.pth  +  best_classical_cnn.pth   ✅ present
  lung   : best_hybrid_lung.pth     +  best_classical_lung.pth   ❌ not present
  breast : best_hybrid_breast.pth   +  best_classical_breast.pth ❌ not present
  liver  : best_hybrid_liver.pth    +  best_classical_liver.pth  ❌ not present
  skin   : best_hybrid_skin.pth     +  best_classical_skin.pth   ❌ not present

Non-brain organs will raise ModelNotAvailableError until their checkpoints
are placed in the checkpoints/ directory.
"""

import os
from pathlib import Path
import torch
import numpy as np
import cv2
import base64
import tensorflow as tf

from backend.organ_config import ORGAN_REGISTRY, get_organ_config
from backend.modules.brain.model.hybrid import HybridModel, load_model
from backend.modules.brain.model.cnn import ClassicalCNNClassifier

ROOT           = Path(__file__).parent.parent.parent.resolve()
BACKEND_DIR    = ROOT / "backend"
CHECKPOINT_DIR = ROOT / "checkpoints"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

_HYBRID_MODELS    = {}
_CLASSICAL_MODELS = {}
_KERAS_MODELS     = {}


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested organ has no trained checkpoint on disk."""


def _check_checkpoint(organ_id: str, ckpt_path: Path, model_kind: str = "hybrid") -> None:
    """Raise ModelNotAvailableError with a clear message if the checkpoint file is absent."""
    if not ckpt_path.exists():
        raise ModelNotAvailableError(
            f"No trained {model_kind} checkpoint found for organ '{organ_id}'. "
            f"Expected file: {ckpt_path}. "
            f"Train the model and place the checkpoint there to enable predictions."
        )


def get_hybrid_model(organ_id: str = "brain") -> HybridModel:
    """Load (or return cached) Hybrid Quantum-Classical model for the specified organ.

    Raises:
        ModelNotAvailableError: if the organ's checkpoint file does not exist.
    """
    global _HYBRID_MODELS
    organ_id = (organ_id or "brain").lower()

    if organ_id in _HYBRID_MODELS and _HYBRID_MODELS[organ_id] is not None:
        return _HYBRID_MODELS[organ_id]

    organ_cfg  = get_organ_config(organ_id) or ORGAN_REGISTRY["brain"]
    ckpt_name  = organ_cfg.get("checkpoint", "best_hybrid_quantum.pth")
    ckpt_path  = CHECKPOINT_DIR / ckpt_name

    # ── Strict: do NOT silently fall back to the brain checkpoint ──────────
    _check_checkpoint(organ_id, ckpt_path, model_kind="hybrid")

    model = load_model(str(ckpt_path), device=DEVICE, n_quantum_layers=3)
    print(f"[{organ_cfg['display_name']}] Hybrid Quantum Model loaded from {ckpt_path}")
    _HYBRID_MODELS[organ_id] = model
    return model


def get_classical_model(organ_id: str = "brain") -> ClassicalCNNClassifier:
    """Load (or return cached) Classical CNN model for the specified organ.

    Raises:
        ModelNotAvailableError: if the organ's checkpoint file does not exist.
    """
    global _CLASSICAL_MODELS
    organ_id = (organ_id or "brain").lower()

    if organ_id in _CLASSICAL_MODELS and _CLASSICAL_MODELS[organ_id] is not None:
        return _CLASSICAL_MODELS[organ_id]

    organ_cfg  = get_organ_config(organ_id) or ORGAN_REGISTRY["brain"]
    ckpt_name  = organ_cfg.get("classical_checkpoint", "best_classical_cnn.pth")
    ckpt_path  = CHECKPOINT_DIR / ckpt_name

    # ── Strict: do NOT silently fall back to the brain checkpoint ──────────
    _check_checkpoint(organ_id, ckpt_path, model_kind="classical")

    model = ClassicalCNNClassifier(pretrained=False).to(DEVICE)
    checkpoint = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[{organ_cfg['display_name']}] Classical CNN model loaded from {ckpt_path}")
    _CLASSICAL_MODELS[organ_id] = model
    return model


def get_keras_model(organ_id: str, ckpt_path: Path):
    global _KERAS_MODELS
    organ_id = organ_id.lower()
    
    if organ_id in _KERAS_MODELS and _KERAS_MODELS[organ_id] is not None:
        return _KERAS_MODELS[organ_id]
        
    _check_checkpoint(organ_id, ckpt_path, model_kind="keras")
    
    try:
        model = tf.keras.models.load_model(str(ckpt_path), compile=False)
        print(f"[Keras] Model loaded for {organ_id} from {ckpt_path}")
        _KERAS_MODELS[organ_id] = model
        return model
    except Exception as e:
        raise ModelNotAvailableError(f"Failed to load Keras model for {organ_id}: {e}")


def run_inference(model: torch.nn.Module, img_tensor: torch.Tensor) -> float:
    """Run model inference on a single image tensor and return a probability float."""
    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor.to(DEVICE))
        prob   = float(output.item())
    return prob


def compute_image_perturbation(img_bytes: bytes) -> float:
    """Calculate a small scan-quality boost/decay from image intensity and contrast."""
    if not img_bytes:
        return 0.0
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        h, w  = img.shape
        cy, cx = h // 2, w // 2
        roi   = img[
            max(0, cy - h // 4): min(h, cy + h // 4),
            max(0, cx - w // 4): min(w, cx + w // 4),
        ]

        roi_f32    = roi.astype(np.float32)
        brightness = float(np.mean(roi_f32)) / 255.0
        contrast   = float(np.std(roi_f32))  / 128.0
        lap_var    = float(cv2.Laplacian(roi_f32, cv2.CV_32F).var())
        sharpness  = float(np.tanh(lap_var / 800.0))

        raw = (brightness * 0.35 + contrast * 0.40 + sharpness * 0.25) - 0.50
        return float(np.clip(raw * 0.25, -0.12, 0.12))
    except Exception:
        return 0.0


def calibrated_prob(base_prob: float, img_bytes: bytes = None) -> float:
    """Apply scan-quality-based probability calibration."""
    perturb = compute_image_perturbation(img_bytes)
    return float(np.clip(base_prob + perturb, 0.05, 0.95))


def _resolve_checkpoint_path(organ_cfg: dict, organ_id: str) -> Path:
    """Resolve the checkpoint path from organ config, handling relative paths."""
    ckpt_name = organ_cfg.get("checkpoint", "")
    if ckpt_name.startswith(".."):
        # Relative to backend dir  e.g. ../models/liver/file.keras
        return (BACKEND_DIR / ckpt_name).resolve()
    return CHECKPOINT_DIR / ckpt_name




def predict_organ_tumor(organ_id: str, img_tensor: torch.Tensor,
                        img_bytes: bytes = None) -> dict:
    """Run full hybrid/keras prediction for the specified organ.

    Raises:
        ModelNotAvailableError: if the organ's checkpoint is absent.
    """
    organ_cfg  = get_organ_config(organ_id) or ORGAN_REGISTRY["brain"]
    framework  = organ_cfg.get("framework", "pytorch")

    if framework == "tensorflow":
        ckpt_path = _resolve_checkpoint_path(organ_cfg, organ_id)
        model = get_keras_model(organ_id, ckpt_path)

        # Decode image from raw bytes
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_shape = organ_cfg.get("input_shape", (224, 224))
        img = cv2.resize(img, input_shape)
        img = img.astype(np.float32)

        # ── Preprocessing must match training pipeline ────────────────
        if organ_id == "skin":
            # Actual saved model is a plain Sequential CNN (128×128)
            # Training used tf.cast -> float32 then sparse_categorical_crossentropy
            # Normalize pixel values to [0, 1]
            img = img / 255.0
        # For liver: MobileNetV2 model has built-in Rescaling(1./127.5, offset=-1)
        # so we pass raw 0-255 float32 values — the model normalises internally.

        img_batch = np.expand_dims(img, axis=0)
        preds = model.predict(img_batch, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        classes = organ_cfg["classes"]

        return {
            "probability":   confidence,
            "confidence":    confidence,
            "prediction":    classes[pred_idx],
            "model":         model,
            "organ_config":  organ_cfg,
            "framework":     "tensorflow"
        }
        
    # PyTorch Flow
    model      = get_hybrid_model(organ_id)                          # may raise
    base_prob  = run_inference(model, img_tensor)
    prob       = calibrated_prob(base_prob, img_bytes) if img_bytes else base_prob

    threshold  = 0.5
    classes    = organ_cfg["classes"]
    prediction = classes[1] if prob >= threshold else classes[0]
    confidence = prob if prob >= threshold else (1.0 - prob)

    return {
        "probability":   prob,
        "confidence":    confidence,
        "prediction":    prediction,
        "model":         model,
        "organ_config":  organ_cfg,
        "framework":     "pytorch"
    }
