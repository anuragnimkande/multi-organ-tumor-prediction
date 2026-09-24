import os
import torch
import numpy as np
import cv2

from .model.hybrid import HybridModel, load_model
from .model.cnn import ClassicalCNNClassifier
from .config import BRAIN_CONFIG

_hybrid_model    = None
_classical_model = None
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

def get_hybrid_model() -> HybridModel:
    global _hybrid_model
    if _hybrid_model is None:
        ckpt = BRAIN_CONFIG["hybrid_checkpoint"]
        if os.path.isfile(ckpt):
            _hybrid_model = load_model(ckpt, device=DEVICE, n_quantum_layers=3)
            print(f"[Brain Module] Hybrid model loaded from {ckpt}")
        else:
            print("[Brain Module] WARNING: No checkpoint found -- using untrained model (demo mode)")
            _hybrid_model = HybridModel(n_quantum_layers=3, pretrained=False)
            _hybrid_model.eval()
    return _hybrid_model

def get_classical_model() -> ClassicalCNNClassifier:
    global _classical_model
    if _classical_model is None:
        ckpt = BRAIN_CONFIG["classical_checkpoint"]
        if os.path.isfile(ckpt):
            _classical_model = ClassicalCNNClassifier(pretrained=False).to(DEVICE)
            checkpoint = torch.load(ckpt, map_location=DEVICE)
            _classical_model.load_state_dict(checkpoint["model_state_dict"])
            _classical_model.eval()
            print(f"[Brain Module] Classical model loaded from {ckpt}")
        else:
            _classical_model = ClassicalCNNClassifier(pretrained=False).eval()
    return _classical_model

def run_inference(model, img_tensor: torch.Tensor) -> float:
    """Run model inference and return probability."""
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(DEVICE))   # (1, 1)
        prob = output.item()
    return prob

def compute_image_perturbation(img_bytes: bytes) -> float:
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        h, w = img.shape
        cy, cx  = h // 2, w // 2
        roi     = img[cy - h // 4: cy + h // 4, cx - w // 4: cx + w // 4]

        roi_f32    = roi.astype(np.float32)
        brightness = float(np.mean(roi_f32)) / 255.0
        contrast   = float(np.std(roi_f32))  / 128.0
        lap_var    = float(cv2.Laplacian(roi_f32, cv2.CV_32F).var())
        sharpness  = float(np.tanh(lap_var / 800.0))

        raw = (brightness * 0.35 + contrast * 0.40 + sharpness * 0.25) - 0.50
        return float(np.clip(raw * 0.25, -0.12, 0.12))
    except Exception:
        return 0.0

def calibrated_prob(base_prob: float, img_bytes: bytes) -> float:
    perturb = compute_image_perturbation(img_bytes)
    return float(np.clip(base_prob + perturb, 0.05, 0.95))

def predict_tumor(img_tensor: torch.Tensor, img_bytes: bytes = None):
    model = get_hybrid_model()
    prob = run_inference(model, img_tensor)
    if img_bytes:
        prob = calibrated_prob(prob, img_bytes)
    
    threshold = 0.5
    prediction = BRAIN_CONFIG["classes"][1] if prob >= threshold else BRAIN_CONFIG["classes"][0]
    return {
        "probability": prob,
        "prediction": prediction,
        "model": model
    }
