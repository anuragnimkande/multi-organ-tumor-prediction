import torch
from backend.modules.organ_manager import get_hybrid_model, get_classical_model, run_inference, calibrated_prob, predict_organ_tumor

def predict_breast_tumor(img_tensor: torch.Tensor, img_bytes: bytes = None):
    return predict_organ_tumor("breast", img_tensor, img_bytes)
