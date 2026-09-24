"""
Hybrid Quantum-Classical Model
==============================
Combines a pretrained ResNet18 CNN with a 4-qubit PennyLane variational
quantum circuit for binary brain tumor classification.

Architecture (v2 — improved):
    Input (batch, 1, 224, 224)
        |
        v  CNNFeatureExtractor (ResNet18)
    (batch, 128)
        |
        v  MLP(128 -> 32 -> 8) + Tanh  [non-linear quantum input compression]
    (batch, 8)
        |
        v  QuantumLayer (TorchLayer, 4-qubit VQC, dense angle embedding)
    (batch, 4)
        |
        v  Linear(4 -> 1) + Sigmoid
    (batch, 1)  -- tumor probability in [0, 1]

Key improvements over v1:
    - pre_quantum is now a 2-layer MLP (128->32->8) instead of Linear(128->4)
    - Outputs 8 features to fill double-angle embedding (RY + RX per qubit)
    - load_model() accepts n_quantum_layers argument for flexibility
"""

import os
import sys
import torch
import torch.nn as nn

# Allow importing from sibling package when running from project root
# Support both direct execution and module-based imports
_MODEL_DIR = os.path.dirname(__file__)
_BRAIN_DIR = os.path.dirname(_MODEL_DIR)
sys.path.insert(0, _BRAIN_DIR)
sys.path.insert(0, _MODEL_DIR)

try:
    # When imported as part of the modules.brain package
    from backend.modules.brain.model.cnn import CNNFeatureExtractor
    from backend.modules.brain.model.quantum import build_quantum_layer, N_INPUT_FEATS
except ImportError:
    # Fallback: direct / relative import
    from model.cnn import CNNFeatureExtractor
    from model.quantum import build_quantum_layer, N_INPUT_FEATS


class HybridModel(nn.Module):
    """
    Hybrid Quantum-Classical model for brain tumor detection.

    Stages:
        1. ResNet18 CNN  -> 128-dim feature vector
        2. MLP(128->32->8) + Tanh  -> quantum-compatible 8-dim input
        3. 4-qubit VQC (PennyLane TorchLayer, dense embedding)  -> 4-dim
        4. Linear(4->1) + Sigmoid  -> probability
    """

    def __init__(self, n_quantum_layers: int = 3, pretrained: bool = True):
        """
        Args:
            n_quantum_layers: Number of VQC variational layers (3 recommended)
            pretrained:       Load ImageNet weights for CNN backbone
        """
        super().__init__()

        # -- Stage 1: CNN feature extraction ----------------------------------
        self.cnn = CNNFeatureExtractor(pretrained=pretrained, output_dim=128)

        # -- Stage 2: Non-linear classical compression for quantum input ------
        # MLP maps 128-dim features -> 8 features for double-angle embedding
        # Tanh maps to [-1, 1], ideal range for angle encoding
        self.pre_quantum = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, N_INPUT_FEATS),   # N_INPUT_FEATS = 8
            nn.Tanh(),
        )

        # -- Stage 3: Variational quantum circuit -----------------------------
        self.quantum_layer = build_quantum_layer(n_layers=n_quantum_layers)

        # -- Stage 4: Classical post-processing -------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the hybrid pipeline.

        Args:
            x: (batch, 1, 224, 224) -- preprocessed, normalised MRI tensor

        Returns:
            (batch, 1) -- tumour probability in [0, 1]
        """
        # Stage 1 -- CNN features
        features = self.cnn(x)              # (batch, 128)

        # Stage 2 -- Non-linear compress to quantum-compatible dimension
        q_input = self.pre_quantum(features)  # (batch, 8)

        # Stage 3 -- Quantum circuit (dense angle embedding + StronglyEntangling)
        q_output = self.quantum_layer(q_input)  # (batch, 4)

        # Stage 4 -- Final probability
        output = self.classifier(q_output)    # (batch, 1)
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 128-dim CNN features only (used for Grad-CAM and analysis).

        Args:
            x: (batch, 1, 224, 224)

        Returns:
            (batch, 128) feature vector
        """
        return self.cnn(x)

    def count_parameters(self) -> dict:
        """Return parameter counts per stage for transparency."""
        cnn_params    = sum(p.numel() for p in self.cnn.parameters())
        pre_q_params  = sum(p.numel() for p in self.pre_quantum.parameters())
        q_params      = sum(p.numel() for p in self.quantum_layer.parameters())
        cls_params    = sum(p.numel() for p in self.classifier.parameters())
        total         = cnn_params + pre_q_params + q_params + cls_params
        return {
            "cnn":         cnn_params,
            "pre_quantum": pre_q_params,
            "quantum":     q_params,
            "classifier":  cls_params,
            "total":       total,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model: HybridModel, path: str, extra: dict = None):
    """
    Save model state dict + optional metadata.

    Args:
        model: HybridModel instance
        path:  Output .pth file path
        extra: Optional dict with epoch, metrics, etc.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[Checkpoint] Saved -> {path}")


def load_model(checkpoint_path: str, device: str = "cpu",
               n_quantum_layers: int = 3) -> HybridModel:
    """
    Load a saved HybridModel from checkpoint file.

    Args:
        checkpoint_path:  Path to .pth checkpoint
        device:           'cpu' or 'cuda'
        n_quantum_layers: Number of quantum layers used during training

    Returns:
        HybridModel in eval mode, ready for inference
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")

    model = HybridModel(n_quantum_layers=n_quantum_layers, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[Checkpoint] Loaded <- {checkpoint_path} (device={device})")
    return model
