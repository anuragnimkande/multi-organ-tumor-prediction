# Brain-specific configuration
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CHECKPOINT_DIR = ROOT / "checkpoints"

BRAIN_CONFIG = {
    "modality": "MRI",
    "classes": ["No Tumor", "Tumor"],
    "img_size": (224, 224),
    "hybrid_checkpoint": str(CHECKPOINT_DIR / "best_hybrid_quantum.pth"),
    "classical_checkpoint": str(CHECKPOINT_DIR / "best_classical_cnn.pth"),
}
