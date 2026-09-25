# Breast-specific configuration
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CHECKPOINT_DIR = ROOT / "checkpoints"

BREAST_CONFIG = {
    "modality": "Mammography",
    "classes": ["Benign", "Malignant Breast Tumor"],
    "img_size": (224, 224),
    "hybrid_checkpoint": str(CHECKPOINT_DIR / "best_hybrid_breast.pth"),
    "classical_checkpoint": str(CHECKPOINT_DIR / "best_classical_breast.pth"),
}
