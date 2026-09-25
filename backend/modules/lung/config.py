# Lung-specific configuration
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CHECKPOINT_DIR = ROOT / "checkpoints"

LUNG_CONFIG = {
    "modality": "CT",
    "classes": ["Normal", "Lung Tumor"],
    "img_size": (224, 224),
    "hybrid_checkpoint": str(CHECKPOINT_DIR / "best_hybrid_lung.pth"),
    "classical_checkpoint": str(CHECKPOINT_DIR / "best_classical_lung.pth"),
}
