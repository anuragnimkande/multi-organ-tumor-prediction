# Liver-specific configuration
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CHECKPOINT_DIR = ROOT / "checkpoints"

LIVER_CONFIG = {
    "modality": "CT",
    "classes": ["Healthy", "Liver Tumor"],
    "img_size": (224, 224),
    "hybrid_checkpoint": str(CHECKPOINT_DIR / "best_hybrid_liver.pth"),
    "classical_checkpoint": str(CHECKPOINT_DIR / "best_classical_liver.pth"),
}
