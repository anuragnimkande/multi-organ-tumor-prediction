# Skin-specific configuration
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.resolve()
CHECKPOINT_DIR = ROOT / "checkpoints"

SKIN_CONFIG = {
    "modality": "Dermoscopy",
    "classes": ["Benign", "Melanoma"],
    "img_size": (224, 224),
    "hybrid_checkpoint": str(CHECKPOINT_DIR / "best_hybrid_skin.pth"),
    "classical_checkpoint": str(CHECKPOINT_DIR / "best_classical_skin.pth"),
}
