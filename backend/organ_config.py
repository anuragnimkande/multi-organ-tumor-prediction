# Centralized configuration for all supported organs

ORGAN_REGISTRY = {
    "brain": {
        "id": "brain",
        "modality": "MRI",
        "display_name": "Brain Tumor",
        "available_tasks": ["tumor_detection", "model_comparison", "grad_cam"],
        "model_status": "trained",
        "classes": ["No Tumor", "Tumor"],
        "module": "backend.modules.brain"
    },
    "lung": {
        "id": "lung",
        "modality": "CT",
        "display_name": "Lung Tumor",
        "available_tasks": ["tumor_detection"],
        "model_status": "unavailable",
        "classes": ["Normal", "Tumor"],
        "module": "backend.modules.lung"
    },
    "breast": {
        "id": "breast",
        "modality": "Mammography",
        "display_name": "Breast Tumor",
        "available_tasks": ["tumor_detection"],
        "model_status": "unavailable",
        "classes": ["Benign", "Malignant"],
        "module": "backend.modules.breast"
    },
    "liver": {
        "id": "liver",
        "modality": "CT",
        "display_name": "Liver Tumor",
        "available_tasks": ["tumor_detection"],
        "model_status": "unavailable",
        "classes": ["Healthy", "Lesion"],
        "module": "backend.modules.liver"
    },
    "skin": {
        "id": "skin",
        "modality": "Dermoscopy",
        "display_name": "Skin Lesion",
        "available_tasks": ["tumor_detection"],
        "model_status": "unavailable",
        "classes": ["Benign", "Melanoma"],
        "module": "backend.modules.skin"
    }
}

def get_organ_config(organ_id: str) -> dict:
    return ORGAN_REGISTRY.get(organ_id, None)
