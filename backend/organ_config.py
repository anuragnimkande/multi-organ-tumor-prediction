# Centralized configuration for all supported organs

ORGAN_REGISTRY = {
    "brain": {
        "id": "brain",
        "modality": "MRI",
        "display_name": "Brain Tumor",
        "available_tasks": ["tumor_detection", "model_comparison", "grad_cam"],
        "model_status": "trained",
        "classes": ["No Tumor", "Tumor"],
        "anomaly_label": "Brain Tumor",
        "normal_label": "No Brain Tumor",
        "module": "backend.modules.brain",
        "checkpoint": "best_hybrid_quantum.pth",
        "classical_checkpoint": "best_classical_cnn.pth",
    },
    "lung": {
        "id": "lung",
        "modality": "CT",
        "display_name": "Lung Tumor",
        "available_tasks": ["tumor_detection", "model_comparison", "grad_cam"],
        "model_status": "trained",
        "classes": ["Normal", "Lung Tumor"],
        "anomaly_label": "Lung Nodule/Tumor",
        "normal_label": "Normal Lung",
        "module": "backend.modules.lung",
        "checkpoint": "best_hybrid_lung.pth",
        "classical_checkpoint": "best_classical_lung.pth",
    },
    "breast": {
        "id": "breast",
        "modality": "Mammography",
        "display_name": "Breast Tumor",
        "available_tasks": ["tumor_detection", "model_comparison", "grad_cam"],
        "model_status": "trained",
        "classes": ["Benign", "Malignant Breast Tumor"],
        "anomaly_label": "Malignant Breast Lesion",
        "normal_label": "Benign Tissue",
        "module": "backend.modules.breast",
        "checkpoint": "best_hybrid_breast.pth",
        "classical_checkpoint": "best_classical_breast.pth",
    },
    "liver": {
        "id": "liver",
        "modality": "CT",
        "display_name": "Liver Tumor",
        "available_tasks": ["tumor_detection", "grad_cam"],
        "model_status": "trained",
        "classes": [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular_Carcinoma"
        ],
        "anomaly_label": "Liver Lesion/Tumor",
        "normal_label": "Healthy Liver",
        "module": "backend.modules.liver",
        "checkpoint": "../models/liver/mobilenetv2_liver_final_patched.keras",
        "framework": "tensorflow",
        "input_shape": (224, 224),
    },
    "skin": {
        "id": "skin",
        "modality": "Dermoscopy",
        "display_name": "Skin Lesion",
        "available_tasks": ["tumor_detection", "grad_cam"],
        "model_status": "trained",
        "classes": [
            "Eczema",
            "Melanoma",
            "Atopic Dermatitis",
            "Basal Cell Carcinoma",
            "Melanocytic Nevi",
            "Benign Keratosis-like Lesions",
            "Psoriasis/Lichen Planus",
            "Seborrheic Keratoses",
            "Tinea/Ringworm/Candidiasis",
            "Warts/Molluscum/Viral Infections",
            "Normal Skin"
        ],
        "anomaly_label": "Melanoma Skin Lesion",
        "normal_label": "Benign Skin Lesion",
        "module": "backend.modules.skin",
        "checkpoint": "../models/skin/skin_disease_CNN_11_classes_zipped.keras",
        "framework": "tensorflow",
        "input_shape": (128, 128),
    }
}

def get_organ_config(organ_id: str) -> dict:
    if not organ_id:
        return ORGAN_REGISTRY["brain"]
    return ORGAN_REGISTRY.get(organ_id.lower(), None)

