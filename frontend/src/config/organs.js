export const ORGANS = {
  // ✅ Brain: trained checkpoints confirmed present (best_hybrid_quantum.pth + best_classical_cnn.pth)
  brain: {
    id: 'brain',
    name: 'Brain Tumor',
    modality: 'MRI',
    icon: '🧠',
    color: '#7c3aed',
    status: 'active',
    classes: ['No Tumor', 'Tumor'],
    description: 'Hybrid quantum-classical detection of brain tumors from MRI scans.',
  },

  // ⏳ Lung: stub code only, no trained model file (best_hybrid_lung.pth missing)
  lung: {
    id: 'lung',
    name: 'Lung Tumor',
    modality: 'CT',
    icon: '🫁',
    color: '#06b6d4',
    status: 'coming_soon',
    classes: ['Normal', 'Lung Tumor'],
    description: 'Early stage detection of lung nodules from CT imaging. Model training in progress.',
  },

  // ⏳ Breast: stub code only, no trained model file (best_hybrid_breast.pth missing)
  breast: {
    id: 'breast',
    name: 'Breast Tumor',
    modality: 'Mammography',
    icon: '🎗️',
    color: '#ec4899',
    status: 'coming_soon',
    classes: ['Benign', 'Malignant Breast Tumor'],
    description: 'Screening and classification of breast lesions via mammography. Model training in progress.',
  },

  // ✅ Liver: Keras MobileNetV2 trained model present
  liver: {
    id: 'liver',
    name: 'Liver Tumor',
    modality: 'CT',
    icon: '🫀',
    color: '#f59e0b',
    status: 'active',
    classes: ['Angiosarcoma', 'Cholangiocarcinoma', 'Healthy', 'Hemangioma', 'Hepatocellular_Carcinoma'],
    description: 'Automated detection of liver lesions. MobileNetV2 model training in progress.',
  },

  // ✅ Skin: Keras CNN trained model present
  skin: {
    id: 'skin',
    name: 'Skin Lesion',
    modality: 'Dermoscopy',
    icon: '🔬',
    color: '#10b981',
    status: 'active',
    classes: [
      'Eczema', 'Melanoma', 'Atopic Dermatitis', 'Basal Cell Carcinoma',
      'Melanocytic Nevi', 'Benign Keratosis-like Lesions',
      'Psoriasis/Lichen Planus', 'Seborrheic Keratoses',
      'Tinea/Ringworm/Candidiasis', 'Warts/Molluscum/Viral Infections', 'Normal Skin',
    ],
    description: 'Multi-class skin lesion classification from dermoscopic images. MobileNetV2 model training in progress.',
  },
};

export const getOrganList = () => Object.values(ORGANS);

