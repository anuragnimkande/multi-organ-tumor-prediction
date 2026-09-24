export const ORGANS = {
  brain:  { id: 'brain',  name: 'Brain Tumor',  modality: 'MRI',          icon: '🧠', color: '#7c3aed', status: 'active',  description: 'Hybrid quantum-classical detection of brain tumors from MRI scans.' },
  lung:   { id: 'lung',   name: 'Lung Tumor',   modality: 'CT',           icon: '🫁', color: '#06b6d4', status: 'coming_soon', description: 'Early stage detection of lung nodules from CT imaging.' },
  breast: { id: 'breast', name: 'Breast Tumor', modality: 'Mammography',  icon: '🎗️', color: '#ec4899', status: 'coming_soon', description: 'Screening and classification of breast lesions via mammography.' },
  liver:  { id: 'liver',  name: 'Liver Tumor',  modality: 'CT',           icon: '🫀', color: '#f59e0b', status: 'coming_soon', description: 'Automated segmentation and detection of liver lesions.' },
  skin:   { id: 'skin',   name: 'Skin Lesion',  modality: 'Dermoscopy',   icon: '🔬', color: '#10b981', status: 'coming_soon', description: 'Melanoma and benign lesion classification from dermoscopic images.' },
};

export const getOrganList = () => Object.values(ORGANS);
