# Hybrid Quantum-Classical Brain Tumor Detection System

## Overview

A production-ready system combining classical deep learning (ResNet18 CNN) with quantum machine learning (PennyLane) to detect brain tumors in MRI scans. The pipeline flows: MRI Image → Preprocessing → CNN Feature Extraction → Quantum Circuit → Binary Prediction.

The dataset is already present at `d:/VIT/6TH_SEM/EDI/Antigravity/brain_tumor_quantum_v3/dataset/` with `yes/` and `no/` subdirectories.

---

## Architecture Diagram

```
MRI Image (150×198 grayscale)
        │
        ▼
┌─────────────────────┐
│  Preprocessing       │  → Pad-resize to 224×224, CLAHE, Gaussian blur, normalize
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ResNet18 (CNN)     │  → Pretrained, 1-ch→3-ch duplication, 128-dim feature vector
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Quantum Circuit    │  → 4 qubits, AngleEmbedding, RX/RY/RZ + CNOT, 2 layers
│  (PennyLane)        │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Dense + Sigmoid    │  → Binary output: Tumor / No Tumor
└─────────────────────┘
```

---

## Project Structure

```
brain_tumor_quantum_v3/
├── dataset/
│   ├── yes/          (tumor images)
│   └── no/           (non-tumor images)
│
├── backend/
│   ├── app.py                    Flask app entry point
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cnn.py                ResNet18 feature extractor
│   │   ├── quantum.py            PennyLane quantum circuit + TorchLayer
│   │   └── hybrid.py             HybridModel combining CNN + Quantum
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocess.py         OpenCV preprocessing pipeline
│   │   └── dataset_loader.py     PyTorch dataset class + loaders
│   ├── static/
│   │   └── uploads/              Temp upload storage
│   └── templates/
│       └── index.html            Minimal fallback HTML
│
├── train.py                      Full training script
├── evaluate.py                   Metrics + confusion matrix + accuracy plots
├── requirements.txt              All Python dependencies
└── README.md                     Setup + usage guide
│
└── frontend/                     React + Tailwind CSS
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── components/
        │   ├── Navbar.jsx
        │   ├── UploadZone.jsx
        │   ├── ResultCard.jsx
        │   ├── ConfidenceMeter.jsx
        │   ├── HeatmapView.jsx
        │   └── LoadingAnimation.jsx
        └── pages/
            ├── HomePage.jsx
            ├── UploadPage.jsx
            └── ResultPage.jsx
```

---

## Proposed Changes

### Backend — Python ML Pipeline

#### [NEW] `backend/utils/preprocess.py`
- Load image as grayscale via OpenCV
- Pad-resize to 224×224 (no distortion)
- Apply Gaussian blur (3×3) → CLAHE enhancement
- Normalize to [0,1] → output tensor shape `(1, 224, 224)`

#### [NEW] `backend/utils/dataset_loader.py`
- `BrainTumorDataset(Dataset)` class scanning `yes/` and `no/` directories
- Applies preprocessing pipeline per sample
- `get_dataloaders()` returning train/val/test splits (70/15/15)

#### [NEW] `backend/model/cnn.py`
- Load pretrained ResNet18
- Adapt first conv layer: `Conv2d(1, 64, ...)` OR duplicate grayscale to 3 channels
- Remove `fc` layer, replace with `Linear(512, 128)` + ReLU
- Output: 128-dim feature vector

#### [NEW] `backend/model/quantum.py`
- `dev = qml.device("default.qubit", wires=4)`
- `@qml.qnode(dev, interface="torch")` circuit
- AngleEmbedding (4 features from compressed 128→4 linear)
- 2 layers: RX, RY, RZ on each qubit + CNOT entanglement ring
- Return `[qml.expval(qml.PauliZ(i)) for i in range(4)]`
- Wrap with `qml.qnn.TorchLayer`

#### [NEW] `backend/model/hybrid.py`
- `HybridModel(nn.Module)`:
  1. CNN feature extractor → 128-dim
  2. Linear(128→4) for quantum input compression
  3. QuantumLayer (TorchLayer) → 4-dim
  4. Linear(4→1) + Sigmoid
- `load_model()` utility function

#### [NEW] `train.py`
- Load dataset, build HybridModel
- Adam optimizer (lr=1e-4), BCELoss, batch_size=16
- Training loop with train/val metrics per epoch
- Save best model checkpoint `best_model.pth`
- Log: accuracy, precision, recall, F1

#### [NEW] `evaluate.py`
- Load saved model, run on test split
- Plot confusion matrix (matplotlib/seaborn)
- Plot accuracy/loss curves
- Generate Grad-CAM heatmap for sample images
- Classical vs Hybrid comparison table

#### [NEW] `backend/app.py`
- Flask app with CORS enabled
- `GET /` → serves React build OR fallback HTML
- `POST /predict` → accepts multipart image, runs full pipeline, returns JSON
- `POST /compare` → returns classical CNN vs hybrid model scores
- `GET /health` → heartbeat endpoint
- Error handling for invalid/missing images

---

### Frontend — React + Tailwind

#### [NEW] `frontend/` (Vite + React + Tailwind)
- **HomePage**: Hero section, "Quantum AI Brain Tumor Detection" branding, animated quantum particle background, CTA button
- **UploadPage**: Drag-and-drop zone, image preview with CLAHE preview toggle, loading state with animated scanner graphic
- **ResultPage**: Prediction badge (Tumor / No Tumor), confidence progress bar, Grad-CAM heatmap overlay, Classical vs Hybrid comparison table, download report button
- **Navbar**: Logo, navigation links, dark mode toggle
- **LoadingAnimation**: Brain scan animation with "Analyzing MRI through quantum circuits..." text

**Design System:**
- Dark mode default
- Color palette: `#0a0f1e` (bg), `#00d4ff` (quantum blue), `#7c3aed` (purple accent), `#10b981` (success), `#ef4444` (danger)
- Google Font: Inter + Space Grotesk
- Glassmorphism cards, gradient borders, smooth transitions

---

## Verification Plan

### Automated Tests
1. Run `python train.py` — confirm training loop completes
2. Run `python evaluate.py` — confirm metrics output + plot files saved
3. Run `python backend/app.py` — confirm Flask starts on port 5000
4. `curl -X POST /predict` with test image — confirm JSON response
5. `npm run dev` in `/frontend` — confirm React app serves on port 5173

### Manual Verification
- Upload a MRI sample from `/dataset/yes/` → expect "Tumor" + high confidence
- Upload a MRI sample from `/dataset/no/` → expect "No Tumor"
- Verify Grad-CAM heatmap renders on result page
- Test drag-and-drop + file picker upload flows
- Verify mobile responsive layout

---

## Open Questions

> [!IMPORTANT]
> **Do you have a GPU available?** Training with `default.qubit` PennyLane device on CPU is functional but slow (~15-30 min for full dataset). Should I add GPU support (`lightning.gpu` device) or keep CPU-only for compatibility?

> [!IMPORTANT]
> **Pre-trained weights**: Should the system ship with pre-trained weights (requires running `train.py` first), or should I mock the quantum inference so the Flask/React app works immediately without training?

> [!NOTE]
> **Dataset size**: How many images are in `/yes` and `/no`? If < 50 per class, I'll add data augmentation (random flip, rotation, brightness jitter) to prevent overfitting.

> [!NOTE]
> **Frontend build**: The React app uses Vite. Do you want it served by Flask (as a static build) or run as a separate dev server on port 5173? For production I'll set up Flask to serve the React build.
