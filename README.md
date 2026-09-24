# ⚛️ QuantumBrain — Hybrid Quantum-Classical Brain Tumor Detection

> **Production-ready hybrid AI system** combining ResNet18 CNN feature extraction with a 4-qubit PennyLane variational quantum circuit for binary brain tumor detection from MRI scans.

---

## 🏗 Architecture

```
MRI Image (150×198 grayscale)
        │
        ▼  Preprocessing (OpenCV)
   Pad-resize 224×224 → Gaussian blur → CLAHE → Normalize [0,1]
        │
        ▼  CNN Feature Extraction (ResNet18, pretrained)
   (batch, 1, 224, 224)  →  (batch, 128)
        │
        ▼  Quantum Input Compression
   Linear(128 → 4) + Tanh
        │
        ▼  Quantum Variational Circuit (PennyLane, 4 qubits)
   AngleEmbedding → [RX, RY, RZ + CNOT ring] × 2 layers → ⟨Z⟩
        │
        ▼  Dense Classifier
   Linear(4 → 1) + Sigmoid  →  Tumor Probability [0, 1]
```

---

## 📁 Project Structure

```
brain_tumor_quantum_v3/
├── dataset/
│   ├── yes/              ← Tumor MRI images
│   └── no/               ← Non-tumor MRI images
│
├── backend/
│   ├── app.py            ← Flask API server
│   ├── model/
│   │   ├── cnn.py        ← ResNet18 feature extractor
│   │   ├── quantum.py    ← PennyLane 4-qubit VQC
│   │   └── hybrid.py     ← HybridModel + checkpointing
│   └── utils/
│       ├── preprocess.py       ← OpenCV preprocessing pipeline
│       └── dataset_loader.py   ← PyTorch dataset + dataloaders
│
├── train.py              ← Full training script
├── evaluate.py           ← Metrics, Grad-CAM, comparison
├── requirements.txt      ← Python dependencies
│
└── frontend/             ← React + Tailwind + Vite
    ├── src/
    │   ├── pages/        ← HomePage, UploadPage, ResultPage
    │   └── components/   ← Navbar, UploadZone, ResultCard, etc.
    └── package.json
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip

---

### 1. Install Python dependencies

```bash
cd brain_tumor_quantum_v3
pip install -r requirements.txt
```

> ⏱ PennyLane + PyTorch install can take 3–5 minutes.

---

### 2. Prepare Dataset

Ensure your dataset is structured as:
```
dataset/
  yes/   ← MRI images WITH tumor  (JPG/PNG)
  no/    ← MRI images WITHOUT tumor
```

---

### 3. Train the Model

```bash
# === OPTION A: Fast Training (Recommended) ===
# Uses pre-cached tensors and is ~10x faster (requires ~1.5GB of disk cache)

# 1. Train hybrid quantum-classical model
python train_fast.py --epochs 15 --batch_size 64 --lr 1e-4

# 2. Train classical CNN baseline for comparison
python train_fast.py --classical --epochs 15 --batch_size 64 --lr 1e-4

# === OPTION B: Standard Training (On-the-fly) ===
# No caching, reads and preprocesses from disk every epoch

# 1. Train hybrid model
python train.py --epochs 15 --batch_size 32 --lr 1e-4

# 2. Train classical CNN
python train.py --classical --epochs 15 --batch_size 32 --lr 1e-4

# Options
python train_fast.py --help
```

**Output:**
- `checkpoints/best_hybrid_quantum.pth` — best model weights
- `logs/training_curves_hybrid_quantum.png` — accuracy/loss plots
- `logs/training_history_hybrid_quantum.json` — per-epoch metrics

---

### 4. Evaluate the Model

```bash
python evaluate.py
```

**Output:**
- `logs/confusion_matrix_hybrid.png`
- `logs/roc_curve_hybrid.png`
- `logs/gradcam_samples.png`
- `logs/evaluation_report.json`

---

### 5. Run the Flask Backend

```bash
python backend/app.py
```

Server starts at: **http://localhost:5000**

---

### 6. Run the React Frontend

```bash
cd frontend
npm install
npm run dev
```

App opens at: **http://localhost:5173**

The Vite dev server proxies all `/predict`, `/compare`, `/health`, `/circuit` calls to Flask on port 5000.

---

## 🌐 API Endpoints

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| GET    | `/`         | API info / serve React build |
| GET    | `/health`   | Server health check |
| POST   | `/predict`  | Upload MRI → tumor prediction |
| POST   | `/compare`  | Upload MRI → classical vs hybrid |
| GET    | `/circuit`  | Quantum circuit metadata |

### Example: POST /predict

```bash
curl -X POST http://localhost:5000/predict \
     -F "file=@path/to/mri.jpg"
```

Response:
```json
{
  "prediction": "Tumor",
  "confidence": 0.9234,
  "raw_prob": 0.9234,
  "gradcam": "data:image/png;base64,...",
  "model": "Hybrid Quantum-Classical (ResNet18 + 4-qubit VQC)"
}
```

---

## ⚛️ Quantum Circuit Details

| Property       | Value |
|----------------|-------|
| Device         | `default.qubit` (PennyLane CPU simulator) |
| Qubits         | 4 |
| Encoding       | `AngleEmbedding` (RY rotation) |
| Gates/layer    | RX + RY + RZ per qubit + CNOT ring |
| Layers         | 2 |
| Measurement    | PauliZ expectation ⟨Z⟩ |
| Trainable params | 24 (2 × 4 × 3) |

---

## 📊 Training Configuration

| Parameter      | Value |
|----------------|-------|
| Loss           | Binary Cross Entropy |
| Optimizer      | Adam |
| Learning rate  | 1e-4 |
| Batch size     | 16 |
| LR Scheduler   | ReduceLROnPlateau (patience=3) |
| Data split     | 70 / 15 / 15 (train/val/test) |
| CNN backbone   | ResNet18 (ImageNet pretrained) |

---

## 🖥 Frontend Pages

| Page         | Route      | Description |
|--------------|-----------|-------------|
| Home         | `/`        | Hero, pipeline overview, circuit diagram |
| Upload       | `/upload`  | Drag-and-drop MRI upload, preview |
| Result       | `/result`  | Prediction, confidence, Grad-CAM, comparison |

---

## 🔧 Advanced Configuration

### Build React for production (served by Flask)

```bash
cd frontend
npm run build
# dist/ is served automatically by Flask at /
```

### GPU support

Install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For PennyLane GPU device:
```python
dev = qml.device("lightning.gpu", wires=4)
```

---

## ⚠️ Important Notes

1. **First run** — the model runs in "demo mode" (untrained) until you run `train.py`
2. **CPU training** — quantum simulation on CPU takes ~20–30 minutes for 20 epochs
3. **Medical disclaimer** — this is a research prototype, not a clinical tool

---

## 📦 Dependencies

| Package         | Version  | Purpose |
|-----------------|----------|---------|
| torch           | 2.3.1    | Deep learning |
| torchvision     | 0.18.1   | ResNet18 pretrained weights |
| pennylane       | 0.36.0   | Quantum ML |
| opencv-python   | 4.10.0   | Image preprocessing |
| flask           | 3.0.3    | Web server |
| flask-cors      | 4.0.1    | CORS headers |
| scikit-learn    | 1.5.1    | Metrics |
| matplotlib      | 3.9.1    | Visualization |
| seaborn         | 0.13.2   | Confusion matrix |

---

*Built with ❤️ using PyTorch + PennyLane + Flask + React*
