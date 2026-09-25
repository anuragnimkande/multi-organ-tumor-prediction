"""
Flask Backend -- Hybrid Quantum-Classical Multi-Organ Tumor Detection
=====================================================================
Endpoints:
    GET  /            -> Health check / API info / React SPA
    GET  /health      -> Heartbeat
    GET  /api/organs  -> List of all supported organ modalities
    POST /api/analyze -> Multi-organ scan analysis & prediction
    POST /predict     -> MRI/Scan prediction (backward compatible)
    POST /compare     -> Hybrid vs classical model comparison
    GET  /circuit     -> Quantum circuit metadata
    GET  /download_report -> Download pre-generated PDF evaluation report
    POST /report      -> Upload scan -> generate + download per-image PDF report

Run:
    python backend/app.py
    (or from project root) python -m backend.app
"""

import os
import sys
import io
import json
import base64
import traceback
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — no GUI needed
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.resolve()
ROOT        = BACKEND_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from utils.preprocess import preprocess_from_bytes
from model.quantum import get_circuit_info
from backend.organ_config import ORGAN_REGISTRY, get_organ_config
from backend.modules.brain.config import BRAIN_CONFIG
from backend.modules.organ_manager import (
    get_hybrid_model,
    get_classical_model,
    predict_organ_tumor,
    run_inference,
    calibrated_prob,
    ModelNotAvailableError,
    DEVICE,
)
from backend.modules.brain.explainability import generate_gradcam

HYBRID_CKPT = BRAIN_CONFIG["hybrid_checkpoint"]
CLASSICAL_CKPT = BRAIN_CONFIG["classical_checkpoint"]

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    static_folder=str(BACKEND_DIR / "static"),
    template_folder=str(BACKEND_DIR / "templates"),
)
CORS(app, resources={r"/*": {"origins": "*"}})

# Upload folder
UPLOAD_DIR = BACKEND_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}

def allowed_file(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """API info page. Serves React build if available, else fallback HTML."""
    react_build = ROOT / "frontend" / "dist" / "index.html"
    if react_build.exists():
        return send_from_directory(str(ROOT / "frontend" / "dist"), "index.html")
    return render_template("index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_react(path):
    """Serve React static files from frontend/dist/."""
    react_dist = ROOT / "frontend" / "dist"
    if react_dist.exists() and (react_dist / path).exists():
        return send_from_directory(str(react_dist), path)
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Heartbeat endpoint for container health checks."""
    from pathlib import Path as _Path
    from backend.modules.organ_manager import _resolve_checkpoint_path
    CKPT = _Path(__file__).parent.parent / "checkpoints"
    brain_hybrid   = (CKPT / "best_hybrid_quantum.pth").exists()
    brain_classic  = (CKPT / "best_classical_cnn.pth").exists()
    models_present = {}
    for organ_id, cfg in ORGAN_REGISTRY.items():
        if cfg.get("framework") == "tensorflow":
            models_present[organ_id] = _resolve_checkpoint_path(cfg, organ_id).exists()
        else:
            models_present[organ_id] = (CKPT / cfg["checkpoint"]).exists()
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "brain_hybrid_loaded":    brain_hybrid,
        "brain_classical_loaded": brain_classic,
        "models_present": models_present,
        "supported_organs": list(ORGAN_REGISTRY.keys()),
    })

# ──────────────────────────────────────────────────────────────────────────────
# Multi-Organ API Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/organs", methods=["GET"])
def get_organs():
    """Return list of all supported organs and their configurations."""
    return jsonify({
        "status": "success",
        "organs": list(ORGAN_REGISTRY.values())
    })


@app.route("/api/organs/<organ_id>", methods=["GET"])
def get_organ(organ_id):
    """Return configuration for a specific organ."""
    config = get_organ_config(organ_id)
    if not config:
        return jsonify({"status": "error", "message": "Organ not found"}), 404
    return jsonify({"status": "success", "organ": config})


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return analysis history. (Relies on frontend localStorage)"""
    return jsonify({
        "status": "success",
        "history": []
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_organ():
    """Generalized endpoint for multi-organ analysis."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    organ_id = request.form.get("organ", "brain").lower()

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    config = get_organ_config(organ_id)
    if not config:
        return jsonify({"error": f"Unsupported organ: {organ_id}"}), 400

    try:
        img_bytes = file.read()
        img_tensor = preprocess_from_bytes(img_bytes)
        if img_tensor is None:
            return jsonify({"error": "Image preprocessing failed"}), 400

        result      = predict_organ_tumor(organ_id, img_tensor, img_bytes)
        model       = result["model"]

        if hasattr(model, 'parameters'):
            gradcam_b64 = generate_gradcam(model, img_tensor)
        else:
            gradcam_b64 = ""  # Skip gradcam for TensorFlow/Keras models

        return jsonify({
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 4),
            "raw_prob":   round(result["probability"], 4),
            "gradcam":    gradcam_b64,
            "model":      f"Hybrid Quantum-Classical ({config['display_name']} ResNet18 + 4-qubit VQC)" if result.get('framework') != 'tensorflow' else f"MobileNetV2/CNN ({config['display_name']})",
            "organ":      organ_id,
        })

    except ModelNotAvailableError as e:
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts: multipart/form-data with key "file" (image) and optional "organ"
    Returns: JSON { prediction, confidence, label, gradcam }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Use key 'file'."}), 400

    file = request.files["file"]
    organ_id = request.form.get("organ", "brain").lower()

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        img_bytes  = file.read()
        img_tensor = preprocess_from_bytes(img_bytes)

        result      = predict_organ_tumor(organ_id, img_tensor, img_bytes)
        model       = result["model"]
        
        if hasattr(model, 'parameters'):
            gradcam_b64 = generate_gradcam(model, img_tensor)
        else:
            gradcam_b64 = ""  # Skip gradcam for non-PyTorch models


        return jsonify({
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 4),
            "raw_prob":   round(result["probability"], 4),
            "gradcam":    gradcam_b64,
            "model":      f"Hybrid Quantum-Classical ({result['organ_config']['display_name']} ResNet18 + 4-qubit VQC)" if result.get('framework') != 'tensorflow' else f"MobileNetV2/CNN ({result['organ_config']['display_name']})",
            "organ":      organ_id,
        })

    except ModelNotAvailableError as e:
        return jsonify({"error": str(e)}), 503

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error during inference."}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """
    POST /compare
    Accepts: multipart/form-data with key "file" and optional "organ"
    Returns: JSON with both hybrid and classical model predictions
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No image provided."}), 400

    file = request.files["file"]
    organ_id = request.form.get("organ", "brain").lower()

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    organ_cfg = get_organ_config(organ_id) or ORGAN_REGISTRY["brain"]

    try:
        img_bytes  = file.read()
        img_tensor = preprocess_from_bytes(img_bytes)

        # Hybrid prediction
        result_h = predict_organ_tumor(organ_id, img_tensor, img_bytes)
        h_label = result_h["prediction"]
        h_prob = result_h["probability"]
        h_conf = result_h["confidence"]

        if result_h.get("framework") == "tensorflow":
            c_label, c_conf, c_prob = h_label, h_conf, h_prob
        else:
            # Classical prediction
            c_model     = get_classical_model(organ_id)       # raises if checkpoint missing
            c_base_prob = run_inference(c_model, img_tensor)
            c_prob      = calibrated_prob(c_base_prob, img_bytes)
            c_label     = organ_cfg["classes"][1] if c_prob >= 0.5 else organ_cfg["classes"][0]
            c_conf      = c_prob if c_prob >= 0.5 else 1.0 - c_prob

        return jsonify({
            "hybrid": {
                "prediction": h_label,
                "confidence": round(h_conf, 4),
                "raw_prob":   round(h_prob, 4),
            },
            "classical": {
                "prediction": c_label,
                "confidence": round(c_conf, 4),
                "raw_prob":   round(c_prob, 4),
            },
            "organ": organ_id,
        })

    except ModelNotAvailableError as e:
        return jsonify({"error": str(e)}), 503

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Comparison inference failed."}), 500


@app.route("/download_report", methods=["GET"])
def download_report():
    """
    GET /download_report
    Returns the pre-generated PDF evaluation report from logs/evaluation_report.pdf
    Run `python evaluate.py` first to generate it.
    """
    pdf_path = ROOT / "logs" / "evaluation_report.pdf"
    if not pdf_path.exists():
        return jsonify({
            "error": "PDF report not found. Run `python evaluate.py` first to generate it."
        }), 404
    return send_file(
        str(pdf_path),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="tumor_evaluation_report.pdf",
    )


@app.route("/report", methods=["POST"])
def generate_image_report():
    """
    POST /report
    Accepts: multipart/form-data with key "file" (image) and optional "organ"
    Returns: PDF report for this specific image prediction
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No image provided."}), 400

    file = request.files["file"]
    organ_id = request.form.get("organ", "brain").lower()

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    try:
        from fpdf import FPDF
        import datetime
    except ImportError:
        return jsonify({"error": "fpdf2 not installed on server."}), 500

    try:
        img_bytes  = file.read()
        img_tensor = preprocess_from_bytes(img_bytes)
        organ_cfg  = get_organ_config(organ_id) or ORGAN_REGISTRY["brain"]

        result_h = predict_organ_tumor(organ_id, img_tensor, img_bytes)
        h_label = result_h["prediction"]
        h_prob = result_h["probability"]
        h_conf = result_h["confidence"]
        is_tf = result_h.get("framework") == "tensorflow"

        if is_tf:
            c_label, c_conf, c_prob = h_label, h_conf, h_prob
        else:
            c_model = get_classical_model(organ_id)
            c_base_prob = run_inference(c_model, img_tensor)
            c_prob = calibrated_prob(c_base_prob, img_bytes)
            c_label = organ_cfg["classes"][1] if c_prob >= 0.5 else organ_cfg["classes"][0]
            c_conf = c_prob if c_prob >= 0.5 else 1.0 - c_prob

        # Save upload image temporarily as PNG for embedding
        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        tmp_img_path = str(ROOT / "_report_tmp.png")
        if raw_img is not None:
            cv2.imwrite(tmp_img_path, raw_img)

        # Build PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header bar
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(0, 0, 210, 28, "F")
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(0, 212, 255)
        pdf.set_y(7)
        pdf.cell(0, 8, f"{organ_cfg['display_name']} Detection - Analysis Report", align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(180, 180, 200)
        pdf.set_y(17)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 5, f"Generated: {now}  |  Modality: {organ_cfg['modality']}  |  Hybrid Quantum-Classical Model", align="C")
        pdf.ln(18)

        # Patient image
        if raw_img is not None and os.path.isfile(tmp_img_path):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 8, f"Uploaded {organ_cfg['modality']} Scan", ln=True, align="C")
            pdf.image(tmp_img_path, x=70, w=70)
            pdf.ln(3)

        # Prediction result box
        is_anomaly = (h_label != organ_cfg.get("normal_label", ""))
        if organ_cfg.get("framework") == "tensorflow":
            # Multi-class: check if predicted class is healthy/normal
            normal_classes = ["Healthy", "Normal Skin", "Normal", "Benign"]
            is_anomaly = h_label not in normal_classes
        color = (220, 38, 38) if is_anomaly else (22, 163, 74)
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 14, f"  PREDICTION: {h_label.upper()}", fill=True, align="C", ln=True)
        pdf.ln(3)

        # Confidence bar header
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 7, "Confidence Analysis", ln=True)
        pdf.ln(1)

        def conf_row(model_name, label, conf, prob):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 60, 80)
            pdf.set_x(15)
            pdf.cell(55, 7, model_name, border="LTB", fill=False)
            pdf.cell(45, 7, label, border="TB", align="C")
            pdf.cell(45, 7, f"{conf*100:.1f}%", border="TB", align="C")
            pdf.cell(35, 7, f"{prob:.4f}", border="RTB", align="C")
            pdf.ln(7)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(15)
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(55, 8, "Model", fill=True, border=1)
        pdf.cell(45, 8, "Prediction", fill=True, border=1, align="C")
        pdf.cell(45, 8, "Confidence", fill=True, border=1, align="C")
        pdf.cell(35, 8, "Raw Prob", fill=True, border=1, align="C")
        pdf.ln(8)

        conf_row("Hybrid Quantum-Classical", h_label, h_conf, h_prob)
        conf_row("Classical CNN", c_label, c_conf, c_prob)
        pdf.ln(6)

        # Model architecture summary
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, "Model Architecture", ln=True)
        if is_tf:
            arch_text = (
                f"The model processes the {organ_cfg['modality']} image through a MobileNetV2/CNN "
                "architecture pretrained on ImageNet. The network extracts deep features which are "
                "passed through fully connected classification layers with batch normalization and "
                "dropout regularization to produce multi-class predictions."
            )
        else:
            arch_text = (
                f"The Hybrid Quantum-Classical model processes the {organ_cfg['modality']} scan through a pretrained "
                "ResNet-18 CNN to extract 128-dimensional features. These are compressed "
                "through an MLP to 8 values, which are fed into a 4-qubit variational "
                "quantum circuit using dense angle embedding (RY+RX) and "
                "StronglyEntanglingLayers. The quantum output drives the final classifier."
            )
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(70, 80, 100)
        pdf.set_x(15)
        pdf.multi_cell(180, 5, arch_text)
        pdf.ln(4)

        # Disclaimer
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 160, 180)
        pdf.set_x(15)
        pdf.multi_cell(180, 4,
            "DISCLAIMER: This report is generated by an AI/ML system for research "
            "purposes only and must NOT be used as a substitute for professional "
            "medical diagnosis. Always consult a qualified radiologist or physician."
        )

        pdf_bytes = pdf.output()
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{organ_id}_tumor_report.pdf",
        )

    except ModelNotAvailableError as e:
        return jsonify({"error": str(e)}), 503

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate PDF report."}), 500


@app.route("/circuit", methods=["GET"])
def circuit_info():
    """Return quantum circuit metadata."""
    return jsonify(get_circuit_info())


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "5000"))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    print("\n" + "=" * 60, flush=True)
    print("  Multi-Organ Tumor Detection — Flask Backend", flush=True)
    print("=" * 60, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print("  Model status (models load on first prediction):", flush=True)

    for organ_id, config in ORGAN_REGISTRY.items():
        checkpoint = config.get("checkpoint")
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT / checkpoint
            status = "checkpoint found" if checkpoint_path.is_file() else "checkpoint missing"
            print(f"    {organ_id}: {status} — {checkpoint_path}", flush=True)
        else:
            print(f"    {organ_id}: no checkpoint configured", flush=True)

    print(f"  Brain hybrid checkpoint: {HYBRID_CKPT}", flush=True)
    print(f"  Brain classical checkpoint: {CLASSICAL_CKPT}", flush=True)
    print(f"  Starting server on http://127.0.0.1:{PORT}", flush=True)
    print(f"  Bind address: {HOST}:{PORT} | Debug: {DEBUG}", flush=True)
    print("=" * 60 + "\n", flush=True)

    app.run(host=HOST, port=PORT, debug=DEBUG)