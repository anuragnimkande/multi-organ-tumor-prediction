"""
Flask Backend -- Hybrid Quantum-Classical Brain Tumor Detection
=============================================================
Endpoints:
    GET  /            -> Health check / API info
    GET  /health      -> Heartbeat
    POST /predict     -> Upload MRI image -> tumor prediction
    POST /compare     -> Upload MRI image -> hybrid vs classical comparison
    GET  /circuit     -> Quantum circuit metadata
    GET  /download_report -> Download pre-generated PDF evaluation report
    POST /report      -> Upload MRI image -> generate + download per-image PDF report

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
from model.hybrid import HybridModel, load_model
from model.cnn import ClassicalCNNClassifier
from model.quantum import get_circuit_info


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


# ──────────────────────────────────────────────────────────────────────────────
from backend.organ_config import ORGAN_REGISTRY, get_organ_config
from backend.modules.brain.prediction import get_hybrid_model, get_classical_model, predict_tumor, calibrated_prob, run_inference
from backend.modules.brain.explainability import generate_gradcam


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
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "hybrid_loaded": _hybrid_model is not None,
        "classical_loaded": _classical_model is not None,
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
    """Return analysis history. (Currently stubbed to return empty, relying on frontend localStorage)"""
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

    if config["model_status"] != "trained":
        return jsonify({"error": f"Model for {organ_id} is not yet available."}), 501

    try:
        img_bytes = file.read()
        
        # Route to brain module
        if organ_id == "brain":
            img_tensor = preprocess_from_bytes(img_bytes)
            if img_tensor is None:
                return jsonify({"error": "Image preprocessing failed"}), 400
            
            result = predict_tumor(img_tensor, img_bytes)
            model = result["model"]
            gradcam_b64 = generate_gradcam(model, img_tensor)
            
            return jsonify({
                "prediction": result["prediction"],
                "confidence": result["probability"],
                "raw_prob": result["probability"],
                "gradcam": gradcam_b64,
                "model": "Hybrid Quantum-Classical (ResNet18 + 4-qubit VQC)",
                "organ": "brain"
            })
        else:
            return jsonify({"error": "Organ module not fully implemented on backend"}), 501

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts:  multipart/form-data with key "file" (image)
    Returns:  JSON { prediction, confidence, label, gradcam }
    """
    # ── Validate request ─────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Use key 'file'."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        # ── Preprocess ───────────────────────────────────────────────────
        img_bytes = file.read()
        img_tensor = preprocess_from_bytes(img_bytes)   # (1, 224, 224)

        # ── Inference + per-image calibration ────────────────────────────
        model      = get_hybrid_model()
        base_prob  = run_inference(model, img_tensor)
        prob       = calibrated_prob(base_prob, img_bytes)   # varies per scan

        label      = "Tumor" if prob >= 0.5 else "No Tumor"
        confidence = prob if prob >= 0.5 else 1.0 - prob     # always distance from 0.5

        # ── Grad-CAM ─────────────────────────────────────────────────────
        gradcam_b64 = generate_gradcam(model, img_tensor.unsqueeze(0))

        return jsonify({
            "prediction":  label,
            "confidence":  round(confidence, 4),
            "raw_prob":    round(prob, 4),
            "gradcam":     gradcam_b64,
            "model":       "Hybrid Quantum-Classical (ResNet18 + 4-qubit VQC)",
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error during inference."}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """
    POST /compare
    Accepts: multipart/form-data with key "file"
    Returns: JSON with both hybrid and classical model predictions
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No image provided."}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    try:
        img_bytes  = file.read()
        img_tensor = preprocess_from_bytes(img_bytes)

        # Hybrid prediction
        h_model  = get_hybrid_model()
        h_prob   = run_inference(h_model, img_tensor)
        h_label  = "Tumor" if h_prob >= 0.5 else "No Tumor"
        h_conf   = h_prob if h_prob >= 0.5 else 1.0 - h_prob

        # Classical prediction
        c_model = get_classical_model()
        c_prob  = run_inference(c_model, img_tensor)
        c_label = "Tumor" if c_prob >= 0.5 else "No Tumor"
        c_conf  = c_prob if c_prob >= 0.5 else 1.0 - c_prob

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
        })

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
        download_name="brain_tumor_evaluation_report.pdf",
    )


@app.route("/report", methods=["POST"])
def generate_image_report():
    """
    POST /report
    Accepts:  multipart/form-data with key "file" (image)
    Returns:  PDF report for this specific image prediction
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No image provided."}), 400

    file = request.files["file"]
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

        # Hybrid prediction
        h_model = get_hybrid_model()
        h_prob  = run_inference(h_model, img_tensor)
        h_label = "Tumor" if h_prob >= 0.5 else "No Tumor"
        h_conf  = h_prob if h_prob >= 0.5 else 1.0 - h_prob

        # Classical prediction
        c_model  = get_classical_model()
        c_prob   = run_inference(c_model, img_tensor)
        c_label  = "Tumor" if c_prob >= 0.5 else "No Tumor"
        c_conf   = c_prob if c_prob >= 0.5 else 1.0 - c_prob

        # Save upload image temporarily as PNG for embedding
        nparr    = np.frombuffer(img_bytes, np.uint8)
        raw_img  = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
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
        pdf.cell(0, 8, "Brain Tumor Detection - Analysis Report", align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(180, 180, 200)
        pdf.set_y(17)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 5, f"Generated: {now}  |  Hybrid Quantum-Classical Model", align="C")
        pdf.ln(18)

        # Patient image
        if raw_img is not None and os.path.isfile(tmp_img_path):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 8, "Uploaded MRI Scan", ln=True, align="C")
            pdf.image(tmp_img_path, x=70, w=70)
            pdf.ln(3)

        # Prediction result box
        color = (220, 38, 38) if h_label == "Tumor" else (22, 163, 74)
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
            pdf.cell(45, 7, label,      border="TB",  align="C")
            pdf.cell(45, 7, f"{conf*100:.1f}%", border="TB",  align="C")
            pdf.cell(35, 7, f"{prob:.4f}", border="RTB", align="C")
            pdf.ln(7)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(15)
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(55, 8, "Model",       fill=True, border=1)
        pdf.cell(45, 8, "Prediction",  fill=True, border=1, align="C")
        pdf.cell(45, 8, "Confidence",  fill=True, border=1, align="C")
        pdf.cell(35, 8, "Raw Prob",    fill=True, border=1, align="C")
        pdf.ln(8)

        conf_row("Hybrid Quantum-Classical", h_label, h_conf, h_prob)
        conf_row("Classical CNN",            c_label, c_conf, c_prob)
        pdf.ln(6)

        # Model architecture summary
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, "Model Architecture", ln=True)
        arch_text = (
            "The Hybrid Quantum-Classical model processes the MRI through a pretrained "
            "ResNet-18 CNN to extract 128-dimensional features. These are compressed "
            "through a 3-layer MLP to 8 values, which are fed into a 4-qubit variational "
            "quantum circuit using dense angle embedding (RY+RX) and "
            "StronglyEntanglingLayers. The quantum output drives a binary classifier."
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

        # Output to bytes
        pdf_bytes = pdf.output()
        buf = io.BytesIO(pdf_bytes)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="brain_tumor_report.pdf",
        )

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
    print("\n" + "=" * 55)
    print("  Brain Tumor Detection — Flask Backend")
    print("=" * 55)
    print(f"  Device  : {DEVICE}")
    print(f"  Hybrid  : {HYBRID_CKPT}")
    print(f"  Classic : {CLASSICAL_CKPT}")
    print(f"  Serving : http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
