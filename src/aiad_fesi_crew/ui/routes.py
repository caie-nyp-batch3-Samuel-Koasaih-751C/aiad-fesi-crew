from __future__ import annotations
from flask import Blueprint, request, render_template, jsonify
import numpy as np, cv2
from .inference import predict

bp = Blueprint("ui", __name__)

@bp.route("/")
def home():
    return render_template("home.html")

@bp.route("/about")
def about():
    return render_template("about.html")

@bp.route("/reviews")
def reviews():
    return render_template("reviews.html")

@bp.route("/model")
def model_page():
    return render_template("model.html")

@bp.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files["image"]
    data = f.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    try:
        result = predict(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
