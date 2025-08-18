from __future__ import annotations
from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for
from pathlib import Path
import numpy as np
import cv2
import tempfile

bp = Blueprint("ui", __name__)

# --------- Labels & constants ----------
DISEASE_TYPES = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy',
    'Potato Early blight', 'Potato healthy', 'Potato Late blight',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato healthy',
    'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato mosaic virus', 'Tomato YellowLeaf Curl Virus'
]
IMAGE_SIZE = (256, 256)
NORMALIZATION_FACTOR = 255.0


# --------- Image helpers ----------
def is_black_background(gray: np.ndarray, threshold: int = 30, sampling_width: int = 20) -> bool:
    top_edge = gray[:sampling_width, :]
    bottom_edge = gray[-sampling_width:, :]
    left_edge = gray[:, :sampling_width]
    right_edge = gray[:, -sampling_width:]
    sampled = np.concatenate((top_edge, bottom_edge, left_edge, right_edge), axis=None)
    return float(np.mean(sampled)) < threshold


def refine_mask_with_smoothing(binary_mask: np.ndarray, *, min_leaf_area: int = 500) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    smoothed = cv2.erode(binary_mask, kernel, iterations=1)
    smoothed = cv2.dilate(smoothed, kernel, iterations=1)

    closing_kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, closing_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    refined = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_leaf_area:
            refined[labels == label] = 255

    dilation_kernel = np.ones((7, 7), np.uint8)
    refined = cv2.dilate(refined, dilation_kernel, iterations=1)
    return refined


def apply_mask_and_crop(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if is_black_background(gray):
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
        brown = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([30, 255, 200]))
        blue  = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))
        yellow= cv2.inRange(hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))
        binary = cv2.bitwise_or(cv2.bitwise_or(green, brown), cv2.bitwise_or(blue, yellow))

    return refine_mask_with_smoothing(binary, min_leaf_area=500)


def preprocess_single_image(image_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(image_bgr, IMAGE_SIZE)
    img = img / NORMALIZATION_FACTOR
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)
    return img


# --------- Pages ----------
@bp.get("/healthz")
def healthz():
    return {"status": "ok"}, 200


@bp.get("/")
def home():
    return render_template("home.html")


@bp.get("/about")
def about():
    return render_template("about.html")


@bp.get("/reviews")
def reviews():
    return render_template("reviews.html")


@bp.get("/model")
def model_page():
    return render_template("model.html")


@bp.get("/start")
def start():
    return redirect(url_for("ui.model_page"), code=302)


# --------- API ----------
@bp.post("/predict")
def predict():
    """
    POST form-data: image=<file>
    Returns JSON: {"prediction": <label>} or {"error": "..."}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    up = request.files["image"]
    if not up or up.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read image from upload into OpenCV
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / up.filename
            up.save(p)
            buf = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Invalid image format"}), 400

        # Mask & preprocess
        mask = apply_mask_and_crop(img)
        img[mask == 0] = (0, 0, 0)
        x = preprocess_single_image(img)

        # Predict
        model = current_app.ensure_model()  # type: ignore[attr-defined]
        y = model.predict(x, verbose=0)
        idx = int(np.argmax(y, axis=1))
        label = DISEASE_TYPES[idx]
        return jsonify({"prediction": label})

    except Exception as e:  # keep JSON on error so the front-end doesn't choke on HTML
        return jsonify({"error": str(e)}), 500
