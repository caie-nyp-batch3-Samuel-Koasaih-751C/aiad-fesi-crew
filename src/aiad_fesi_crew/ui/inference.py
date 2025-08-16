from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import cv2
import tensorflow as tf

# Resolve model from env or default to Kedro convention (data/06_models)
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "data" / "06_models" / "model.h5"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))

# Load once at import time
_model = tf.keras.models.load_model(MODEL_PATH)
# classes inferred from 05_model_input/train/* unless provided
CLASSES_DIR = Path(os.getenv("CLASSES_DIR", str(DEFAULT_MODEL.parents[2] / "05_model_input" / "train")))
if CLASSES_DIR.exists():
    CLASS_NAMES = sorted([p.name for p in CLASSES_DIR.iterdir() if p.is_dir()])
else:
    CLASS_NAMES = [
        'Pepper bell Bacterial spot','Pepper bell healthy',
        'Potato Early blight','Potato healthy','Potato Late blight',
        'Tomato Bacterial spot','Tomato Early blight','Tomato healthy',
        'Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot',
        'Tomato Spider mites Two spotted spider mite','Tomato Target Spot',
        'Tomato mosaic virus','Tomato YellowLeaf Curl Virus'
    ]

IMAGE_SIZE = (256, 256)

def _is_black_background(gray: np.ndarray, threshold=30, sampling_width=20) -> bool:
    top = gray[:sampling_width, :]
    bottom = gray[-sampling_width:, :]
    left = gray[:, :sampling_width]
    right = gray[:, -sampling_width:]
    sampled = np.concatenate((top, bottom, left, right), axis=None)
    return np.mean(sampled) < threshold

def _refine_mask(binary_mask: np.ndarray, min_leaf_area=500) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.erode(binary_mask, kernel, iterations=1)
    m = cv2.dilate(m, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    refined = np.zeros_like(m)
    for lab in range(1, num):
        if stats[lab, cv2.CC_STAT_AREA] >= min_leaf_area:
            refined[labels == lab] = 255
    return cv2.dilate(refined, np.ones((7, 7), np.uint8), iterations=1)

def mask_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if _is_black_background(gray):
        _, binmask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (30, 30, 30), (90, 255, 255))
        brown = cv2.inRange(hsv, (10, 30, 20), (30, 255, 200))
        blue  = cv2.inRange(hsv, (100, 100, 50), (130, 255, 255))
        yellow= cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
        binmask = green | brown | blue | yellow
    return _refine_mask(binmask)

def preprocess_for_model(image_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(image_bgr, IMAGE_SIZE).astype("float32") / 255.0
    return img[None, ...]  # (1,H,W,3)

def predict(image_bgr: np.ndarray) -> dict:
    mask = mask_image(image_bgr)
    img = image_bgr.copy()
    img[mask == 0] = (0, 0, 0)
    x = preprocess_for_model(img)
    probs = _model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
        "confidence": float(probs[idx]),
    }
