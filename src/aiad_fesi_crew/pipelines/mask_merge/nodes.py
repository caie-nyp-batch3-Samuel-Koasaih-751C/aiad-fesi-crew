from __future__ import annotations
from typing import Dict, Tuple, Literal
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

Mode = Literal["foreground", "alpha", "crop"]

def _mask_bool(mask_img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    m = mask_img.convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return np.asarray(m) > 0  # WHITE=keep

def _bbox(m: np.ndarray, margin: int = 0):
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = m.shape
    return max(0, x0 - margin), max(0, y0 - margin), min(w - 1, x1 + margin), min(h - 1, y1 + margin)

def _key_norm(p: str) -> str:
    """Normalize PartitionedDataSet keys so images and masks match even if roots differ.
    We keep only the filename (and optional one parent dir) to be tolerant."""
    parts = Path(p).parts
    # try "<parent>/<file>" if available, else just "<file>"
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

def combine_images_and_masks(
    raw_images: Dict[str, Image.Image],
    leaf_masks: Dict[str, Image.Image],
    mode: Mode = "foreground",
    background_color: Tuple[int, int, int] = (0, 0, 0),
    bbox_margin: int = 5,
):
    """Merge originals + masks (white=keep). Returns (combined_images, bbox_index)."""
    # Build lookup by normalized key so folder structures can differ
    imgs = {_key_norm(k): v.convert("RGB") for k, v in raw_images.items()}
    masks = {_key_norm(k): v for k, v in leaf_masks.items()}

    out: Dict[str, Image.Image] = {}
    rows = []
    common = sorted(set(imgs) & set(masks))
    missing = sorted(set(imgs) - set(masks))

    for key in common:
        img = imgs[key]
        w, h = img.size
        keep = _mask_bool(masks[key], (w, h))
        box = _bbox(keep, bbox_margin)

        if mode == "alpha":
            rgba = np.dstack([np.asarray(img), (keep * 255).astype(np.uint8)])
            merged = Image.fromarray(rgba, mode="RGBA")
        elif mode == "crop" and box:
            x0, y0, x1, y1 = box
            merged = img.crop((x0, y0, x1 + 1, y1 + 1))
        else:
            arr = np.asarray(img).copy()
            arr[~keep] = np.array(background_color, dtype=arr.dtype)
            merged = Image.fromarray(arr, mode="RGB")

        out[key] = merged
        if box:
            x0, y0, x1, y1 = box
            rows.append({"filepath": key, "has_mask": True, "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1})
        else:
            rows.append({"filepath": key, "has_mask": False, "x_min": 0, "y_min": 0, "x_max": w - 1, "y_max": h - 1})

    # log images without masks (useful to catch mismatches)
    for key in missing:
        w, h = imgs[key].size
        rows.append({"filepath": key, "has_mask": False, "x_min": 0, "y_min": 0, "x_max": w - 1, "y_max": h - 1})

    return out, pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)
