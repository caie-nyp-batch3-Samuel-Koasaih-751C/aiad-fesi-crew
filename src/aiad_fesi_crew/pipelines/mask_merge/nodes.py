import os
from pathlib import Path
from typing import Iterable, Tuple, Dict
import cv2

def _is_image(name: str, exts: Iterable[str]) -> bool:
    return any(name.lower().endswith(e.lower()) for e in exts)

def apply_folder_masks(
    input_folder: str,
    mask_folder: str,
    output_folder: str,
    valid_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Dict[str, int]:
    """
    Apply grayscale masks (white=keep) to all images under input_folder,
    matching files by name inside the same category subfolder. Writes to output_folder.
    Returns summary stats for logging.
    """
    inp = Path(input_folder)
    msk = Path(mask_folder)
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    processed = skipped_nonimg = missing_mask = resized = read_err = 0

    for category in sorted([p for p in inp.iterdir() if p.is_dir()]):
        input_category_path = category
        mask_category_path = msk / category.name
        output_category_path = out / category.name

        if not mask_category_path.exists():
            # Skip categories that don't exist in masks
            continue

        output_category_path.mkdir(parents=True, exist_ok=True)

        for file in sorted(os.listdir(input_category_path)):
            if not _is_image(file, valid_exts):
                skipped_nonimg += 1
                continue

            img_path = input_category_path / file
            mask_path = mask_category_path / file

            if not mask_path.exists():
                missing_mask += 1
                continue

            image = cv2.imread(str(img_path))
            mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                read_err += 1
                continue

            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                resized += 1

            masked_image = cv2.bitwise_and(image, image, mask=mask)
            cv2.imwrite(str(output_category_path / file), masked_image)
            processed += 1

    return {
        "processed": processed,
        "missing_mask": missing_mask,
        "skipped_nonimg": skipped_nonimg,
        "resized_masks": resized,
        "read_errors": read_err,
        "output_dir": str(out),
    }
