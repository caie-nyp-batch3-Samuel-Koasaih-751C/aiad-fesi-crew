import os
import cv2
import numpy as np
from pathlib import Path

def is_black_background(image, threshold=30, sampling_width=20):
    top_edge = image[:sampling_width, :]
    bottom_edge = image[-sampling_width:, :]
    left_edge = image[:, :sampling_width]
    right_edge = image[:, -sampling_width:]
    sampled_region = np.concatenate((top_edge, bottom_edge, left_edge, right_edge), axis=None)
    mean_intensity = np.mean(sampled_region)
    return mean_intensity < threshold

def smooth_leaf_mask(binary_mask):
    kernel = np.ones((3, 3), np.uint8)
    smoothed_mask = cv2.erode(binary_mask, kernel, iterations=1)
    smoothed_mask = cv2.dilate(smoothed_mask, kernel, iterations=1)
    return smoothed_mask

def refine_mask_with_smoothing(binary_mask, min_leaf_area=500):
    smoothed_mask = smooth_leaf_mask(binary_mask)
    closing_kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, closing_kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    refined_mask = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_leaf_area:
            refined_mask[labels == label] = 255
    dilation_kernel = np.ones((8, 8), np.uint8)
    return cv2.dilate(refined_mask, dilation_kernel, iterations=1)

def preprocess_images(input_dir: str, output_dir: str):
    """
    Processes all images in `input_dir` and writes masks to `output_dir`.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                rel_path = Path(root).relative_to(input_path)
                out_subfolder = output_path / rel_path
                out_subfolder.mkdir(parents=True, exist_ok=True)
                out_file_path = out_subfolder / file

                image = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

                if is_black_background(gray_image):
                    _, binary_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
                else:
                    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                    green_mask = cv2.inRange(image_hsv, np.array([25, 25, 25]), np.array([90, 255, 255]))
                    brown_mask = cv2.inRange(image_hsv, np.array([8, 30, 18]), np.array([34, 220, 190]))
                    blue_mask = cv2.inRange(image_hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))
                    yellow_mask = cv2.inRange(image_hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))
                    binary_mask = cv2.bitwise_or(
                        cv2.bitwise_or(green_mask, brown_mask),
                        cv2.bitwise_or(blue_mask, yellow_mask)
                    )

                smoothed_mask = refine_mask_with_smoothing(binary_mask)

                if np.sum(smoothed_mask) == 0:
                    _, fallback_mask = cv2.threshold(gray_image, 5, 255, cv2.THRESH_BINARY_INV)
                    kernel = np.ones((5, 5), np.uint8)
                    fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, kernel)
                    fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, kernel)
                    smoothed_mask = fallback_mask

                dilation_kernel = np.ones((8, 8), np.uint8)
                smoothed_mask = cv2.dilate(smoothed_mask, dilation_kernel, iterations=1)

                cv2.imwrite(str(out_file_path), smoothed_mask)

    return str(output_path)
