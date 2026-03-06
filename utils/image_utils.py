import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity to 0–255 for visualization.
    """
    image = image - np.min(image)
    image = image / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)
    return image

def extract_mid_sagittal_slice(volume: np.ndarray) -> np.ndarray:
    """
    Extract the middle brain slice from a 3D MRI volume.
    """
    mid_index = volume.shape[0] // 2
    slice_img = volume[mid_index, :, :]
    return slice_img

def save_image(image: np.ndarray, output_path: Path):
    """
    Save image to disk using OpenCV.
    """
    cv2.imwrite(str(output_path), image)

def visualize_overlay(slice_img: np.ndarray, mask_img: np.ndarray, output_path: Path = None):
    """
    Overlay segmentation mask on MRI slice and optionally save.
    """
    mask_img = cv2.resize(mask_img, (slice_img.shape[1], slice_img.shape[0]))
    slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)
    overlay = slice_rgb.copy()
    overlay[mask_img > 0] = [255, 0, 0]  # Red color
    alpha = 0.4
    result = cv2.addWeighted(slice_rgb, 1 - alpha, overlay, alpha, 0)
    plt.imshow(result)
    plt.title("Segmentation Overlay")
    plt.axis("off")
    plt.show()
    if output_path:
        cv2.imwrite(str(output_path), result)
