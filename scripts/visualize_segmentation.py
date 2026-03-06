import cv2
import numpy as np
from pathlib import Path
from utils.image_utils import visualize_overlay


# Set base directory for local or Colab
import os
base_dir = Path(os.getcwd())
data_dir = base_dir / "data"
processed_dir = data_dir / "processed"
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Paths
slice_path = processed_dir / "mid_slice.png"
mask_path = results_dir / "segmentation_mask.png"
output_path = results_dir / "segmentation_overlay.png"

# Load images
slice_img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

# Visualize and save overlay
visualize_overlay(slice_img, mask_img, output_path)
print("Segmentation visualization saved.")