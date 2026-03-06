import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
slice_path = Path("../data/processed/mid_slice.png")
mask_path = Path("../data/processed/segmentation_mask.png")

# Load images
slice_img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

# Resize mask to match MRI slice size
mask_img = cv2.resize(mask_img, (slice_img.shape[1], slice_img.shape[0]))
# Convert MRI to RGB
slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)

# Create red overlay
overlay = slice_rgb.copy()
overlay[mask_img > 0] = [255, 0, 0]  # Red color

# Blend images
alpha = 0.4
result = cv2.addWeighted(slice_rgb, 1 - alpha, overlay, alpha, 0)

# Display
plt.imshow(result)
plt.title("Segmentation Overlay")
plt.axis("off")
plt.show()

# Save result
output_path = Path("../data/processed/segmentation_overlay.png")
cv2.imwrite(str(output_path), result)

print("Segmentation visualization saved.")