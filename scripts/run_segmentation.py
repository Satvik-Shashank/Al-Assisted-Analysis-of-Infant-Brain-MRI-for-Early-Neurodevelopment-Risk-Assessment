
import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import cv2
import numpy as np

from models.unet_segmentation import UNet

# Set base directory for local or Colab
base_dir = Path(os.getcwd())
data_dir = base_dir / "data"
processed_dir = data_dir / "processed"
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Path to processed MRI slice
image_path = processed_dir / "mid_slice.png"

# Load image
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Processed MRI slice not found: {image_path}")

# Resize for model
image = cv2.resize(image, (256, 256))

# Normalize
image = image / 255.0

# Convert to tensor
image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)

print("Image loaded and prepared.")


# Load model
model = UNet()
model.eval()

print("Running segmentation model...")


# Run prediction
with torch.no_grad():
    output = model(image)


# Convert output to image
mask = output.squeeze().numpy()

mask = (mask * 255).astype(np.uint8)


# Save mask
output_path = results_dir / "segmentation_mask.png"
cv2.imwrite(str(output_path), mask)
print(f"Segmentation mask saved to results folder: {output_path}")