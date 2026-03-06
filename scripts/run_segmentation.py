import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import cv2
import numpy as np
from pathlib import Path

from models.unet_segmentation import UNet


# Path to processed MRI slice
image_path = Path("../data/processed/mid_slice.png")

# Load image
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Processed MRI slice not found.")

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
output_path = Path("../data/processed/segmentation_mask.png")

cv2.imwrite(str(output_path), mask)

print("Segmentation mask saved.")