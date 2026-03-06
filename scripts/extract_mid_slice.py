import nibabel as nib
import numpy as np
import cv2
from pathlib import Path

# MRI file path
mri_path = Path("../data/raw/sample.nii")

# Output file
output_path = Path("../data/processed/mid_slice.png")

print("Loading MRI...")

# Load MRI
img = nib.load(str(mri_path))
data = img.get_fdata()

print("MRI Shape:", data.shape)

# Find mid-sagittal slice
mid_index = data.shape[0] // 2
mid_slice = data[mid_index, :, :, 0]

# Normalize image
mid_slice = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice))
mid_slice = (mid_slice * 255).astype(np.uint8)

# Save image
cv2.imwrite(str(output_path), mid_slice)

print("Mid-sagittal slice saved.")