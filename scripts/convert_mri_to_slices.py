import nibabel as nib
import numpy as np
import cv2
from pathlib import Path

# Path to MRI file
mri_path = Path("../data/raw/sample.nii")

# Output folder
output_folder = Path("../data/processed/slices")
output_folder.mkdir(parents=True, exist_ok=True)

print("Loading MRI file...")

# Load MRI
img = nib.load(str(mri_path))
data = img.get_fdata()

print("MRI Shape:", data.shape)

# Loop through slices
for i in range(data.shape[2]):
    slice_img = data[:, :, i, 0]
    
    # Normalize slice
    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
    slice_img = (slice_img * 255).astype(np.uint8)

    # Save slice
    output_file = output_folder / f"slice_{i}.png"
    cv2.imwrite(str(output_file), slice_img)

print("All slices saved successfully.")
