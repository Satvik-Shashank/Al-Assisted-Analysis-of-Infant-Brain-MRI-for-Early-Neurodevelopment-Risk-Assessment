import nibabel as nib
from pathlib import Path
from utils.image_utils import normalize_image, save_image


# Set base directory for local or Colab
import os
base_dir = Path(os.getcwd())
data_dir = base_dir / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

MRI_PATH = raw_dir / "sample.nii"
OUTPUT_PATH = processed_dir / "mid_slice.png"

print("Loading MRI...")
img = nib.load(str(MRI_PATH))
data = img.get_fdata()
print("MRI Shape:", data.shape)

# Find mid-sagittal slice
mid_index = data.shape[0] // 2
mid_slice = data[mid_index, :, :, 0]

# Normalize and save
mid_slice = normalize_image(mid_slice)
save_image(mid_slice, OUTPUT_PATH)

print("Mid-sagittal slice saved.")