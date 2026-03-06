import nibabel as nib
from pathlib import Path
from utils.image_utils import normalize_image, save_image


# Set base directory for local or Colab
import os
base_dir = Path(os.getcwd())
data_dir = base_dir / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
slices_dir = processed_dir / "slices"
slices_dir.mkdir(parents=True, exist_ok=True)

# Path to MRI file
mri_path = raw_dir / "sample.nii"

print("Loading MRI file...")
# Load MRI
img = nib.load(str(mri_path))
data = img.get_fdata()
print("MRI Shape:", data.shape)


# Loop through slices
for i in range(data.shape[2]):
    slice_img = data[:, :, i, 0]
    slice_img = normalize_image(slice_img)
    output_file = slices_dir / f"slice_{i}.png"
    save_image(slice_img, output_file)

print("All slices saved successfully.")
