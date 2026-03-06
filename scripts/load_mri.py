import os
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_mri(file_path: Path) -> np.ndarray:
    """
    Load MRI scan from NIfTI format and return 3D brain volume.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"MRI file not found: {file_path}")

    img = nib.load(str(file_path))
    data = img.get_fdata()

    return data


def extract_mid_sagittal_slice(volume: np.ndarray) -> np.ndarray:
    """
    Extract the middle brain slice from a 3D MRI volume.
    """

    mid_index = volume.shape[0] // 2
    slice_img = volume[mid_index, :, :]

    return slice_img


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity to 0–255 for visualization.
    """

    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)

    return image


def save_image(image: np.ndarray, output_path: Path):
    """
    Save processed image to disk.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def display_image(image: np.ndarray):
    """
    Display image using matplotlib.
    """

    plt.imshow(image, cmap="gray")
    plt.title("Mid-Sagittal Brain Slice")
    plt.axis("off")
    plt.show()


def main():

    # Define paths
    mri_path = Path("../data/raw/sample.nii.gz")
    output_path = Path("../data/processed/mid_slice.png")

    print("\nLoading MRI scan...")

    # Load MRI
    volume = load_mri(mri_path)

    print("MRI loaded successfully")
    print("----------------------------------")
    print(f"MRI Shape: {volume.shape}")
    print(f"Data Type: {volume.dtype}")
    print("----------------------------------")

    # Extract slice
    mid_slice = extract_mid_sagittal_slice(volume)

    # Normalize image
    processed_slice = normalize_image(mid_slice)

    # Save slice
    save_image(processed_slice.T, output_path)

    print(f"Mid-sagittal slice saved at: {output_path}")

    # Display slice
    display_image(processed_slice.T)


if __name__ == "__main__":
    main()