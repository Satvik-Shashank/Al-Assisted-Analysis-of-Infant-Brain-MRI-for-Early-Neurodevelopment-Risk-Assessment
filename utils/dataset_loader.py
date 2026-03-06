import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class MRISegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading MRI slices and segmentation masks.
    Args:
        slices_dir (Path): Directory containing MRI slice images.
        masks_dir (Path): Directory containing segmentation mask images.
        transform (callable, optional): Optional transform to apply to both image and mask.
    """
    def __init__(self, slices_dir, masks_dir, transform=None):
        self.slices_dir = Path(slices_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.slice_files = sorted([f for f in self.slices_dir.glob('*.png')])
        self.mask_files = sorted([f for f in self.masks_dir.glob('*.png')])
        assert len(self.slice_files) == len(self.mask_files), "Mismatch between slices and masks"

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        slice_path = self.slice_files[idx]
        mask_path = self.mask_files[idx]
        image = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        image = torch.tensor(image).unsqueeze(0)  # Shape: [1, H, W]
        mask = torch.tensor(mask).unsqueeze(0)    # Shape: [1, H, W]
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
