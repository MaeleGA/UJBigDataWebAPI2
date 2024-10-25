import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, Dataset
from osgeo import gdal  # For reading TIFF images
import numpy as np
import os

# Custom Dataset Class for TIFF images
class HyperspectralDataset(Dataset):
    def __init__(self, main_folder_path):
        self.image_files = []
        self.labels = []

        # Loop through all subfolders and assign labels based on folder names
        for subfolder_name in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for img_name in os.listdir(subfolder_path):
                    self.image_files.append(os.path.join(subfolder_path, img_name))
                    self.labels.append(self.get_label_from_folder(subfolder_name))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = self.read_tiff_image(img_name)

        # If the image is None, skip it (i.e., bands don't match)
        if image is None:
            return self.__getitem__((idx + 1) % len(self.image_files))

        label = self.labels[idx]
        return torch.Tensor(image), torch.LongTensor([label])

    def read_tiff_image(self, file_path, expected_bands=125):
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open {file_path}. Please check the file path and permissions.")

        image = dataset.ReadAsArray()  # Get the image data as a numpy array
        image = image.astype(np.float32)  # Convert to float32

        # Check the number of bands
        if image.shape[0] != expected_bands:
            print(f"Skipping {file_path}: Expected {expected_bands} bands, got {image.shape[0]}.")
            return None

        return image

    def get_label_from_folder(self, folder_name):
        # Map folder names to numeric labels
        if 'Health' in folder_name:
            return 0  # Healthy
        elif 'Rust' in folder_name:
            return 1  # Rust
        else:
            return 2  # Other