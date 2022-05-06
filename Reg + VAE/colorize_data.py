import os
from typing import Tuple
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2gray
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class ColorizeData(Dataset):
    def __init__(self, landscape_dataset, data_directory):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale

        self.landscape_dataset = landscape_dataset
        self.data_directory = data_directory
        self.input_transform = T.Compose([T.Resize(size=(224, 224))])

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.landscape_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        image_path = os.path.join(
            self.data_directory, self.landscape_dataset[index])

        # Load image
        image = Image.open(image_path)
        image_original = self.input_transform(image)
        image_original = np.asarray(image_original)
        image_original = rgb2lab((image_original)+128)/255
        image_ab = image_original[:, :, 1:]
        image_ab = torch.from_numpy(image_ab.transpose((2, 0, 1))).float()
        image_original = rgb2gray(image_original)
        image_original = torch.from_numpy(image_original).unsqueeze(0)
        return image_original, image_ab