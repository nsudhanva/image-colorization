import os
from typing import Tuple

import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class ColorizeGANDataloader(Dataset):
    def __init__(self, landscape_dataset, data_directory):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.landscape_dataset = landscape_dataset
        self.data_directory = data_directory
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(224, 224))])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(224, 224))])

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.landscape_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        image_path = os.path.join(
            self.data_directory, self.landscape_dataset[index])

        color_image = cv2.imread(image_path)
        l_channel, _, _ = cv2.split(color_image)

        l_channel = l_channel / 100
        l_channel = self.input_transform(l_channel)
        l_channel = l_channel.type(torch.FloatTensor)
        color_image = self.target_transform(color_image)
        color_image = color_image.type(torch.FloatTensor)
        return (l_channel, color_image)
