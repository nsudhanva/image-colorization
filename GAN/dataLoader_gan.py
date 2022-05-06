import os
from typing import Tuple
import cv2
from skimage.color import rgb2lab
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
        
        img = cv2.imread(image_path)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        l = img_lab[:, :, 0] / 50. - 1.  # Between -1 and 1
        ab = img_lab[:, :, 1:3] / 110.  # Between -1 and 1
        l = self.input_transform(l)
        ab = self.target_transform(ab)

        return(l, ab)