import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from torchvision import transforms

from basic_model import Net


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


if __name__ == '__main__':
    # Read the two input parameters, which is the model checkpoint and grayscale image
    model_checkpoint, gray_image = sys.argv[1], sys.argv[2]
    # Load model from basic_model.py by calling the class constructor
    model = Net()
    # If GPU available, set current execution to the CUDA instance, else, use CPU
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint, strict=False)

    # Open the grayscale image
    img = Image.open(gray_image)
    # Reshape it to the same size as the one that is accepted by the network
    image_l = img.resize((256, 256))
    # Although a grayscale image, it might have 3 channels, so check for number of channels. If three, convert to grayscale
    if len(image_l.getbands()) == 3:
        image_l = rgb2gray(image_l)
    # Use the transform to convert to a float tensor
    image_l = transforms.ToTensor()(image_l).float()

    # Evaluate the image from the model
    model.eval()

    with torch.no_grad():
        preds = model(image_l.unsqueeze(0).to(device))

    # output_tensor = inverse_normalize(tensor=preds, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    color_image = preds[0].cpu().numpy()
    color_image = color_image.transpose((1, 2, 0))
    min_val = np.min(color_image)
    max_val = np.max(color_image)
    color_image = (color_image - min_val) / (max_val - min_val)

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    plt.imsave('output_baseline.jpg', color_image)

    print("The output of the model has been stored in the \"outputs\" directory as output_baseline.jpg")
