import os
import sys

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import lab2rgb, rgb2gray
from torchvision import transforms

from basic_model import Net
from enc_dec import AE_conv

if __name__ == '__main__':
    # Read the two input parameters, which is the model checkpoint and grayscale image
    model_checkpoint, gray_image = sys.argv[1], sys.argv[2]
    # Load model from basic_model.py by calling the class constructor
    model = AE_conv()
    # If GPU available, set current execution to the CUDA instance, else, use CPU
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)

    # Open the grayscale image
    img = Image.open(gray_image)
    # Reshape it to the same size as the one that is accepted by the network
    image_l = img.resize((224, 224))
    # Although a grayscale image, it might have 3 channels, so check for number of channels. If three, convert to grayscale
    if len(image_l.getbands()) == 3:
        image_l = rgb2gray(image_l)
    # Use the transform to convert to a float tensor
    image_l = transforms.ToTensor()(image_l).float()

    # Evaluate the image from the model
    model.eval()

    with torch.no_grad():
        preds = model(image_l.unsqueeze(0).to(device))

    ab_output = preds[0].cpu()
    color_image = torch.cat((image_l, ab_output), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0] = color_image[:, :, 0] * 100
    color_image[:, :, 1:3] = (color_image[:, :, 1:3] - 128) * 255
    color_image = lab2rgb(color_image.astype(np.float64))

    plt.imsave('output_baseline.jpg', color_image)
    print("The output of the model has been stored in the \"outputs\" directory as output_baseline.jpg")
