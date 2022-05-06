# Comparing Regression, Autoencoder and Generative Adversarial Network (GAN) for Image Colorization

## CS 6140 - Machine Learning Project
- Instructor: Prof. Ehsan Elhamifar
- Contributors: Vikram Bharadwaj, Sudhanva Narayana

## Introduction
Traditional image colorization methods are based on manual colorization methods such as Photoshop or GIMP.
However, these methods are not robust to the changes in the lighting conditions of the scene.
Recent advances in deep learning have made it possible to learn a colorization model from grayscale images, where the input for the networks are a set of grayscale images and the output the network learns is the colorized image. 
In this work, we present three methods for colorization: regression, autoencoder and generative adversarial network (GAN).

- **Convolutional Neural Network + Regression:** Loss between actual values of *a and *b channels and predicted *a and *b channels using vanilla CNN, followed by regression.

- **Convnet Autoencoder:** Uses CNN in an encoder-decoder fashion, where the encoder represents the grayscale image as a latent representation, which the decoder then converts to a 3-channel color image.

- **GAN:**  Uses a custom minimax log-loss function with a generator and a discriminator. The generator takes a grayscale image as an input, which it tries to convert to a 3-channel color image. The discriminator tries to tell 2 sets of images apart, i.e., (grayscale, original-color) and (grayscale, generator-color-image).

## Usage
- Run the code in the `pwd` directory.
- To train the models regression and autoencoder, run the following command:
```
python execute_model.py <path to the training directory>
```
- To train the GAN, run the following command:
```
python execute_model.py <path to the training directory>
```
## How To Run
- To get the inference results, run the following command:
```
inference.ipynb
gan_inference.ipynb
```
## Contributions

**Sudhanva Narayana:**
- GAN architecture, code, tests and experiments.
- Trained the models on the dataset on AWS/Colab/GPU.
- Model inference and reporting.

**Vikram Bharadwaj:**
- Regression architecture, code, tests and experiments.
- Autoencoder architecture, code, tests and experiments.
- Model tuning and reporting.

## Report
- Please find the report here: `./reports/report.pdf`