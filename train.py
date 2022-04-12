import math
import os

import numpy as np
import torch
import torch.nn as nn

from colorize_data import ColorizeData
from enc_dec import AE_conv


class Trainer:
    def __init__(self):
        # Define hparams here or load them from a config file
        self.batch_size = 64
        self.learning_rate = 0.001
        self.device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        self.criterion = nn.MSELoss()
        self.model = AE_conv()

    # Writing a function to compute the PSNR performance metric that measures similarity between the output from the model and the target.
    def psnr(self, target, output, max_val=1.):
        """
        Compute Peak Signal to Noise Ratio (the higher the better).
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
        """
        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        difference_image = output - target
        rmse = math.sqrt(np.mean((difference_image) ** 2))
        if rmse == 0:
            return 100
        else:
            psnr = 20 * math.log10(max_val / rmse)
            return psnr

    def train(self, data_directory, current_epoch):
        print("Loading dataset --")
        dataset = os.listdir(data_directory)
        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        # dataloaders
        # Train dataloader
        train_dataset = ColorizeData(train_dataset, data_directory)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True)

        # Validation dataloader
        val_dataset = ColorizeData(val_dataset, data_directory)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False)

        # Model and the Loss function to use
        # You may also use a combination of more than one loss function
        # or create your own.

        if self.device == 'CUDA':
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.learning_rate)

        # train loop
        # Define Running Loss to keep track later in the train loop
        running_loss = 0.0
        running_psnr = 0.0

        # Print loaded model
        # print("Model loaded ->", self.model)

        for i, (input, target) in enumerate(train_dataloader):
            if self.device == 'CUDA':
                input = input.cuda()
                target = target.cuda()

            output_image = self.model(input)
            loss = self.criterion(output_image, target)

            # Set the gradient of tensors to zero.
            optimizer.zero_grad()
            # Compute the gradient of the current tensor by employing chain rule and propogating it back in the network.
            loss.backward()
            # Update the parameters in the direction of the gradient.
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 0:
                print("Current Epoch = ", current_epoch,
                      "\nCurrent loss = ", loss)

        final_loss = running_loss / len(train_dataloader)
        final_psnr = running_psnr / len(train_dataloader)

        return val_dataloader, final_loss, final_psnr

    def validate(self, val_dataloader, current_epoch):
        # Validation loop begin
        # ------
        running_loss = 0.0
        running_psnr = 0.0
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                if self.device == 'CUDA':
                    input = input.cuda()
                    target = target.cuda()

                output_image = self.model(input)
                loss = self.criterion(output_image, target)

                running_loss += loss.item()
                batch_psnr = self.psnr(target, output_image)
                running_psnr += batch_psnr

                if i % 50 == 0:
                    print("Current Epoch = ", current_epoch,
                          "\nCurrent loss = ", loss)

        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        final_loss = running_loss / len(val_dataloader)
        final_psnr = running_psnr / len(val_dataloader)

        return self.model, final_loss, final_psnr
