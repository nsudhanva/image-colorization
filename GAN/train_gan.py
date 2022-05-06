import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from gan import ColorizeGAN_Generator, ColorizeGAN_Discriminator
from dataLoader_gan import ColorizeGANDataloader

class GAN_trainer():
    def __init__(self):
        # Define hparams here or load them from a config file
        self.batch_size = 64
        self.learning_rate = 0.001
        self.device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        self.model_discriminator = ColorizeGAN_Discriminator()
        self.model_generator = ColorizeGAN_Generator()
        self.generator_criterion = nn.BCELoss()
        self.discriminator_criterion = nn.BCELoss()
        self.L1 = nn.L1Loss()

    def train(self, data_directory, current_epoch):
        dataset = os.listdir(data_directory)
        train_size = int(0.8*len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        # dataloaders
        # Train dataloader
        train_dataset = ColorizeGANDataloader(train_dataset, data_directory)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True)

        # Validation dataloader
        val_dataset = ColorizeGANDataloader(val_dataset, data_directory)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False)

        # Model and the Loss function to use
        # You may also use a combination of more than one loss function
        # or create your own.

        if self.device == 'CUDA':
            self.model_generator = self.model_generator.cuda()
            self.model_discriminator = self.model_discriminator.cuda()
            self.generator_criterion = self.generator_criterion.cuda()
            self.discriminator_criterion = self.discriminator_criterion.cuda()

        generator_optimizer = torch.optim.Adam(self.model_generator.parameters(), lr=self.learning_rate)
        discriminator_optimizer = torch.optim.Adam(self.model_discriminator.parameters(), lr=self.learning_rate)

        # Defind real and fake labels
        real_label = 1
        fake_label = 0

        # train loop
        # Define Running Loss to keep track later in the train loop
        running_loss = 0.0
        for i, (input, target) in enumerate(train_dataloader):
            # print("Size of input = ", input.size())
            # print("Size of target = ", target.size())
            if self.device == 'CUDA':
                input = Variable(input.cuda())
                target = Variable(target.cuda())

            ## Train the discriminator on the real images
            discriminator_optimizer.zero_grad()
            real_output = self.model_discriminator(target)
            d_label = Variable(torch.FloatTensor(target.size(0)).fill_(real_label).cuda())
            real_output = real_output.squeeze()
            real_output = real_output.unsqueeze(1)
            d_label = d_label.unsqueeze(1)
            real_loss = self.discriminator_criterion(real_output, d_label)
            # real_loss.backward()
            
            ## Train the discriminator on the fake images
            fake = self.model_generator(input)
            fake_output = self.model_discriminator(fake.detach())
            d_label = Variable(torch.FloatTensor(target.size(0)).fill_(fake_label).cuda())
            fake_output = fake_output.squeeze()
            fake_output = fake_output.unsqueeze(1)
            d_label = d_label.unsqueeze(1)
            fake_loss = self.discriminator_criterion(fake_output, d_label)
            total_discriminator_loss = (real_loss + fake_loss) * 0.5
            total_discriminator_loss.backward()
            
            # Update the discriminator
            discriminator_optimizer.step()
            
            ## Train the generator
            generator_optimizer.zero_grad()
            g_label = Variable(torch.FloatTensor(target.size(0)).fill_(real_label).cuda())
            g_output = self.model_discriminator(fake)
            g_output = g_output.squeeze()
            g_output = g_output.unsqueeze(1)
            g_label = g_label.unsqueeze(1)
            g_loss = self.generator_criterion(g_output, g_label)
            g_l1 = self.L1(fake.view(fake.size(0),-1), target.view(target.size(0),-1))
            total_generator_loss = g_loss + 0.1*g_l1
            total_generator_loss.backward()
            generator_optimizer.step()

            # Print the loss
            running_loss += total_generator_loss.item() + total_discriminator_loss.item()
            loss = running_loss / len(train_dataloader)
            if i % 500 == 0:
                print("Current Epoch = ", current_epoch,
                      "\nCurrent loss = ", loss)

        final_loss = running_loss / len(train_dataloader)

        return val_dataloader, final_loss

    def validate(self, val_dataloader, current_epoch):
        # Validation loop begin
        # ------
        running_loss = 0.0
        real_label = 1
        fake_label = 0
        self.model_generator.eval()
        self.model_discriminator.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                if self.device == 'CUDA':
                    input = Variable(input.cuda())
                    target = Variable(target.cuda())

                # Validate the discriminator on the real images
                output = self.model_discriminator(target)
                d_label = Variable(torch.FloatTensor(target.size(0)).fill_(real_label).cuda())
                output = output.squeeze()
                output = output.unsqueeze(1)
                d_label = d_label.unsqueeze(1)
                real_error = self.discriminator_criterion(output, d_label)
                
                # Validate the discriminator on the fake images
                fake = self.model_generator(input)
                g_label = Variable(torch.FloatTensor(target.size(0)).fill_(fake_label).cuda())
                g_output = self.model_discriminator(fake.detach())
                g_output = g_output.squeeze()
                g_output = g_output.unsqueeze(1)
                g_label = g_label.unsqueeze(1)
                fake_error = self.discriminator_criterion(g_output, g_label)
                discriminator_error = (real_error + fake_error) * 0.5

                # Validate the generator
                g_label = Variable(torch.FloatTensor(target.size(0)).fill_(real_label).cuda())
                g_output = self.model_discriminator(fake)

                # Calculate the loss
                g_output = g_output.squeeze()
                g_output = g_output.unsqueeze(1)
                g_label = g_label.unsqueeze(1)
                generatore_error = self.generator_criterion(g_output, g_label)
                generator_loss = generatore_error + 0.1*self.L1(fake.view(fake.size(0),-1), target.view(target.size(0),-1))

                # Print the loss
                running_loss += generator_loss.item() + discriminator_error.item()
                if i % 100 == 0:
                    print("Current Epoch = ", current_epoch,
                          "\nCurrent loss = ", running_loss)

        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        final_loss = running_loss / len(val_dataloader)

        return self.model_generator, final_loss