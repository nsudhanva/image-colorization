import os
import sys

import matplotlib.pyplot as plt
import torch

from train import Trainer


def plot_loss_psnr(training_loss, training_psnr, validation_loss, validation_psnr):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(training_loss, color='orange', label='train loss')
    plt.plot(validation_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./checkpoints/loss.png')
    plt.show()
    # psnr plots
    plt.figure(figsize=(10, 7))
    plt.plot(training_psnr, color='green', label='train PSNR dB')
    plt.plot(validation_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('./checkpoints/psnr.png')


def main(data_dir):

    # Initialize best loss and best psnr
    current_best_loss = 10000.0
    current_best_psnr = 0.0

    # Initialize training object from the Train class
    epochs = 100
    trainer = Trainer()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Define lists to keep track of training/validation loss and PSNR
    training_loss = []
    validation_loss = []
    training_psnr = []
    validation_psnr = []

    # Begin training
    print("Begininng training!")
    for epoch in range(0, epochs):

        # Train
        val_dataloader, train_loss, train_psnr = trainer.train(
            data_directory=data_dir, current_epoch=epoch)
        training_loss.append(train_loss)
        training_psnr.append(train_psnr)

        # Validate
        model, current_loss, current_psnr = trainer.validate(
            val_dataloader=val_dataloader, current_epoch=epoch)
        validation_loss.append(current_loss)
        validation_psnr.append(current_psnr)

        current_best_psnr = max(current_best_psnr, current_psnr)
        if current_loss < current_best_loss:
            current_best_loss = current_loss

            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')

            torch.save(model.state_dict(),
                       'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))

        print("-----------------------------------")
        print("Train & Validation complete for epoch - ", epoch)
        print("Final Validation Loss = ", current_best_loss,
              "\nFinal PSNR = ", current_best_psnr)
        print("===================================")

    print("Training and validation complete - current Loss = ", current_best_loss)
    print("Training and validation complete - current PSNR = ", current_best_psnr)
    plot_loss_psnr(training_loss,  training_psnr,
                   validation_loss, validation_psnr)

    return current_best_loss, epoch


if __name__ == '__main__':
    data_dir = sys.argv[1]
    print("Starting the execution!")
    final_loss, epoch = main(data_dir)
    print("Train and validation complete! -> Epochs = ",
          epoch, "\tFinal Loss = ", final_loss)
