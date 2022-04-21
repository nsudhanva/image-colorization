import torch
import torch.nn as nn


class AE_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True)
        )

        self.linear_encoder = nn.Linear(256*14*14, 256)
        self.linear_decoder = nn.Linear(256, 256*14*14)

        self.decoder = nn.Sequential(
            # note that here, we have the same number of output channels
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=(
                3, 3), stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # print("Size of input: ", x.size())
        x = self.encoder(x)
        # print("Size of encoded: ", x.size())
        x = x.view(x.size(0), -1)
        # print("Size of flattened: ", x.size())
        x = self.linear_encoder(x)
        x = self.linear_decoder(x)
        x = x.view(-1, 256, 14, 14)
        logits = self.decoder(x)
        return nn.sigmoid(logits)
