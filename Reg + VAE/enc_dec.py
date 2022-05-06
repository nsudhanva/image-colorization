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
        return torch.tanh(logits)

# class Auto_Colorizer(nn.Module):
    
#     '''The decoder is a series of convolutional layers with upsampling.
        
#         Model accepts the Lightness channel from LAB iamges.
        
#         Generatates A&B color channels in the LAB colorspace'''
        
#     def __init__(self):
        
#         '''Encoder'''
        
#         super(Auto_Colorizer, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size = 7, stride=2, padding = 3),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.Conv2d(64,64, kernel_size = 3, stride = 1, padding = 1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
#             nn.Conv2d(64, 128, kernel_size = 3, stride=1, padding = 1),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.Conv2d(128,128, kernel_size = 3, stride = 1, padding = 1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(128)
            
#         )
        
#         '''Decoder, convolutions with upsampling'''

#         self.decoder = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
#             nn.Upsample(scale_factor=2)
#         )

#     def forward(self, input):
#         features = self.encoder(input)
#         out = self.decoder(features)
#         return out