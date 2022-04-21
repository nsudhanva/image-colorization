import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorizeGAN_Generator(nn.Module):
    def __init__(self):
        super(ColorizeGAN_Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)

        self.deconv6 = nn.ConvTranspose2d(
            512, 512, 3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(
            512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(
            64, 3, 3, stride=2, padding=1, output_padding=1)
        self.bn10 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)  # 64,112,112
        pool1 = h

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)  # 128,56,56
        pool2 = h

        h = self.conv3(h)  # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)
        pool3 = h

        h = self.conv4(h)  # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)
        pool4 = h

        h = self.conv5(h)  # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.deconv6(h)
        h = self.bn6(h)
        h = self.relu6(h)  # 512,14,14
        h += pool4

        h = self.deconv7(h)
        h = self.bn7(h)
        h = self.relu7(h)  # 256,28,28
        h += pool3

        h = self.deconv8(h)
        h = self.bn8(h)
        h = self.relu8(h)  # 128,56,56
        h += pool2

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h)  # 64,112,112
        h += pool1

        h = self.deconv10(h)
        h = F.tanh(h)  # 3,224,224

        return h


class ColorizeGAN_Discriminator(nn.Module):
    def __init__(self):
        super(ColorizeGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)

        self.conv6 = nn.Conv2d(512, 512, 7, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1, 1, stride=1, padding=0)

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)  # 64,112,112

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)  # 128,56,56

        h = self.conv3(h)  # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h)  # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h)  # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h)

        h = self.conv7(h)
        h = F.sigmoid(h)

        return h
