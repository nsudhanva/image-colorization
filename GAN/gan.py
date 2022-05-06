# import torch
# import torch.nn as nn

# class ColorizeGAN_Generator(nn.Module):
#     def __init__(self):
#         super(ColorizeGAN_Generator, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(True)

#         self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.relu2 = nn.ReLU(True)

#         self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.relu3 = nn.ReLU(True)

#         self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.relu4 = nn.ReLU(True)

#         self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.relu5 = nn.ReLU(True)

#         self.deconv6 = nn.ConvTranspose2d(
#             512, 512, 3, stride=2, padding=1, output_padding=1)
#         self.bn6 = nn.BatchNorm2d(512)
#         self.relu6 = nn.ReLU()

#         self.deconv7 = nn.ConvTranspose2d(
#             512, 256, 3, stride=2, padding=1, output_padding=1)
#         self.bn7 = nn.BatchNorm2d(256)
#         self.relu7 = nn.ReLU()

#         self.deconv8 = nn.ConvTranspose2d(
#             256, 128, 3, stride=2, padding=1, output_padding=1)
#         self.bn8 = nn.BatchNorm2d(128)
#         self.relu8 = nn.ReLU()

#         self.deconv9 = nn.ConvTranspose2d(
#             128, 64, 3, stride=2, padding=1, output_padding=1)
#         self.bn9 = nn.BatchNorm2d(64)
#         self.relu9 = nn.ReLU()

#         self.deconv10 = nn.ConvTranspose2d(
#             64, 2, 3, stride=2, padding=1, output_padding=1)
#         self.bn10 = nn.BatchNorm2d(3)
#         self.relu10 = nn.ReLU()

#     def forward(self, x):
#         h = x
#         h = self.conv1(h)
#         h = self.bn1(h)
#         h = self.relu1(h)  # 64,112,112
#         pool1 = h

#         h = self.conv2(h)
#         h = self.bn2(h)
#         h = self.relu2(h)  # 128,56,56
#         pool2 = h

#         h = self.conv3(h)  # 256,28,28
#         h = self.bn3(h)
#         h = self.relu3(h)
#         pool3 = h

#         h = self.conv4(h)  # 512,14,14
#         h = self.bn4(h)
#         h = self.relu4(h)
#         pool4 = h

#         h = self.conv5(h)  # 512,7,7
#         h = self.bn5(h)
#         h = self.relu5(h)

#         h = self.deconv6(h)
#         h = self.bn6(h)
#         h = self.relu6(h)  # 512,14,14
#         h += pool4

#         h = self.deconv7(h)
#         h = self.bn7(h)
#         h = self.relu7(h)  # 256,28,28
#         h += pool3

#         h = self.deconv8(h)
#         h = self.bn8(h)
#         h = self.relu8(h)  # 128,56,56
#         h += pool2

#         h = self.deconv9(h)
#         h = self.bn9(h)
#         h = self.relu9(h)  # 64,112,112
#         h += pool1

#         h = self.deconv10(h)
#         h = torch.tanh(h)  # 3,224,224

#         return h


# class ColorizeGAN_Discriminator(nn.Module):
#     def __init__(self):
#         super(ColorizeGAN_Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(2, 64, 3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(True)

#         self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.relu2 = nn.ReLU(True)

#         self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.relu3 = nn.ReLU(True)

#         self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.relu4 = nn.ReLU(True)

#         self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.relu5 = nn.ReLU(True)

#         self.conv6 = nn.Conv2d(512, 512, 7, stride=1, padding=0)
#         self.bn6 = nn.BatchNorm2d(512)
#         self.relu6 = nn.ReLU(True)

#         self.conv7 = nn.Conv2d(512, 1, 1, stride=1, padding=0)

#     def forward(self, x):
#         h = x
#         h = self.conv1(h)
#         h = self.bn1(h)
#         h = self.relu1(h)  # 64,112,112

#         h = self.conv2(h)
#         h = self.bn2(h)
#         h = self.relu2(h)  # 128,56,56

#         h = self.conv3(h)  # 256,28,28
#         h = self.bn3(h)
#         h = self.relu3(h)

#         h = self.conv4(h)  # 512,14,14
#         h = self.bn4(h)
#         h = self.relu4(h)

#         h = self.conv5(h)  # 512,7,7
#         h = self.bn5(h)
#         h = self.relu5(h)

#         h = self.conv6(h)
#         h = self.bn6(h)
#         h = self.relu6(h)

#         h = self.conv7(h)
#         h = torch.sigmoid(h)

#         return h

from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth'))
	return model

class color_ecv(nn.Module):
    def __init__(self, in_channels=3):
        super(color_ecv, self).__init__()
        
        self.model = eccv16(pretrained=True)
    
    def forward(self, x):
        ecv_output = self.model(x)
        return ecv_output