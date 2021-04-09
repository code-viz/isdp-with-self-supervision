import torch
import torch.nn as nn
from torchvision import models

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        
        self.conv1      = resnet50.conv1
        self.bn1        = resnet50.bn1
        self.relu       = resnet50.relu
        self.maxpool    = resnet50.maxpool
        self.layer1     = resnet50.layer1
        self.layer2     = resnet50.layer2
        self.layer3     = resnet50.layer3
        self.layer4     = resnet50.layer4
        self.avgpool    = resnet50.avgpool
        self.__in_features = resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self.__in_features

class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        alexnet50       = models.alexnet(pretrained=True)
        self.features   = alexnet50.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), alexnet.classifier[i])
        self.__in_features = alexnet.classifier[6].in_features

        self.atten = Self_Attn(256, 'relu')

    def forward(self, x):
        x = self.features(self, x)
        x = self.atten(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
    def output_num(self):
        return self.__in_features

class Self_Attn(nn.Module):

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


network_dict = {"ResNet50Fc":ResNet50Fc, "AlexNetFc":AlexNetFc}