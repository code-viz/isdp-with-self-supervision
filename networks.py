import torch
import torch.nn as nn
from torchvision import models

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, slef).__init__()
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
    
    def output_num(slef):
        return self.__in_features