import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18
import torch.nn.init as init
from timm import create_model
import torch.nn.functional as F
from torch.hub import load

class Resnet(nn.Module):
    def __init__(self, emb, pretrained=True, is_norm=True, bn_freeze = True, type = 'resnet50'):
        super(Resnet, self).__init__()
        self.is_norm = is_norm
        self.emb = emb


        if type == 'resnet50':
            self.model = resnet50(pretrained)
        
        elif type == 'resnet18':
            self.model = resnet18(pretrained)
        self.features = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.embedding = nn.Linear(self.features, self.emb)
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        if self.is_norm:
            x = self.l2_norm(x)
        
        return x
