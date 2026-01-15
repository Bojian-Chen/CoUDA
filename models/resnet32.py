import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
# import models.modified_linear as modified_linear
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if not self.last: 
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = modified_linear.CosineLinear(64 * block.expansion, num_classes)
        # self.fc = modified_linear.EuclideanLinear(64 * block.expansion, num_classes)
 
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else: 
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = self.fc(feature)

        return x, feature
    
def resnet20(pretrained=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

def resnet14(pretrained=False, **kwargs):
    n = 2
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

class LearnableFNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)

    def forward(self, x, epoch):
        
        Freq = torch.fft.fft(torch.fft.fft(x.permute(0,2,1), dim=-1), dim=-2)

        b, patches, c = Freq.shape
        print(b, patches, c)
        D_0 = patches // 2 + (patches // 8 - patches // 2) * (epoch / 120)
        lowpass_filter_l = torch.exp(-0.5 * torch.square(torch.linspace(0, patches // 2 - 1, patches // 2).unsqueeze(1).repeat(1,c).cuda() / (D_0))).view(1, patches // 2, c).cuda()
        lowpass_filter_r = torch.flip(torch.exp(-0.5 * torch.square(torch.linspace(1, patches // 2 , patches // 2).unsqueeze(1).repeat(1,c).cuda() / (D_0))).view(1, patches // 2, c).cuda(), [1])
        lowpass_filter = torch.concat((lowpass_filter_l, lowpass_filter_r), dim=1)
        
        low_Freq = Freq * lowpass_filter
        lowFreq_feature = torch.fft.ifft(torch.fft.ifft(low_Freq, dim=-2), dim=-1).real

        weights = 0.5 * torch.sigmoid(self.projection(x).permute(0,2,1).mean(dim=1)).unsqueeze(dim=1) + 0.5
        out = weights * lowFreq_feature + (1 - weights) * (x.permute(0,2,1) - lowFreq_feature)

        return out.permute(0,2,1)
# model = resnet32(num_classes=10)


def resnet_mini(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1], **kwargs)  #8layers
    return model

# model = resnet14()
# import torchinfo
# torchinfo.summary(model, (2050, 1, 32, 32))
# torch.cuda.empty_cache()
# from thop import profile
# input = torch.randn(1000, 1, 32, 32)  
# macs, params = profile(model, inputs=(input, ))
# print(f"FLOPs: {macs*2 / 1e6} M") 
# print(f"Params: {params / 1e6} M")


# Simple Resnet50 example to demonstrate how to capture memory visuals.
