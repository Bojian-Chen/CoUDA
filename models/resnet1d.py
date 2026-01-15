import torch
import torch.nn as nn
import models.modified_linear as modified_linear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 需要进行降采样时，使用 1x1 卷积进行通道变换
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)   # 第一个卷积层
        out = self.bn1(out)   # 批归一化层
        out = self.relu(out)  # ReLU 激活函数

        out = self.conv2(out) # 第二个卷积层
        out = self.bn2(out)   # 批归一化层

        if self.downsample is not None:
            identity = self.downsample(x)  # 进行降采样

        out += identity  # 残差连接,这也是上面要进行下采样的原因
        out = self.relu(out)  # ReLU 激活函数

        return out

class ResNet1d(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet1d, self).__init__()

        self.in_channels = 16

        # 第一个卷积层
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)

        # 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 四个 ResNet 层
        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 128, num_blocks[3], stride=2)

        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(128*block.expansion , num_classes)
        # self.fc = modified_linear.CosineLinear(512 * block.expansion, num_classes)
        

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        # 创建第一个 BasicBlock，可能与其他 block 不同，所以单独处理
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels*block.expansion

        # 创建剩余的 BasicBlock
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # 第一个卷积层
        x = self.bn1(x)     # 批归一化层
        x = self.relu(x)    # ReLU 激活函数
        x = self.maxpool(x) # 最大池化层

        x = self.layer1(x)  # 第一个 ResNet 层
        x = self.layer2(x)  # 第二个 ResNet 层
        x = self.layer3(x)  # 第三个 ResNet 层
        x = self.layer4(x)  # 第四个 ResNet 层

        x = self.avgpool(x)             # 自适应平均池化层
        feature = x.view(x.size(0), -1)

        x = self.fc(feature)                  # 全连接层

        return x, feature
    

def resnet18(pretrained=False, **kwargs):
    n = 2
    model = ResNet1d(BasicBlock, [n, n, n, n], **kwargs)
    return model


# class FCNet(nn.Module):
#     def __init__(self, num_classes):
#         super(FCNet, self).__init__()
#         self.relu = nn.ReLU(inplace=True)

#         self.fc2 = nn.Linear(2048, 1024)
#         self.fc3 = nn.Linear(1024, 512)
#         self.fc4 = nn.Linear(512, 256)
#         self.fc5 = nn.Linear(256, 128)
#         self.fc6 = nn.Linear(128, 64)
#         self.fc7 = nn.Linear(64, num_classes)

#     def forward(self, x):
        
        
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.fc4(x)
#         x = self.relu(x)
#         x = self.fc5(x)
#         x = self.relu(x)
#         x = self.fc6(x)
#         feature = x.view(x.size(0), -1)
#         # print(feature)
#         x = self.fc7(feature)
#         return x, feature
    
# import torch.nn as nn
# import math
# import torch.utils.model_zoo as model_zoo
# # import models.modified_linear as modified_linear
# import torch


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.last = last

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         if not self.last: 
#             out = self.relu(out)

#         return out

# class ResNet(nn.Module):

#     def __init__(self, block, layers, num_classes=10):
#         self.inplanes = 16
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.layer1 = self._make_layer(block, 16, layers[0])
#         self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
#         self.avgpool = nn.AvgPool2d(8, stride=1)
#         # self.fc = modified_linear.CosineLinear(64 * block.expansion, num_classes)
#         # self.fc = modified_linear.EuclideanLinear(64 * block.expansion, num_classes)
 
#         self.fc = nn.Linear(64 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         if last_phase:
#             for i in range(1, blocks-1):
#                 layers.append(block(self.inplanes, planes))
#             layers.append(block(self.inplanes, planes, last=True))
#         else: 
#             for i in range(1, blocks):
#                 layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         x = self.avgpool(x)
#         feature = x.view(x.size(0), -1)
#         x = self.fc(feature)

#         return x, feature
    
# def resnet20(pretrained=False, **kwargs):
#     n = 3
#     model = ResNet(BasicBlock, [n, n, n], **kwargs)
#     return model

# def resnet32(pretrained=False, **kwargs):
#     n = 5
#     model = ResNet(BasicBlock, [n, n, n], **kwargs)
#     return model

# def resnet18(pretrained=False, **kwargs):
#     n = 2
#     model = ResNet(BasicBlock, [n, n, n], **kwargs)
#     return model
