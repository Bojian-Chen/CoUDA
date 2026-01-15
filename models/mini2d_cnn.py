import torch
import torch.nn as nn

class ActNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(ActNetwork, self).__init__()
        # self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=1)
        )
        self.classifier = nn.Linear(32 * 1, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        fea = x.view(x.size(0), -1)
        x = self.classifier(fea)

        return x, fea

model = ActNetwork()
input = torch.randn(1, 1, 32, 32)
output = model(input)   
print(output)
                # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                #                bias=False)