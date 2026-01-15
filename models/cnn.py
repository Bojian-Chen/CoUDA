import torch.nn as nn

class feature(nn.Module):
    def __init__(self):
        super(feature, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
    def forward(self, input_data):
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 5 * 5)
        return feature


class classfier(nn.Module):
    def __init__(self, num_classes = 10):
        super(classfier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 5 * 5, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, num_classes))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))
    def forward(self, feature):
        class_output = self.class_classifier(feature)
        return class_output

class CNN(nn.Module):
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.feature = feature()
        self.classifier = classfier(num_classes)

    def forward(self, x):
        feature = self.feature(x)
        x = self.classifier(feature)

        return x, feature
    

