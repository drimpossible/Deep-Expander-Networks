import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNetExpander1', 'alexnetexpander1']


class AlexNetExpander1(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.05),
            ExpanderLinear(256 * 6 * 6, 4096, expandSize=256),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.05),
            ExpanderLinear(4096, 4096, expandSize=256),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            ExpanderLinear(4096, num_classes, expandSize=512),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
