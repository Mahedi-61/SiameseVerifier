import torch
import torch.nn as nn 
from torchvision import models
from torchsummary import summary
# 1 x 96 x 96

class SiameseResentNetwork(nn.Module):
    def __init__(self, img_dim=1):
        super().__init__()
        self.cnn_block = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-2])
        self.cnn_block[0] = nn.Conv2d(img_dim, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)

        self.fc = nn.Sequential(
            nn.Linear(4608, 256)
        )

        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.cnn_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward1(self, in1, in2):
        out1 = self.forward_one_side(in1)
        out2 = self.forward_one_side(in2)

        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0.5, std=0.01)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.2)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0.0, std=0.01)


if __name__ == "__main__":
    net = SiameseResentNetwork()
    summary(net, (1, 96, 96), device="cpu")