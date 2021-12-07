import torch
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self,):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Linear(9216, 4096)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(4096, 1)

    def single_side(self, x):
        x = self.conv_block(x)
        x = self.sig(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x 

    def forward(self, x1, x2):
        one_side = self.single_side(x1)
        other_side = self.single_side(x2)
        dis = torch.abs(one_side - other_side)

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

# for test
if __name__ == "__main__":
    net = SiameseNetwork()
    x1 = torch.randn((16, 1, 100, 100))
    x2 = torch.randn((16, 1, 100, 100))
    #out = net(x1, x2)
    #print(out.size())