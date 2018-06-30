import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)


class SRMD(nn.Module):
    def __init__(self, num_blocks=11, num_channels=18, conv_dim=128, scale_factor=1):
        super(SRMD, self).__init__()
        self.num_channels = num_channels
        self.conv_dim = conv_dim
        self.sf = scale_factor

        self.nonlinear_mapping = self.make_layers(num_blocks)

        self.conv_last = nn.Sequential(
                            nn.Conv2d(self.conv_dim, 3*self.sf**2, kernel_size=3, padding=1),
                            nn.PixelShuffle(self.sf),
                            nn.Sigmoid()
                         )

    def forward(self, x):
        b_size = x.shape[0]
        h, w = list(x.shape[2:])
        x = self.nonlinear_mapping(x)
        x = self.conv_last(x)
        return x

    def make_layers(self, num_blocks):
        layers = []
        in_channels = self.num_channels
        for i in range(num_blocks):
            conv2d = nn.Conv2d(in_channels, self.conv_dim, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(self.conv_dim), nn.ReLU(inplace=True)]
            in_channels = self.conv_dim

        return nn.Sequential(*layers)
