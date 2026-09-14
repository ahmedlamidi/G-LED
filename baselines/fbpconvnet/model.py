"""FBPConvNet (Jin et al., IEEE TIP 2017): a U-Net on the FBP image with the
input added back at the output, so the network learns the artifacts."""
import torch
import torch.nn as nn


def _double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FBPConvNet(nn.Module):
    def __init__(self, base=64, depth=4):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)]      # 64 ... 1024 in the paper
        self.enc = nn.ModuleList([_double_conv(1 if i == 0 else chans[i - 1], chans[i])
                                  for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(chans[i + 1], chans[i], 3, stride=2, padding=1, output_padding=1),
                          nn.BatchNorm2d(chans[i]), nn.ReLU(inplace=True))
            for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_double_conv(2 * chans[i], chans[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips, h = [], x
        for i, block in enumerate(self.enc):
            h = block(h)
            if i < len(self.enc) - 1:
                skips.append(h)
                h = self.pool(h)
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        return x + self.head(h)
