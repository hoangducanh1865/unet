import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_group):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(groupnorm_num_group, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )
        self.groupnorm_2 = nn.GroupNorm(groupnorm_num_group, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.residual_match = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity
        )

    def forward(self, x):
        residual_connection = x
        x = self.groupnorm_1(x)
        x = F.relu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = x + self.residual_match(residual_connection)
        return x


if __name__ == "__main__":
    rand = torch.randn(4, 32, 256, 256)
    print(rand.shape)
    block = ResidualBlock(in_channels=32, out_channels=64, groupnorm_num_group=16)
    out_rand = block(rand)
    print(out_rand.shape)
