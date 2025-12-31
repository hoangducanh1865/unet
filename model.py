import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_group):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups=groupnorm_num_group, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.groupnorm_2 = nn.GroupNorm(num_groups=groupnorm_num_group, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )
        self.residual_match = (
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
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


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=False):
        super().__init__()
        if interpolate:  # Use Upsample, usually be used in Diffusion
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),  # Unlearnable
                # In Diffusion, usually use mode="bilinear", align_corners=False or align_corners=None
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            )
        else:  # Use ConvTranspose2d, usually be used in old GAN
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )  # Learnable

    def forward(self, x):
        return self.upsample(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=150,
        start_dim=64,
        dim_mults=(1, 2, 4, 8),
        residual_blocks_per_group=3,
        groupnorm_num_group=16,
        interpolated_upsample=True,
    ):
        super().__init__()
        self.input_image_channels = in_channels
        self.interpolate = interpolated_upsample
        channel_sizes = [start_dim * i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]

        # Encoder config
        self.encoder_config = []
        for idx, d in enumerate(channel_sizes):
            for _ in range(residual_blocks_per_group):
                self.encoder_config.append(((d, d), "residual"))
            self.encoder_config.append(((d, d), "downsample"))
            if idx <= len(channel_sizes) - 2:
                self.encoder_config.append(((d, channel_sizes[idx + 1]), "residual"))

        """for item in self.encoder_config:
            print(item)"""

        # Residual blocks for each group
        self.bottleneck_config = []
        for _ in range(residual_blocks_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))

        out_dim = ending_channel_size

        # reversed_encoder_config = self.encoder_config[::-1]
        reversed_encoder_config = [
            self.encoder_config[i] for i in range(len(self.encoder_config) - 1, -1, -1)
        ]

        # Decoder config
        # Note: look at the visual image of U-Net to understand this block of code better
        self.decoder_config = []
        for idx, (metadata, type) in enumerate(reversed_encoder_config):
            enc_in_channels, enc_out_channels = metadata
            concat_num_channels = out_dim + enc_out_channels
            self.decoder_config.append(((concat_num_channels, enc_in_channels), "residual"))
            if type == "downsample":
                self.decoder_config.append(((enc_in_channels, enc_in_channels), "upsample"))
            out_dim = enc_in_channels

        # Special last block of decoder config
        concat_num_channels = starting_channel_size * 2
        self.decoder_config.append(((concat_num_channels, starting_channel_size), "residual"))

        # Model
        self.conv_in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=starting_channel_size,
            kernel_size=3,
            padding="same",
        )

        self.encoder = nn.ModuleList()
        for (in_channels, out_channels), type in self.encoder_config:
            if type == "residual":
                self.encoder.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groupnorm_num_group=groupnorm_num_group,
                    )
                )
            elif type == "downsample":
                self.encoder.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )

        self.bottleneck = nn.ModuleList()
        for (in_channels, out_channels), type in self.bottleneck_config:
            self.bottleneck.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groupnorm_num_group=groupnorm_num_group,
                )
            )

        self.decoder = nn.ModuleList()
        for (in_channels, out_channels), type in self.decoder_config:
            if type == "residual":
                self.decoder.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groupnorm_num_group=groupnorm_num_group,
                    )
                )
            elif type == "upsample":
                self.decoder.append(
                    UpsampleBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        interpolate=interpolated_upsample,
                    )
                )

        self.conv_out_proj = nn.Conv2d(
            in_channels=starting_channel_size, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        residuals = []
        x = self.conv_in_proj(x)
        residuals.append(x)

        for module in self.encoder:
            x = module(x)
            residuals.append(x)

        for module in self.bottleneck:
            x = module(x)

        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                respective_residual_tensor_from_encoder = (
                    residuals.pop()
                )  # Take the last element and then remove it from residuals
                x = torch.cat(
                    [x, respective_residual_tensor_from_encoder], dim=1
                )  # dim=1 means that we concate them in channel dimension
                x = module(x)

            else:  # module is an Upsample Layer
                x = module(x)

        x = self.conv_out_proj(x)

        print(x.shape)

        return x


def test():
    x = torch.randn(4, 3, 256, 256)
    print(x.shape)
    unet = Unet()
    print(unet(x))


if __name__ == "__main__":
    # test()
    pass
