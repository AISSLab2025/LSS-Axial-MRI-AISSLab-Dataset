import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        img_size: int | tuple[int, int] = 224,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        # CNN Encoder: ResNet34
        resnet = models.resnet34(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels, H/4, W/4
        self.layer2 = resnet.layer2  # 128 channels, H/8, W/8
        self.layer3 = resnet.layer3  # 256 channels, H/16, W/16
        self.layer4 = resnet.layer4  # 512 channels, H/32, W/32

        # Transformer bottleneck
        grid_size = (self.img_size[0] // 32, self.img_size[1] // 32)
        num_patches = grid_size[0] * grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder Blocks (Upsampling with skip connections)
        self.up1 = DecoderBlock(512, 256, 256)  # input 512, skip 256, output 256
        self.up2 = DecoderBlock(256, 128, 128)  # input 256, skip 128, output 128
        self.up3 = DecoderBlock(128, 64, 64)    # input 128, skip 64, output 64
        self.up4 = DecoderBlock(64, 64, 64)     # input 64, skip 64 (from layer1/conv1), output 64

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet Backbone features extraction
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64 channels, H/2, W/2
        x_pool = self.maxpool(x0)                # 64 channels, H/4, W/4

        x1 = self.layer1(x_pool)  # 64 channels, H/4, W/4
        x2 = self.layer2(x1)      # 128 channels, H/8, W/8
        x3 = self.layer3(x2)      # 256 channels, H/16, W/16
        x4 = self.layer4(x3)      # 512 channels, H/32, W/32

        # Flatten features into sequence for Transformer
        B, C, h_grid, w_grid = x4.shape
        x_flat = x4.flatten(2).transpose(1, 2)  # (B, L, 512)
        x_flat = x_flat + self.pos_embed[:, : x_flat.shape[1]]

        trans_out = self.transformer(x_flat)    # (B, L, 512)
        trans_out = trans_out.transpose(1, 2).reshape(B, C, h_grid, w_grid)

        # Decode features merging skip connections
        y1 = self.up1(trans_out, x3)  # H/16, W/16, 256 channels
        y2 = self.up2(y1, x2)         # H/8, W/8, 128 channels
        y3 = self.up3(y2, x1)         # H/4, W/4, 64 channels
        y4 = self.up4(y3, x0)         # H/2, W/2, 64 channels

        y_final = self.final_up(y4)   # H, W, 32 channels
        out = self.final_conv(y_final)
        return out
