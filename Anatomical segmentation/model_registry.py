import torch
import torch.nn as nn
import torchvision.models.segmentation as torchvision_seg
import monai.networks.nets as monai_nets
import segmentation_models_pytorch as smp

from models.trans_unet import TransUNet2D


def get_model(name: str, in_channels: int, out_channels: int, img_size: tuple[int, int]) -> nn.Module:
    name_lower = name.lower().replace("_", "").replace("-", "")

    if name_lower == "deeplabv3":
        model = torchvision_seg.deeplabv3_resnet50(weights=None, num_classes=out_channels)
        if in_channels != 3:
            model.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        class DeepLabV3Wrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
            def forward(self, x):
                return self.base(x)["out"]
        return DeepLabV3Wrapper(model)

    elif name_lower == "attentionunet":
        return monai_nets.AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    elif name_lower == "unetr":
        return monai_nets.UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            spatial_dims=2,
        )

    elif name_lower == "swinunetr":
        return monai_nets.SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            spatial_dims=2,
        )

    elif name_lower == "swintransformer":
        return smp.Unet(
            encoder_name="tu-swin_tiny_patch4_window7_224",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
        )

    elif name_lower == "segresnet":
        return monai_nets.SegResNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=8,
        )

    elif name_lower == "transunet":
        return TransUNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
        )

    else:
        raise ValueError(f"Unknown model name: '{name}'. Supported: DeepLabV3, AttentionUNet, UNETR, SwinUNETR, SwinTransformer, SegResNet, TransUNet.")
