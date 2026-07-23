import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss
