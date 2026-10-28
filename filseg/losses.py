import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """BCE for per-pixel calibration plus batch-level soft Dice against the heavy foreground imbalance.

    Works with soft targets (annotator averages) as well as binary ones.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float | None = None, smooth: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth
        self.register_buffer("pos_weight", torch.tensor([pos_weight]) if pos_weight else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        dice = 1 - (2 * inter + self.smooth) / (prob.sum() + target.sum() + self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice
