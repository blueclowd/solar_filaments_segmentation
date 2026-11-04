import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -torch.maximum(F.max_pool2d(-x, (3, 1), 1, (1, 0)), F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable skeleton of a (B, 1, H, W) map in [0, 1] (Shit et al., clDice, CVPR 2021).
    `iters` must exceed the half-width of the thickest structure, in pixels."""
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(prob: torch.Tensor, target: torch.Tensor, iters: int, smooth: float = 1.0) -> torch.Tensor:
    """1 - clDice: harmonic mean of the predicted skeleton lying inside the target and the target
    skeleton covered by the prediction. Rewards unbroken, centred thin structures."""
    skel_pred, skel_true = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    tprec = ((skel_pred * target).sum() + smooth) / (skel_pred.sum() + smooth)
    tsens = ((skel_true * prob).sum() + smooth) / (skel_true.sum() + smooth)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def lovasz_hinge(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image Lovász hinge (Berman et al., CVPR 2018): a convex surrogate that directly optimises
    the foreground IoU, the quantity PQ's segmentation term averages."""
    losses = []
    for lg, t in zip(logits.flatten(1), (target > 0.5).float().flatten(1)):
        errors = 1 - lg * (2 * t - 1)
        errors_sorted, order = torch.sort(errors, descending=True)
        gt_sorted = t[order]
        gts = gt_sorted.sum()
        inter = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1 - inter / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        losses.append(torch.dot(F.relu(errors_sorted), jaccard))
    return torch.stack(losses).mean()


class BCEDiceLoss(nn.Module):
    """BCE for per-pixel calibration plus batch-level soft Dice against the heavy foreground imbalance,
    with optional clDice (thin-structure continuity) and Lovász hinge (IoU) terms.

    BCE, Dice and clDice work with soft targets (annotator averages) as well as binary ones.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float | None = None,
                 smooth: float = 1.0, cldice_weight: float = 0.0, cldice_iters: int = 15,
                 lovasz_weight: float = 0.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth
        self.cldice_weight, self.cldice_iters, self.lovasz_weight = cldice_weight, cldice_iters, lovasz_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]) if pos_weight else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        dice = 1 - (2 * inter + self.smooth) / (prob.sum() + target.sum() + self.smooth)
        loss = self.bce_weight * bce + self.dice_weight * dice
        if self.cldice_weight:
            loss = loss + self.cldice_weight * cldice_loss(prob, target, self.cldice_iters, self.smooth)
        if self.lovasz_weight:
            loss = loss + self.lovasz_weight * lovasz_hinge(logits, target)
        return loss
