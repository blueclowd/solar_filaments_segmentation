"""Class-agnostic Panoptic Quality (Kirillov et al., CVPR 2019) plus Dice and split/merge counts."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImageStats:
    tp: int
    fp: int
    fn: int
    iou_sum: float
    dice: float
    gt_split: int    # GT filaments overlapped by more than one prediction (one-to-many)
    pred_merge: int  # predictions overlapping more than one GT filament (many-to-one)

    @property
    def pq(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denom if denom else 1.0


def match_instances(pred_labels: np.ndarray, gt_masks: np.ndarray, iou_threshold: float = 0.5) -> ImageStats:
    """`pred_labels`: int label map (non-overlapping). `gt_masks`: (N, H, W) bool, may overlap."""
    n_pred, n_gt = int(pred_labels.max()), len(gt_masks)
    pred_area = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:]
    inter = np.zeros((n_gt, n_pred))
    for g, m in enumerate(gt_masks):
        inter[g] = np.bincount(pred_labels[m], minlength=n_pred + 1)[1:]
    gt_area = gt_masks.reshape(n_gt, -1).sum(1) if n_gt else np.zeros(0)
    union = gt_area[:, None] + pred_area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    tp, iou_sum, used_g, used_p = 0, 0.0, set(), set()
    order = np.unravel_index(np.argsort(-iou, axis=None), iou.shape) if iou.size else ([], [])
    for g, p in zip(*order):
        if iou[g, p] <= iou_threshold:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        tp += 1
        iou_sum += float(iou[g, p])

    pred_fg = pred_labels > 0
    gt_fg = gt_masks.any(0) if n_gt else np.zeros_like(pred_fg)
    denom = pred_fg.sum() + gt_fg.sum()
    dice = 2 * np.logical_and(pred_fg, gt_fg).sum() / denom if denom else 1.0

    overlaps = inter > 0
    return ImageStats(
        tp=tp,
        fp=n_pred - tp,
        fn=n_gt - tp,
        iou_sum=iou_sum,
        dice=float(dice),
        gt_split=int((overlaps.sum(1) > 1).sum()),
        pred_merge=int((overlaps.sum(0) > 1).sum()),
    )


@dataclass
class PQAccumulator:
    stats: list[ImageStats] = field(default_factory=list)

    def add(self, s: ImageStats) -> None:
        self.stats.append(s)

    def summary(self) -> dict[str, float]:
        if not self.stats:
            return {}
        tp = sum(s.tp for s in self.stats)
        fp = sum(s.fp for s in self.stats)
        fn = sum(s.fn for s in self.stats)
        iou_sum = sum(s.iou_sum for s in self.stats)
        denom = tp + 0.5 * fp + 0.5 * fn
        return {
            "pq": iou_sum / denom if denom else 1.0,              # dataset-level PQ
            "pq_image_mean": float(np.mean([s.pq for s in self.stats])),
            "sq": iou_sum / tp if tp else 0.0,                    # mean IoU of matched pairs
            "rq": tp / denom if denom else 1.0,                   # F1-style detection quality
            "dice": float(np.mean([s.dice for s in self.stats])),
            "tp": tp, "fp": fp, "fn": fn,
            "gt_split": sum(s.gt_split for s in self.stats),
            "pred_merge": sum(s.pred_merge for s in self.stats),
        }
