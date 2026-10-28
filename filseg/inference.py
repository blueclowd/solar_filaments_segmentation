import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import FilamentEvalDataset, decode_instances
from .metrics import PQAccumulator, match_instances
from .postprocess import probs_to_instances, upsample


def forward_logits(model: nn.Module, images: torch.Tensor, amp: bool = True) -> torch.Tensor:
    with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=amp and images.is_cuda):
        return model(images).float()


def predict_probs(
    model: nn.Module, images: torch.Tensor, tta: bool = False, amp: bool = True, logits: torch.Tensor | None = None
) -> torch.Tensor:
    """Sigmoid probabilities, optionally averaged over horizontal/vertical flips."""
    probs = torch.sigmoid(forward_logits(model, images, amp) if logits is None else logits)
    if tta:
        for dims in [(-1,), (-2,), (-1, -2)]:
            probs = probs + torch.sigmoid(forward_logits(model, images.flip(dims), amp)).flip(dims)
        probs = probs / 4
    return probs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    postprocess_cfg: dict,
    thresholds: list[float] | None = None,
    tta: bool = False,
    amp: bool = True,
    criterion: nn.Module | None = None,
) -> dict[float, dict]:
    """PQ at the original resolution for each threshold.

    Every annotator's annotation set counts as a separate ground truth, mirroring how the
    organisers treat multiply-annotated images.
    """
    model.eval()
    dataset: FilamentEvalDataset = loader.dataset
    thresholds = thresholds or [postprocess_cfg["threshold"]]
    pp = {k: v for k, v in postprocess_cfg.items() if k != "threshold"}
    accs = {t: PQAccumulator() for t in thresholds}
    loss_sum, n = 0.0, 0

    for images, targets, idxs in tqdm(loader, desc="val", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = forward_logits(model, images, amp)
        if criterion is not None:
            loss_sum += criterion(logits, targets).item() * len(images)
            n += len(images)
        probs = predict_probs(model, images, tta=tta, amp=amp, logits=logits)
        for prob, idx in zip(probs[:, 0].cpu().numpy(), idxs.tolist()):
            prob = upsample(prob)
            gts = [decode_instances(rles) for rles in dataset.gt_rles[idx]]
            for t in thresholds:
                labels = probs_to_instances(prob, threshold=t, **pp)
                for gt in gts:
                    accs[t].add(match_instances(labels, gt))

    results = {t: acc.summary() for t, acc in accs.items()}
    if criterion is not None:
        for r in results.values():
            r["val_loss"] = loss_sum / max(n, 1)
    return results
