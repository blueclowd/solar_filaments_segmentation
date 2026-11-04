"""Train the box-conditioned single-filament U-Net (second stage after the YOLO detector).

    python train_instance.py --config configs/instance_r34.yaml --split-from runs/unet_r34
    python train_instance.py --epochs 30 --crop-size 384 --encoder resnet50
    python train_instance.py --split-from runs/unet_r34 --full --output-dir runs/instance_full  # final model

Validation reports, per ground-truth filament with its true box: mean mask IoU, the fraction with
IoU > 0.5 (a PQ true positive), and `pq_oracle` = mean of IoU over those matches, i.e. the PQ this model
would reach with perfect boxes. The best epoch by `pq_oracle` is saved as best.pt.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from filseg.config import ANN_NAME, load_config, resolve_data_root
from filseg.data import load_records, split_records
from filseg.inference import forward_logits
from filseg.instance import InstanceDataset, WindowParams
from filseg.losses import BCEDiceLoss
from filseg.model import build_model
from train import cosine_with_warmup, save_checkpoint, seed_everything, unwrap

FLAGS = [
    ("--output-dir", "output_dir", str),
    ("--seed", "seed", int),
    ("--arch", "model.arch", str),
    ("--encoder", "model.encoder", str),
    ("--epochs", "train.epochs", int),
    ("--batch-size", "train.batch_size", int),
    ("--lr", "train.lr", float),
    ("--crop-size", "data.crop_size", int),
    ("--margin", "data.margin", float),
    ("--box-jitter", "data.box_jitter", float),
    ("--num-workers", "data.num_workers", int),
]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, criterion: nn.Module,
             threshold: float = 0.5) -> dict:
    model.eval()
    ious, loss_sum, n = [], 0.0, 0
    for x, t in tqdm(loader, desc="val", leave=False):
        x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
        logits = forward_logits(model, x, amp)
        loss_sum += criterion(logits, t).item() * len(x)
        n += len(x)
        pred, gt = torch.sigmoid(logits) > threshold, t > 0.5
        inter = (pred & gt).sum((1, 2, 3)).float()
        union = (pred | gt).sum((1, 2, 3)).float()
        ious.append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).cpu())
    iou = torch.cat(ious).numpy()
    matched = iou > 0.5
    return {
        "val_loss": loss_sum / max(n, 1),
        "mean_iou": float(iou.mean()),
        "match_rate": float(matched.mean()),
        "pq_oracle": float((iou * matched).sum() / len(iou)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/instance_r34.yaml")
    parser.add_argument("--split-from", help="run folder whose split.json defines the validation images")
    parser.add_argument("--full", action="store_true",
                        help="final model: train on all images; validation images are still scored (optimistic, "
                             "monitoring only) and best.pt is the last epoch")
    for flag, key, typ in FLAGS:
        parser.add_argument(flag, type=typ, default=None, help=f"[config: {key}]")
    parser.add_argument("overrides", nargs="*", help="any other setting as dotted.key=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    for flag, key, _ in FLAGS:
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is not None:
            *parents, leaf = key.split(".")
            node = cfg
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
    dcfg, tcfg = cfg["data"], cfg["train"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tcfg["amp"] and device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    root = resolve_data_root(dcfg["root"])
    image_dir = root / "train" / "train_images"
    records = load_records(root / "train" / ANN_NAME)
    if args.split_from:
        val_names = set(json.loads((Path(args.split_from) / "split.json").read_text())["val"])
        train_recs = [r for r in records if r.file_name not in val_names]
        val_recs = [r for r in records if r.file_name in val_names]
    else:
        train_recs, val_recs = split_records(records, dcfg["val_fraction"], cfg["seed"], dcfg["split_by"])
    (out_dir / "split.json").write_text(json.dumps({"val": [r.file_name for r in val_recs]}, indent=1))
    if args.full:
        train_recs = records

    params = WindowParams(dcfg["crop_size"], dcfg["margin"], dcfg["min_window"], dcfg["max_aspect"])
    t0 = time.time()
    train_ds = InstanceDataset(train_recs, image_dir, params, dcfg["box_jitter"], augment=True,
                               cache_images=dcfg["cache_images"])
    val_ds = InstanceDataset(val_recs, image_dir, params, cache_images=dcfg["cache_images"])
    print(f"data root: {root}\ntrain: {len(train_recs)} images / {len(train_ds)} filaments   "
          f"val: {len(val_recs)} images / {len(val_ds)} filaments   (ready in {time.time() - t0:.0f}s)")

    loader_kw = dict(num_workers=dcfg["num_workers"], pin_memory=True, persistent_workers=dcfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=tcfg["val_batch_size"], shuffle=False, **loader_kw)

    model = build_model(cfg["model"]).to(device)
    if tcfg.get("channels_last"):
        model = model.to(memory_format=torch.channels_last)
    criterion = BCEDiceLoss(**tcfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(tcfg["warmup_epochs"] * len(train_loader), total_steps, tcfg["min_lr_ratio"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, history = -1.0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False)
        for step, (x, t) in enumerate(pbar, 1):
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            loss = criterion(forward_logits(model, x, amp), t)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        row = {"epoch": epoch, "train_loss": running / len(train_loader), "lr": scheduler.get_last_lr()[0]}
        if epoch % tcfg["val_every"] == 0 or epoch == tcfg["epochs"]:
            row.update(evaluate(model, val_loader, device, amp, criterion, cfg["postprocess"]["threshold"]))
            if row["pq_oracle"] > best or args.full:
                best = row["pq_oracle"]
                save_checkpoint(out_dir / "best.pt", unwrap(model), cfg, epoch, row)
        save_checkpoint(out_dir / "last.pt", unwrap(model), cfg, epoch, row)
        row["epoch_time_s"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    print(f"best pq_oracle {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
