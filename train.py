"""Train a U-Net-family filament segmenter.

    python train.py --config configs/unet_r34.yaml
    python train.py --config configs/unet_r34.yaml model.arch=UnetPlusPlus model.encoder=efficientnet-b3
"""

import argparse
import json
import math
import random
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
from filseg.data import FilamentEvalDataset, FilamentTrainDataset, load_records, split_records
from filseg.inference import evaluate, forward_logits
from filseg.losses import BCEDiceLoss
from filseg.model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(warmup_steps: int, total_steps: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "config": cfg,
            "postprocess": dict(cfg["postprocess"]),
            "epoch": epoch,
            "metrics": {k: float(v) for k, v in metrics.items()},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet_r34.yaml")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
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
    train_recs, val_recs = split_records(records, dcfg["val_fraction"], cfg["seed"], dcfg["split_by"])
    print(f"data root: {root}\nimages: {len(records)}  train: {len(train_recs)}  val: {len(val_recs)} "
          f"(val annotation sets: {sum(len(r.annotations) for r in val_recs)})")
    (out_dir / "split.json").write_text(json.dumps({"val": [r.file_name for r in val_recs]}, indent=1))

    t0 = time.time()
    train_ds = FilamentTrainDataset(
        train_recs, image_dir, dcfg["image_size"], dcfg["crop_size"], dcfg["label_mode"],
        dcfg["fg_crop_prob"], dcfg["repeat"], augment=True, cache_images=dcfg["cache_images"],
    )
    val_ds = FilamentEvalDataset(val_recs, image_dir, dcfg["image_size"], cache_images=dcfg["cache_images"])
    print(f"datasets ready in {time.time() - t0:.0f}s")

    loader_kw = dict(num_workers=dcfg["num_workers"], pin_memory=True, persistent_workers=dcfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=tcfg["val_batch_size"], shuffle=False, **loader_kw)

    model = build_model(cfg["model"]).to(device)
    if tcfg.get("multi_gpu") and torch.cuda.device_count() > 1:
        print(f"using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = BCEDiceLoss(**tcfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(tcfg["warmup_epochs"] * len(train_loader), total_steps, tcfg["min_lr_ratio"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_pq, history = -1.0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False)
        for step, (images, targets) in enumerate(pbar, 1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            loss = criterion(forward_logits(model, images, amp), targets)
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
            metrics = evaluate(model, val_loader, device, cfg["postprocess"], tta=cfg["eval"]["tta"],
                               amp=amp, criterion=criterion)
            row.update(next(iter(metrics.values())))
            if row["pq"] > best_pq:
                best_pq = row["pq"]
                save_checkpoint(out_dir / "best.pt", model, cfg, epoch, row)
        save_checkpoint(out_dir / "last.pt", model, cfg, epoch, row)
        row["epoch_time_s"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    # Tune the binarisation threshold on the validation split using the best checkpoint.
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"])
    sweep = evaluate(model, val_loader, device, cfg["postprocess"], thresholds=cfg["eval"]["threshold_sweep"],
                     tta=cfg["eval"]["tta"], amp=amp)
    for t, m in sweep.items():
        print(f"threshold={t:.2f}  pq={m['pq']:.4f}  sq={m['sq']:.4f}  rq={m['rq']:.4f}  dice={m['dice']:.4f}  "
              f"split={m['gt_split']}  merge={m['pred_merge']}")
    best_t = max(sweep, key=lambda t: sweep[t]["pq"])
    ckpt["postprocess"]["threshold"] = float(best_t)
    ckpt["metrics"] = {k: float(v) for k, v in sweep[best_t].items()}
    torch.save(ckpt, out_dir / "best.pt")
    (out_dir / "threshold_sweep.json").write_text(json.dumps({str(t): m for t, m in sweep.items()}, indent=1))
    print(f"best epoch {ckpt['epoch']}  threshold {best_t}  val PQ {sweep[best_t]['pq']:.4f}  -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
