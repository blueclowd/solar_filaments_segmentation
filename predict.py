"""Predict filament instances on the test set and write the Kaggle submission CSV.

    python predict.py --checkpoint runs/unet_r34/best.pt --out submission.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from filseg.config import resolve_data_root
from filseg.data import InferenceDataset
from filseg.inference import predict_probs
from filseg.model import build_model
from filseg.postprocess import probs_to_instances, upsample


def encode_rle_counts(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", help="test image folder (default: <data root>/test/test_images)")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold", type=float, help="override the tuned threshold stored in the checkpoint")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, pp = ckpt["config"], dict(ckpt["postprocess"])
    if args.threshold is not None:
        pp["threshold"] = args.threshold
    print(f"checkpoint epoch {ckpt['epoch']}  postprocess {pp}")

    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    image_dir = Path(args.images) if args.images else resolve_data_root(cfg["data"]["root"]) / "test" / "test_images"
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    loader = DataLoader(InferenceDataset(paths, cfg["data"]["image_size"]), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    rows, empty, t0 = [], 0, time.time()
    with torch.no_grad():
        for images, idxs in tqdm(loader, desc="predict"):
            probs = predict_probs(model, images.to(device), tta=args.tta, amp=device.type == "cuda")
            for prob, idx in zip(probs[:, 0].cpu().numpy(), idxs.tolist()):
                labels = probs_to_instances(upsample(prob), **pp)
                n = int(labels.max())
                empty += n == 0
                stem = paths[idx].stem
                rows += [(f"{stem}_{k}", encode_rle_counts(labels == k)) for k in range(1, n + 1)]
    elapsed = time.time() - t0

    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(f"{len(paths)} images, {len(rows)} filaments ({empty} images with none), "
          f"{elapsed:.1f}s total ({elapsed / len(paths):.2f}s/image) -> {args.out}")


if __name__ == "__main__":
    main()
