"""Draw a trained model's predictions against the ground truth on the validation split.

    python visualize_predictions.py --run-dir runs/unet_r34 --n 16 --sort worst

Writes, into <run-dir>/val_vis/:
  * one JPEG per image with three panels:
      ground truth | prediction (green = matched, red = false positive, yellow = missed filament) | probability
  * val_per_image.csv with PQ / TP / FP / FN / split / merge per image (all validation images)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from filseg.config import ANN_NAME, resolve_data_root
from filseg.data import decode_instances, filaments_to_rles, load_gray, load_records, preprocess
from filseg.inference import predict_probs
from filseg.metrics import match_instances, match_pairs, overlap_matrices
from filseg.model import build_model
from filseg.postprocess import probs_to_instances, upsample

# BGR
GREEN, RED, YELLOW, WHITE = (60, 200, 60), (60, 60, 255), (0, 230, 255), (255, 255, 255)


def outline(canvas: np.ndarray, mask: np.ndarray, color: tuple, thickness: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness)


def tint(canvas: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.35) -> None:
    canvas[mask] = (canvas[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def label_bar(text: str, width: int, height: int = 34) -> np.ndarray:
    bar = np.zeros((height, width, 3), np.uint8)
    cv2.putText(bar, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return bar


def render(gray: np.ndarray, prob: np.ndarray, labels: np.ndarray, gt: np.ndarray, title: str, size: int) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pairs = match_pairs(overlap_matrices(labels, gt)[1])
    matched_g, matched_p = {g for g, _, _ in pairs}, {p for _, p, _ in pairs}

    gt_panel = base.copy()
    for m in gt:
        outline(gt_panel, m, WHITE, 3)

    pred_panel = base.copy()
    for k in range(1, int(labels.max()) + 1):
        m = labels == k
        color = GREEN if (k - 1) in matched_p else RED
        tint(pred_panel, m, color)
        outline(pred_panel, m, color, 2)
    for g, m in enumerate(gt):
        if g not in matched_g:
            outline(pred_panel, m, YELLOW, 4)

    prob_panel = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    panels = [cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) for p in (gt_panel, pred_panel, prob_panel)]
    captions = ["ground truth (annotator 1)", "prediction: green=matched  red=false pos.  yellow=missed", "probability"]
    columns = [np.vstack([label_bar(c, size), p]) for c, p in zip(captions, panels)]
    body = np.hstack(columns)
    return np.vstack([label_bar(title, body.shape[1], 44), body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="training output folder (has best.pt and split.json)")
    parser.add_argument("--checkpoint", help="default: <run-dir>/best.pt")
    parser.add_argument("--n", type=int, default=16, help="number of images to draw")
    parser.add_argument("--sort", choices=["worst", "best", "random"], default="worst", help="which images to draw")
    parser.add_argument("--threshold", type=float, help="override the tuned threshold stored in the checkpoint")
    parser.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    parser.add_argument("--size", type=int, default=1024, help="side length of each panel")
    parser.add_argument("--out", help="default: <run-dir>/val_vis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "val_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint or run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg, pp = ckpt["config"], dict(ckpt["postprocess"])
    if args.threshold is not None:
        pp["threshold"] = args.threshold
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if cfg["train"].get("channels_last"):
        model.to(memory_format=torch.channels_last)
    print(f"checkpoint epoch {ckpt['epoch']}  postprocess {pp}")

    root = resolve_data_root(cfg["data"]["root"])
    image_dir = root / "train" / "train_images"
    val_names = set(json.loads((run_dir / "split.json").read_text())["val"])
    records = [r for r in load_records(root / "train" / ANN_NAME) if r.file_name in val_names]

    # Pass 1: score every validation image; keep low-resolution probabilities for drawing.
    rows, probs = [], {}
    with torch.no_grad():
        for rec in tqdm(records, desc="predict"):
            x = torch.from_numpy(preprocess(image_dir / rec.file_name, cfg["data"]["image_size"]))[None, None]
            prob = predict_probs(model, x.to(device), tta=args.tta, amp=device.type == "cuda")[0, 0].cpu().numpy()
            probs[rec.file_name] = prob.astype(np.float16)
            labels = probs_to_instances(upsample(prob), **pp)
            stats = [match_instances(labels, decode_instances(filaments_to_rles(a))) for a in rec.annotations]
            rows.append({
                "file_name": rec.file_name,
                "pq": float(np.mean([s.pq for s in stats])),
                "dice": float(np.mean([s.dice for s in stats])),
                "n_pred": int(labels.max()),
                "n_gt": float(np.mean([len(a) for a in rec.annotations])),
                "tp": float(np.mean([s.tp for s in stats])),
                "fp": float(np.mean([s.fp for s in stats])),
                "fn": float(np.mean([s.fn for s in stats])),
                "gt_split": float(np.mean([s.gt_split for s in stats])),
                "pred_merge": float(np.mean([s.pred_merge for s in stats])),
                "annotators": len(rec.annotations),
            })

    df = pd.DataFrame(rows).sort_values("pq").reset_index(drop=True)
    df.to_csv(out_dir / "val_per_image.csv", index=False)
    print(df.describe().loc[["mean", "50%", "min", "max"], ["pq", "dice", "n_pred", "n_gt", "fp", "fn"]].round(3))

    if args.sort == "worst":
        chosen = df.head(args.n)
    elif args.sort == "best":
        chosen = df.tail(args.n).iloc[::-1]
    else:
        chosen = df.sample(min(args.n, len(df)), random_state=args.seed)

    # Pass 2: draw the chosen images at full resolution, matched against the first annotator.
    by_name = {r.file_name: r for r in records}
    for rank, row in enumerate(tqdm(chosen.itertuples(), total=len(chosen), desc="draw"), 1):
        rec = by_name[row.file_name]
        prob = upsample(probs[rec.file_name].astype(np.float32))
        labels = probs_to_instances(prob, **pp)
        gt = decode_instances(filaments_to_rles(rec.annotations[0]))
        title = (f"{rec.stem}   PQ={row.pq:.3f}  Dice={row.dice:.3f}   pred={row.n_pred}  gt={row.n_gt:g}   "
                 f"FP={row.fp:g}  FN={row.fn:g}  split={row.gt_split:g}  merge={row.pred_merge:g}   "
                 f"(averaged over {row.annotators} annotator(s))")
        img = render(load_gray(image_dir / rec.file_name), prob, labels, gt, title, args.size)
        cv2.imwrite(str(out_dir / f"{rank:02d}_{rec.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"wrote {len(chosen)} images and val_per_image.csv to {out_dir}")


if __name__ == "__main__":
    main()
