"""YOLO + U-Net hybrid: U-Net pixels grouped into filaments by YOLO boxes.

    # 1) tune on the validation split and compare with U-Net-only post-processing
    python predict_hybrid.py --unet-run runs/unet_r34 --yolo-weights runs/yolo/train/weights/best.pt --split val
    # 2) write a submission with the tuned settings (read from <unet-run>/hybrid_settings.json)
    python predict_hybrid.py --unet-run runs/unet_r34 --yolo-weights runs/yolo/train/weights/best.pt --split test
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from filseg.config import ANN_NAME, resolve_data_root
from filseg.data import InferenceDataset, decode_instances, filaments_to_rles, load_records
from filseg.hybrid import combine_instances
from filseg.inference import predict_probs
from filseg.metrics import PQAccumulator, match_instances
from filseg.model import build_model
from filseg.postprocess import probs_to_instances, upsample
from predict import encode_rle_counts


def load_unet(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if ckpt["config"]["train"].get("channels_last"):
        model.to(memory_format=torch.channels_last)
    return model, ckpt


def run_models(paths, unet, image_size, yolo, imgsz, nms_iou, tta, device, batch_size=4, num_workers=4):
    """Per image: U-Net probability at `image_size` (float16) and YOLO boxes (xyxy, 2048 frame) + scores."""
    loader = DataLoader(InferenceDataset(paths, image_size), batch_size=batch_size, num_workers=num_workers)
    probs, dets = {}, {}
    with torch.no_grad():
        for images, idxs in tqdm(loader, desc="unet + yolo"):
            p = predict_probs(unet, images.to(device), tta=tta, amp=device.type == "cuda")[:, 0].cpu().numpy()
            batch_paths = [paths[i] for i in idxs.tolist()]
            results = yolo.predict([str(x) for x in batch_paths], imgsz=imgsz, conf=0.01, iou=nms_iou,
                                   max_det=300, half=device.type == "cuda", verbose=False)
            for path, prob, r in zip(batch_paths, p, results):
                probs[path.name] = prob.astype(np.float16)
                dets[path.name] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
    return probs, dets


def score(records, probs, dets, make_labels) -> dict:
    acc = PQAccumulator()
    for rec in records:
        labels = make_labels(upsample(probs[rec.file_name].astype(np.float32)), *dets[rec.file_name])
        for a in rec.annotations:
            acc.add(match_instances(labels, decode_instances(filaments_to_rles(a))))
    return acc.summary()


def fmt(m: dict) -> str:
    return (f"PQ={m['pq']:.4f}  PQ/img={m['pq_image_mean']:.4f}  SQ={m['sq']:.3f}  RQ={m['rq']:.3f}  "
            f"FP={m['fp']}  FN={m['fn']}  split={m['gt_split']}  merge={m['pred_merge']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unet-run", required=True, help="U-Net run folder (best.pt, split.json)")
    parser.add_argument("--unet-checkpoint", help="default: <unet-run>/best.pt")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference size (match training)")
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True, help="U-Net flip TTA")
    parser.add_argument("--settings", help="hybrid settings JSON (default: <unet-run>/hybrid_settings.json)")
    parser.add_argument("--out", default="submission_hybrid.csv", help="submission path (--split test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.unet_run)
    settings_path = Path(args.settings) if args.settings else run_dir / "hybrid_settings.json"
    unet, ckpt = load_unet(Path(args.unet_checkpoint or run_dir / "best.pt"), device)
    pp = dict(ckpt["postprocess"])
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)

    root = resolve_data_root(ckpt["config"]["data"]["root"])
    image_size = ckpt["config"]["data"]["image_size"]

    if args.split == "val":
        val_names = set(json.loads((run_dir / "split.json").read_text())["val"])
        records = [r for r in load_records(root / "train" / ANN_NAME) if r.file_name in val_names]
        paths = [root / "train" / "train_images" / r.file_name for r in records]
        probs, dets = run_models(paths, unet, image_size, yolo, args.imgsz, args.nms_iou, args.tta, device)

        base = score(records, probs, dets, lambda prob, b, s: probs_to_instances(prob, **pp))
        print(f"U-Net only   {pp}\n  {fmt(base)}")

        common = dict(close_radius=pp["close_radius"], min_area=pp["min_area"])
        rows = []
        def trial(**kw):
            s = {**common, **kw}
            m = score(records, probs, dets, lambda prob, b, sc: combine_instances(prob, b, sc, **s))
            rows.append({**s, **m})
            print(f"hybrid {kw}\n  {fmt(m)}")
            return m["pq"]

        # coordinate search: box confidence x unmatched handling, then splitting, then the U-Net threshold
        for conf, keep in itertools.product([0.1, 0.2, 0.3, 0.4, 0.5], [True, False]):
            trial(threshold=pp["threshold"], conf=conf, keep_unmatched=keep, split_min_frac=0.2)
        best = max(rows, key=lambda r: r["pq"])
        trial(threshold=pp["threshold"], conf=best["conf"], keep_unmatched=best["keep_unmatched"], split_min_frac=1.0)
        best = max(rows, key=lambda r: r["pq"])
        for t in (pp["threshold"] - 0.1, pp["threshold"] + 0.1):
            if 0.05 <= t <= 0.95:
                trial(threshold=round(t, 2), conf=best["conf"], keep_unmatched=best["keep_unmatched"],
                      split_min_frac=best["split_min_frac"])
        best = max(rows, key=lambda r: r["pq"])

        keys = ["threshold", "close_radius", "conf", "min_cover", "split_min_frac", "keep_unmatched",
                "fragment_min_area", "min_area"]
        defaults = dict(min_cover=0.5, fragment_min_area=30)
        settings = {k: best.get(k, defaults.get(k)) for k in keys}
        settings_path.write_text(json.dumps({**settings, "imgsz": args.imgsz, "nms_iou": args.nms_iou}, indent=1))
        pd.DataFrame(rows).sort_values("pq", ascending=False).to_csv(run_dir / "hybrid_sweep.csv", index=False)
        print(f"\nbest hybrid {settings}\n  PQ {best['pq']:.4f} vs U-Net only {base['pq']:.4f} "
              f"-> {settings_path}")
        return

    settings = json.loads(settings_path.read_text())
    imgsz, nms_iou = settings.pop("imgsz", args.imgsz), settings.pop("nms_iou", args.nms_iou)
    print(f"hybrid settings {settings}  (yolo imgsz={imgsz}, nms_iou={nms_iou})")
    image_dir = root / "test" / "test_images"
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    t0 = time.time()
    probs, dets = run_models(paths, unet, image_size, yolo, imgsz, nms_iou, args.tta, device)
    rows, empty = [], 0
    for p in tqdm(paths, desc="combine"):
        labels = combine_instances(upsample(probs[p.name].astype(np.float32)), *dets[p.name], **settings)
        n = int(labels.max())
        empty += n == 0
        rows += [(f"{p.stem}_{k}", encode_rle_counts(labels == k)) for k in range(1, n + 1)]
    elapsed = time.time() - t0
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(f"{len(paths)} images, {len(rows)} filaments ({empty} images with none), "
          f"{elapsed:.1f}s total ({elapsed / len(paths):.2f}s/image) -> {args.out}")


if __name__ == "__main__":
    main()
