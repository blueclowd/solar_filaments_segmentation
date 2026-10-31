"""Two-stage prediction: YOLO boxes -> box-conditioned U-Net mask for each filament.

    # validation: oracle score with ground-truth boxes, then tune the YOLO pipeline's settings
    python predict_instances.py --seg-run runs/instance_r34 --yolo-weights runs/yolo/train/weights/best.pt --split val
    # submission with the tuned settings (<seg-run>/instance_settings.json)
    python predict_instances.py --seg-run runs/instance_r34 --yolo-weights runs/yolo/train/weights/best.pt \
        --split test --out submission_instances.csv

Train YOLO and the segmentation model on the same split (`--unet-run` / `--split-from` the same run folder).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from filseg.config import ANN_NAME, resolve_data_root
from filseg.data import decode_instances, filaments_to_rles, load_gray, load_records
from filseg.inference import predict_probs
from filseg.instance import WindowParams, assemble_instances, filament_xyxy, instance_inputs
from filseg.metrics import PQAccumulator, match_instances
from predict import encode_rle_counts
from predict_hybrid import fmt, load_unet


def seg_probs(model, img, boxes, params, device, tta, batch=64):
    x, wins = instance_inputs(img, boxes, params)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.from_numpy(x[i:i + batch]).to(device)
            out.append(predict_probs(model, t, tta=tta, amp=device.type == "cuda")[:, 0].cpu().numpy().astype(np.float16))
    probs = np.concatenate(out) if out else np.zeros((0, params.crop_size, params.crop_size), np.float16)
    return probs, wins


def detect(yolo, paths, imgsz, nms_iou, min_conf, device, batch=8) -> dict:
    dets = {}
    for i in tqdm(range(0, len(paths), batch), desc="yolo"):
        chunk = paths[i:i + batch]
        results = yolo.predict([str(p) for p in chunk], imgsz=imgsz, conf=min_conf, iou=nms_iou, max_det=100,
                               half=device.type == "cuda", verbose=False)
        for p, r in zip(chunk, results):
            dets[p.name] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
    return dets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seg-run", required=True, help="train_instance.py output folder (best.pt, split.json)")
    parser.add_argument("--seg-checkpoint", help="default: <seg-run>/best.pt")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference size (match its training)")
    parser.add_argument("--yolo-nms-iou", type=float, default=0.6)
    parser.add_argument("--min-conf", type=float, default=0.05, help="lowest YOLO confidence kept for tuning")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True, help="flip TTA on each crop")
    parser.add_argument("--settings", help="default: <seg-run>/instance_settings.json")
    parser.add_argument("--out", default="submission_instances.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.seg_run)
    settings_path = Path(args.settings) if args.settings else run_dir / "instance_settings.json"
    model, ckpt = load_unet(Path(args.seg_checkpoint or run_dir / "best.pt"), device)
    cfg, pp = ckpt["config"], dict(ckpt["postprocess"])
    d = cfg["data"]
    params = WindowParams(d["crop_size"], d["margin"], d["min_window"], d["max_aspect"])
    root = resolve_data_root(d["root"])
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)

    if args.split == "val":
        val_names = set(json.loads((run_dir / "split.json").read_text())["val"])
        records = [r for r in load_records(root / "train" / ANN_NAME) if r.file_name in val_names]
        paths = [root / "train" / "train_images" / r.file_name for r in records]
        dets = detect(yolo, paths, args.imgsz, args.yolo_nms_iou, args.min_conf, device)

        oracle, cache = PQAccumulator(), {}
        for rec, path in zip(tqdm(records, desc="segment"), paths):
            img = load_gray(path)
            gt_rles = [filaments_to_rles(a) for a in rec.annotations]
            for a, rles in zip(rec.annotations, gt_rles):
                boxes = np.array([filament_xyxy(f) for f in a]).reshape(-1, 4)
                probs, wins = seg_probs(model, img, boxes, params, device, args.tta)
                labels = assemble_instances(probs, wins, np.ones(len(boxes)), conf=0, threshold=pp["threshold"],
                                            min_area=pp["min_area"], nms_iou=1.0)
                oracle.add(match_instances(labels, decode_instances(rles)))
            boxes, scores = dets[path.name]
            probs, wins = seg_probs(model, img, boxes, params, device, args.tta)
            cache[rec.file_name] = (probs, wins, scores, gt_rles)
        print(f"oracle (ground-truth boxes): {fmt(oracle.summary())}")

        rows = []
        def trial(**s):
            acc = PQAccumulator()
            for probs, wins, scores, gt_rles in cache.values():
                labels = assemble_instances(probs, wins, scores, **s)
                for rles in gt_rles:
                    acc.add(match_instances(labels, decode_instances(rles)))
            m = acc.summary()
            rows.append({**s, **m})
            print(f"yolo {s}\n  {fmt(m)}")

        base = dict(threshold=pp["threshold"], min_area=pp["min_area"], nms_iou=pp["nms_iou"])
        for conf in (0.1, 0.2, 0.3, 0.4, 0.5):
            trial(conf=conf, **base)
        for key, values in (("threshold", (pp["threshold"] - 0.1, pp["threshold"] + 0.1)),
                            ("nms_iou", (0.3, 0.7)), ("min_area", (100, 300))):
            best = max(rows, key=lambda r: r["pq"])
            current = {k: best[k] for k in ("conf", "threshold", "min_area", "nms_iou")}
            for v in values:
                trial(**{**current, key: round(v, 2) if isinstance(v, float) else v})
        best = max(rows, key=lambda r: r["pq"])
        settings = {k: best[k] for k in ("conf", "threshold", "min_area", "nms_iou")}
        settings_path.write_text(json.dumps(
            {**settings, "imgsz": args.imgsz, "yolo_nms_iou": args.yolo_nms_iou}, indent=1))
        pd.DataFrame(rows).sort_values("pq", ascending=False).to_csv(run_dir / "instance_sweep.csv", index=False)
        print(f"\nbest {settings}  PQ {best['pq']:.4f}  (oracle {oracle.summary()['pq']:.4f}) -> {settings_path}")
        return

    settings = json.loads(settings_path.read_text()) if settings_path.exists() else \
        {k: pp[k] for k in ("conf", "threshold", "min_area", "nms_iou")}
    imgsz = settings.pop("imgsz", args.imgsz)
    yolo_nms = settings.pop("yolo_nms_iou", args.yolo_nms_iou)
    print(f"settings {settings}  (yolo imgsz={imgsz}, nms_iou={yolo_nms})")
    image_dir = root / "test" / "test_images"
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    t0 = time.time()
    dets = detect(yolo, paths, imgsz, yolo_nms, settings["conf"], device)
    rows, empty = [], 0
    for path in tqdm(paths, desc="segment"):
        boxes, scores = dets[path.name]
        probs, wins = seg_probs(model, load_gray(path), boxes, params, device, args.tta)
        labels = assemble_instances(probs, wins, scores, **settings)
        n = int(labels.max())
        empty += n == 0
        rows += [(f"{path.stem}_{k}", encode_rle_counts(labels == k)) for k in range(1, n + 1)]
    elapsed = time.time() - t0
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(f"{len(paths)} images, {len(rows)} filaments ({empty} images with none), "
          f"{elapsed:.1f}s total ({elapsed / len(paths):.2f}s/image) -> {args.out}")


if __name__ == "__main__":
    main()
