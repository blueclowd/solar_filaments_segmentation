"""Train a YOLO filament detector (boxes only) for the YOLO + U-Net hybrid.

    python train_yolo.py --unet-run runs/unet_r34 --model yolo11s.pt --imgsz 1024 --epochs 100

`--unet-run` reuses that run's validation split so both models are tuned and scored on the same images.
Extra Ultralytics training arguments can be passed as key=value, e.g. `mosaic=0.5 lr0=0.005`.
"""

import argparse
import json
from pathlib import Path

import yaml

from filseg.config import ANN_NAME, load_config, resolve_data_root
from filseg.data import load_records, split_records
from filseg.yolo_data import export_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/unet_r34.yaml", help="for the data root and split settings")
    parser.add_argument("--unet-run", help="U-Net run folder whose split.json defines the validation images")
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics detection weights, e.g. yolo11n/s/m.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default="runs/yolo")
    parser.add_argument("overrides", nargs="*", help="extra Ultralytics train arguments as key=value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    root = resolve_data_root(dcfg["root"])
    records = load_records(root / "train" / ANN_NAME)
    if args.unet_run:
        val_names = set(json.loads((Path(args.unet_run) / "split.json").read_text())["val"])
        train = [r for r in records if r.file_name not in val_names]
        val = [r for r in records if r.file_name in val_names]
    else:
        train, val = split_records(records, dcfg["val_fraction"], cfg["seed"], dcfg["split_by"])
    out_dir = Path(args.out_dir)
    data_yaml = export_yolo_dataset(train, val, root / "train" / "train_images", out_dir / "data")
    print(f"YOLO dataset: {len(train)} train images ({sum(len(r.annotations) for r in train)} annotation sets), "
          f"{len(val)} val images -> {data_yaml}")

    from ultralytics import YOLO

    train_args = dict(
        data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, patience=args.patience,
        workers=args.workers, project=str(out_dir.resolve()), name="train", exist_ok=True, single_cls=True,
        # grayscale full-disk images: no hue/saturation jitter; any flip is valid once classes are merged
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, fliplr=0.5, flipud=0.5, degrees=0.0,
    )
    train_args.update({k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in args.overrides)})
    YOLO(args.model).train(**train_args)
    print(f"best weights: {out_dir / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
