"""Export MAGFiLO to the Ultralytics YOLO detection format (one class: filament).

Training images get one label file per annotator (the image is symlinked under a different name for
each annotation set); validation uses the first annotator only, for Ultralytics' own mAP.
"""

import os
from pathlib import Path

import numpy as np
import yaml

from .data import ORIG_SIZE, Filament, Record


def filament_box(fil: Filament) -> tuple[float, float, float, float]:
    """Normalised (cx, cy, w, h) of a filament's polygon(s)."""
    pts = np.concatenate([np.asarray(p, np.float64).reshape(-1, 2) for p in fil])
    x0, y0 = np.clip(pts.min(0), 0, ORIG_SIZE)
    x1, y1 = np.clip(pts.max(0), 0, ORIG_SIZE)
    return (x0 + x1) / 2 / ORIG_SIZE, (y0 + y1) / 2 / ORIG_SIZE, (x1 - x0) / ORIG_SIZE, (y1 - y0) / ORIG_SIZE


def export_yolo_dataset(train: list[Record], val: list[Record], image_dir: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    for split, recs in (("train", train), ("val", val)):
        img_out, lbl_out = out_dir / "images" / split, out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for rec in recs:
            sets = list(enumerate(rec.annotations)) if split == "train" else [(0, rec.annotations[0])]
            for k, filaments in sets:
                name = f"{rec.stem}__a{k}"
                link = img_out / f"{name}.jpg"
                if not link.exists():
                    os.symlink((image_dir / rec.file_name).resolve(), link)
                lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in map(filament_box, filaments)]
                (lbl_out / f"{name}.txt").write_text("\n".join(lines) + "\n")
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "names": {0: "filament"}}
    ))
    return data_yaml
