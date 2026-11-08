"""Pipeline figure for the report: input -> boxes -> per-box window (image + box channel) -> mask -> instances.

    python visualize_pipeline.py --csv outputs/submission_final.csv --stem 20130825141154Bh

Illustrated with the final predictions of one test image: the boxes drawn are the bounding boxes of the
predicted filaments and the per-box mask is the corresponding predicted filament.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from visualize_submission import SIZE, color, decode_rle

MARGIN, MIN_WINDOW, MAX_ASPECT, CROP = 0.25, 48, 2.0, 256
TILE, GAP, LABEL_H, PAD = 420, 90, 120, 20


def window(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Box enlarged by MARGIN per side, at least MIN_WINDOW, aspect ratio at most MAX_ASPECT (cf. filseg.instance)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max((x1 - x0) * (1 + 2 * MARGIN), MIN_WINDOW), max((y1 - y0) * (1 + 2 * MARGIN), MIN_WINDOW)
    w, h = max(w, h / MAX_ASPECT), max(h, w / MAX_ASPECT)
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def crop(arr: np.ndarray, win: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = win
    out = np.zeros((y1 - y0, x1 - x0) + arr.shape[2:], arr.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, SIZE), min(y1, SIZE)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def tile(arr: np.ndarray, size: int = TILE) -> Image.Image:
    return Image.fromarray(arr).convert("RGB").resize((size, size), Image.LANCZOS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("outputs/submission_final.csv"))
    p.add_argument("--images", type=Path,
                   default=Path("/Users/vincentliu/Documents/MAGFiLO_1.0_Kaggle_2026/test/test_images"))
    p.add_argument("--stem", default="20130825141154Bh")
    p.add_argument("--out", type=Path, default=Path("report/figures/pipeline.jpg"))
    args = p.parse_args()

    sub = pd.read_csv(args.csv)
    rles = sub.segmentation_rle[sub.filament_id.str.rsplit("_", n=1).str[0] == args.stem].tolist()
    masks = [decode_rle(r) for r in rles]
    gray = np.asarray(Image.open(next(args.images.glob(args.stem + ".*"))).convert("L"))
    rgb = np.stack([gray] * 3, -1)
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    k = int(np.argmax([m.sum() for m in masks]))  # largest filament for the per-box panels

    # (b) boxes
    det = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(det)
    for i, b in enumerate(boxes):
        d.rectangle(b, outline=(255, 210, 0) if i != k else (255, 60, 60), width=12)
    # (c) window: image crop and box channel
    win = window(boxes[k])
    img_c = np.asarray(Image.fromarray(crop(gray, win)).resize((CROP, CROP), Image.BILINEAR))
    box_ch = np.zeros(SIZE * SIZE, np.uint8).reshape(SIZE, SIZE)
    x0, y0, x1, y1 = boxes[k]
    box_ch[y0:y1, x0:x1] = 255
    box_c = np.asarray(Image.fromarray(crop(box_ch, win)).resize((CROP, CROP), Image.NEAREST))
    half = TILE // 2 - 6
    stack = Image.new("RGB", (TILE, TILE), (255, 255, 255))
    stack.paste(tile(img_c, half), ((TILE - half) // 2, 0))
    stack.paste(tile(box_c, half), ((TILE - half) // 2, TILE - half))
    # (d) per-filament mask on the crop
    m_c = np.asarray(Image.fromarray(crop(masks[k].astype(np.uint8) * 255, win)).resize((CROP, CROP),
                                                                                     Image.NEAREST)) > 127
    seg = np.stack([img_c] * 3, -1).astype(np.float32)
    seg[m_c] = 0.4 * seg[m_c] + 0.6 * np.array((255, 60, 60), np.float32)
    # (e) assembled instances
    inst = rgb.astype(np.float32)
    for i, m in enumerate(masks):
        inst[m] = 0.3 * inst[m] + 0.7 * np.array(color(i), np.float32)

    panels = [
        (tile(rgb), "Input image\n2048x2048, grayscale"),
        (tile(np.asarray(det)), "1. YOLO11m @ 1536\none box per filament"),
        (stack, "2. Window per box (+25%)\n256x256: image + box mask"),
        (tile(seg.astype(np.uint8)), "3. Box-conditioned U-Net\n(ResNet-34): filament mask"),
        (tile(inst.astype(np.uint8)), "4. Paste back by confidence\nmask NMS -> instances"),
    ]
    W = len(panels) * TILE + (len(panels) - 1) * GAP + 2 * PAD
    fig = Image.new("RGB", (W, TILE + LABEL_H + 2 * PAD), (255, 255, 255))
    d = ImageDraw.Draw(fig)
    for j, (im, label) in enumerate(panels):
        x = PAD + j * (TILE + GAP)
        fig.paste(im, (x, PAD))
        d.multiline_text((x + TILE // 2, PAD + TILE + 18), label, fill=(0, 0, 0), font_size=34,
                         anchor="ma", align="center", spacing=6)
        if j:
            ax0, ay = x - GAP + 14, PAD + TILE // 2
            d.line((ax0, ay, x - 18, ay), fill=(60, 60, 60), width=8)
            d.polygon([(x - 6, ay), (x - 30, ay - 16), (x - 30, ay + 16)], fill=(60, 60, 60))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(args.out, quality=92)
    print(f"{args.stem}: {len(masks)} filaments, window {win} for filament {k + 1} -> {args.out}")


if __name__ == "__main__":
    main()
