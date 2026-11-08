"""Draw a submission's predicted filaments on test images (one colour per filament), for the report.

    python visualize_submission.py --csv submission.csv --images <data>/test/test_images

Needs only numpy and Pillow: the COCO compressed-RLE strings are decoded here.
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SIZE = 2048


def decode_rle(counts: str, h: int = SIZE, w: int = SIZE) -> np.ndarray:
    """Decode a pycocotools compressed RLE string (runs alternate 0/1, column-major)."""
    runs, p = [], 0
    while p < len(counts):
        x, k, more = 0, 0, True
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p, k = p + 1, k + 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    flat = np.repeat(np.arange(len(runs)) % 2, runs).astype(bool)
    return np.pad(flat, (0, h * w - len(flat))).reshape(w, h).T


def color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def overlay(img_path: Path, rles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    union = np.zeros((SIZE, SIZE), bool)
    for i, counts in enumerate(rles):
        m = decode_rle(counts)
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        rgb[m] = 0.6 * rgb[m] + 0.4 * np.array(color(i), np.float32)
        rgb[edge] = color(i)
        union |= m
    return rgb.astype(np.uint8), union


def panel(rgb: np.ndarray, size: int, title: str) -> Image.Image:
    out = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    ImageDraw.Draw(out).text((12, 8), title, fill=(255, 255, 255), font_size=size // 28,
                             stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def zoom_box(union: np.ndarray, side: int) -> tuple[int, int, int, int]:
    """Square window of `side` px (full resolution) containing the most predicted filament pixels."""
    ii = np.pad(union.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[side:, side:] - ii[:-side, side:] - ii[side:, :-side] + ii[:-side, :-side]
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return x, y, x + side, y + side


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images", type=Path,
                   default=Path("/Users/vincentliu/Documents/MAGFiLO_1.0_Kaggle_2026/test/test_images"))
    p.add_argument("--n", type=int, default=4, help="images, spread from fewest to most filaments")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--zoom", type=int, default=640, help="side of the full-resolution zoom window, px")
    p.add_argument("--out", type=Path, default=Path("report/latex/figures/test_predictions.jpg"))
    args = p.parse_args()

    sub = pd.read_csv(args.csv)
    sub["image"] = sub.filament_id.str.rsplit("_", n=1).str[0]
    per_image = sub.groupby("image").size().sort_values()
    stems = per_image.index[np.linspace(0, len(per_image) - 1, args.n).astype(int)]
    panels = []
    for s in stems:
        rgb, union = overlay(next(args.images.glob(s + ".*")), sub.segmentation_rle[sub.image == s].tolist())
        panels.append(panel(rgb, args.size, f"{s}: {per_image[s]} filaments"))
    x0, y0, x1, y1 = zoom_box(union, args.zoom)  # busiest image (last), at full resolution
    panels.append(panel(rgb[y0:y1, x0:x1], args.size, f"{stems[-1]}, zoom ({args.zoom} px, full res.)"))
    sheet = Image.new("RGB", (args.size * len(panels), args.size))
    for k, pan in enumerate(panels):
        sheet.paste(pan, (k * args.size, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=90)
    print(f"{len(sub)} filaments in {len(per_image)} images; drew {', '.join(stems)} -> {args.out}")


if __name__ == "__main__":
    main()
