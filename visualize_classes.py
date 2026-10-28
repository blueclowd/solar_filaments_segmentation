"""Full-resolution crops of example filaments for each MAGFiLO class (Left / Right / Unidentifiable)."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from visualize_annotations import ANN_PATH, CATEGORY_COLORS, IMG_DIR, pairs

TILE = 360
MARGIN = 30


def crop_pair(file_name: str, ann: dict) -> Image.Image:
    """Return [raw | outlined] crop around one filament, contrast-stretched and resized to TILE height."""
    x, y, w, h = ann["bbox"]
    side = max(w, h) + 2 * MARGIN
    cx, cy = x + w / 2, y + h / 2
    box = tuple(int(v) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))

    raw = ImageOps.autocontrast(Image.open(IMG_DIR / file_name).convert("L").crop(box), cutoff=1).convert("RGB")
    overlay = raw.copy()
    pen = ImageDraw.Draw(overlay)
    shift = lambda pts: [(u - box[0], v - box[1]) for u, v in pts]
    poly = shift(pairs(ann["segmentation"][0]))
    pen.line(poly + poly[:1], fill=CATEGORY_COLORS[ann["category_id"]], width=max(1, int(side / 300)))
    if ann.get("spine"):
        pen.line(shift(pairs(ann["spine"])), fill=(0, 255, 255), width=max(1, int(side / 400)))

    pair = Image.new("RGB", (2 * TILE + 4, TILE), (0, 0, 0))
    pair.paste(raw.resize((TILE, TILE), Image.LANCZOS), (0, 0))
    pair.paste(overlay.resize((TILE, TILE), Image.LANCZOS), (TILE + 4, 0))
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-area-quantile", type=float, default=0.75,
                        help="sample only from larger filaments so barbs are visible")
    parser.add_argument("--out", type=Path, default=Path("outputs/class_examples.jpg"))
    args = parser.parse_args()

    data = json.loads(ANN_PATH.read_text())
    file_of = {i["id"]: i["file_name"] for i in data["images"]}
    rng = random.Random(args.seed)
    class_ids = [1, 2, 3]
    names = {c["id"]: c["name"] for c in data["categories"]}

    label_w, gap = 270, 8
    cell_w = 2 * TILE + 4
    sheet = Image.new("RGB", (label_w + args.per_class * (cell_w + gap), len(class_ids) * (TILE + gap)), (20, 20, 20))
    pen = ImageDraw.Draw(sheet)

    for row, cid in enumerate(class_ids):
        anns = sorted((a for a in data["annotations"] if a["category_id"] == cid), key=lambda a: a["area"])
        pool = anns[int(len(anns) * args.min_area_quantile):]
        y0 = row * (TILE + gap)
        pen.text((12, y0 + TILE // 2 - 20), names[cid], fill=CATEGORY_COLORS[cid], font_size=34)
        for k, ann in enumerate(rng.sample(pool, args.per_class)):
            sheet.paste(crop_pair(file_of[ann["image_id"]], ann), (label_w + k * (cell_w + gap), y0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(f"Saved {args.out.resolve()}  {sheet.size}")


if __name__ == "__main__":
    main()
