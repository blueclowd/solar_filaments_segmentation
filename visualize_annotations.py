"""Draw MAGFiLO annotations (polygon + spine) on randomly sampled training images."""

import argparse
import colorsys
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

DATA_ROOT = Path("/Users/vincentliu/Documents/MAGFiLO_1.0_Kaggle_2026")
ANN_PATH = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = DATA_ROOT / "train" / "train_images"

CATEGORY_COLORS = {1: (255, 80, 80), 2: (80, 160, 255), 3: (255, 220, 0), 4: (200, 80, 255)}


def instance_color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def pairs(flat: list[float]) -> list[tuple[float, float]]:
    return list(zip(flat[0::2], flat[1::2]))


def draw(image_info: dict, anns: list[dict], categories: dict, color_by: str) -> Image.Image:
    base = Image.open(IMG_DIR / image_info["file_name"]).convert("RGB")
    fill_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill = ImageDraw.Draw(fill_layer)
    for i, ann in enumerate(anns):
        color = CATEGORY_COLORS[ann["category_id"]] if color_by == "category" else instance_color(i)
        for poly in ann["segmentation"]:
            fill.polygon(pairs(poly), fill=color + (90,))

    out = Image.alpha_composite(base.convert("RGBA"), fill_layer).convert("RGB")
    pen = ImageDraw.Draw(out)
    for i, ann in enumerate(anns):
        color = CATEGORY_COLORS[ann["category_id"]] if color_by == "category" else instance_color(i)
        for poly in ann["segmentation"]:
            pen.line(pairs(poly) + pairs(poly)[:1], fill=color, width=3)
        if ann.get("spine"):
            pen.line(pairs(ann["spine"]), fill=(255, 255, 255), width=2)
        x, y, w, h = ann["bbox"]
        pen.rectangle([x, y, x + w, y + h], outline=color, width=1)

    legend = ", ".join(f"{c['id']}={c['name']}" for c in categories.values())
    pen.rectangle([0, 0, out.width, 70], fill=(0, 0, 0))
    pen.text((10, 8), f"{image_info['id']}   filaments={len(anns)}", fill=(255, 255, 255), font_size=28)
    pen.text((10, 40), f"color by {color_by}  ({legend}); white=spine", fill=(200, 200, 200), font_size=22)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=1024, help="output side length")
    parser.add_argument("--color-by", choices=["category", "instance"], default="category")
    parser.add_argument("--out", type=Path, default=Path("outputs/annotation_check"))
    args = parser.parse_args()

    data = json.loads(ANN_PATH.read_text())
    categories = {c["id"]: c for c in data["categories"]}
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    samples = random.Random(args.seed).sample(data["images"], args.n)
    args.out.mkdir(parents=True, exist_ok=True)

    thumbs = []
    for info in samples:
        img = draw(info, anns_by_image[info["id"]], categories, args.color_by)
        img = img.resize((args.size, args.size), Image.LANCZOS)
        img.save(args.out / f"{info['id']}.jpg", quality=92)
        thumbs.append(img)
        print(f"{info['id']}: {len(anns_by_image[info['id']])} filaments")

    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    t = 512
    sheet = Image.new("RGB", (cols * t, rows * t))
    for k, img in enumerate(thumbs):
        sheet.paste(img.resize((t, t), Image.LANCZOS), ((k % cols) * t, (k // cols) * t))
    sheet.save(args.out / "contact_sheet.jpg", quality=90)
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
