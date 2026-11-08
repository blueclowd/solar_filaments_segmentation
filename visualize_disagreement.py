"""Find images where annotators disagree most and draw each annotator's filaments side by side.

Panels: annotator A (red), annotator B (blue), and the filaments from either annotator with
no match (mask IoU > 0.5) in the other (yellow).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA_ROOT = Path("/Users/vincentliu/Documents/MAGFiLO_1.0_Kaggle_2026")
ANN_PATH = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = DATA_ROOT / "train" / "train_images"
ORIG = 2048
GRID = 512  # resolution used for IoU computation

COLOR_A, COLOR_B, COLOR_DIFF = (255, 50, 50), (50, 120, 255), (255, 230, 0)


def filament_mask(fil: list[list[float]], size: int) -> np.ndarray:
    s = size / ORIG
    img = Image.new("1", (size, size), 0)
    d = ImageDraw.Draw(img)
    for poly in fil:
        d.polygon([(x * s, y * s) for x, y in zip(poly[0::2], poly[1::2])], fill=1)
    return np.asarray(img, bool)


def compare(a: list, b: list) -> dict:
    """Greedy one-to-one matching at mask IoU > 0.5, plus split/merge counts."""
    ma = np.stack([filament_mask(f, GRID) for f in a]).reshape(len(a), -1) if a else np.zeros((0, GRID * GRID), bool)
    mb = np.stack([filament_mask(f, GRID) for f in b]).reshape(len(b), -1) if b else np.zeros((0, GRID * GRID), bool)
    inter = ma.astype(np.float32) @ mb.T.astype(np.float32)
    area_a, area_b = ma.sum(1)[:, None], mb.sum(1)[None, :]
    iou = inter / np.maximum(area_a + area_b - inter, 1)
    matched_a, matched_b = set(), set()
    for i, j in sorted(np.argwhere(iou > 0.5).tolist(), key=lambda p: -iou[p[0], p[1]]):
        if i not in matched_a and j not in matched_b:
            matched_a.add(i), matched_b.add(j)
    # a filament in one set overlapping >=2 filaments of the other (each covering >20% of the smaller)
    cover = inter / np.maximum(np.minimum(area_a, area_b), 1)
    splits = int(((cover > 0.2).sum(1) >= 2).sum()) if len(b) else 0
    merges = int(((cover > 0.2).sum(0) >= 2).sum()) if len(a) else 0
    n = len(a) + len(b)
    f1 = 2 * len(matched_a) / n if n else 1.0
    union = ma.any(0) | mb.any(0) if n else np.zeros(1, bool)
    pix_iou = (ma.any(0) & mb.any(0)).sum() / max(union.sum(), 1) if n else 1.0
    link = cover > 0.2
    return dict(matched_a=matched_a, matched_b=matched_b, f1=f1, pix_iou=float(pix_iou),
                splits=splits + merges, na=len(a), nb=len(b),
                reason_a={i: reason(link, i, "B") for i in range(len(a)) if i not in matched_a},
                reason_b={j: reason(link.T, j, "A") for j in range(len(b)) if j not in matched_b})


def reason(link: np.ndarray, i: int, other: str) -> str:
    """Why filament i has no match: the other annotator didn't draw it, drew it as a different
    number of pieces, or drew the same piece with a different outline (IoU <= 0.5)."""
    js = np.flatnonzero(link[i]) if link.shape[1] else []
    if len(js) == 0:
        return f"not in {other}"
    if len(js) >= 2 or link[:, js[0]].sum() >= 2:
        return "split/merge"
    return "outline"


def draw_set(base: Image.Image, fils: list, c: tuple, title: str) -> Image.Image:
    out = base.copy().convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for fil in fils:
        for poly in fil:
            pts = list(zip(poly[0::2], poly[1::2]))
            d.polygon(pts, fill=c + (80,), outline=c + (255,), width=4)
    out = Image.alpha_composite(out, layer).convert("RGB")
    ImageDraw.Draw(out).text((20, 15), title, fill=(255, 255, 255), font_size=56)
    return out


def draw_unmatched(base: Image.Image, a: list, b: list, s: dict) -> Image.Image:
    unmatched = [(a[k], r) for k, r in s["reason_a"].items()] + [(b[k], r) for k, r in s["reason_b"].items()]
    out = base.copy()
    d = ImageDraw.Draw(out)
    for fil, _ in unmatched:
        for poly in fil:
            d.polygon(list(zip(poly[0::2], poly[1::2])), fill=COLOR_DIFF)
    for fil, r in unmatched:
        xs = [x for poly in fil for x in poly[0::2]]
        ys = [y for poly in fil for y in poly[1::2]]
        d.text((max(xs) + 10, min(ys)), r, fill=COLOR_DIFF, font_size=44, stroke_width=3, stroke_fill=(0, 0, 0))
    d.text((20, 15), f"unmatched filaments: {len(unmatched)} of {len(a) + len(b)}",
           fill=(255, 255, 255), font_size=56)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--min-filaments", type=int, default=4, help="skip near-empty images")
    p.add_argument("--pick", choices=["typical", "worst"], default="typical",
                   help="typical = agreement closest to the median image")
    p.add_argument("--panel", type=int, default=768)
    p.add_argument("--out", type=Path, default=Path("outputs/disagreement"))
    args = p.parse_args()

    data = json.loads(ANN_PATH.read_text())
    fils_by_id = defaultdict(list)
    for ann in data["annotations"]:
        fils_by_id[ann["image_id"]].append(ann["segmentation"])
    sets_by_file = defaultdict(list)
    for img in sorted(data["images"], key=lambda i: i["id"]):
        sets_by_file[img["file_name"]].append((img["id"], fils_by_id[img["id"]]))

    rows = []
    for fname, sets in sets_by_file.items():
        if len(sets) < 2:
            continue
        (ida, a), (idb, b) = sets[0], sets[1]
        rows.append((fname, ida, idb, a, b, compare(a, b)))
    print(f"{len(sets_by_file)} images, {len(rows)} with >=2 annotators")
    total = sum(r[5]["na"] + r[5]["nb"] for r in rows)
    reasons = Counter(x for r in rows for x in [*r[5]["reason_a"].values(), *r[5]["reason_b"].values()])
    unmatched = sum(reasons.values())
    print(f"filaments drawn: {total}, matched: {total - unmatched} ({1 - unmatched / total:.0%}), "
          f"unmatched: {unmatched} ({unmatched / total:.0%})")
    for r in ["not in A", "not in B", "split/merge", "outline"]:
        print(f"  {r:12s} {reasons[r]:5d}  ({reasons[r] / total:.0%} of all filaments)")

    # 95% bootstrap intervals, resampling images
    keys = ["matched", "single", "split/merge", "outline"]
    per_img = np.array([[r[5]["na"] + r[5]["nb"], 2 * len(r[5]["matched_a"])]
                        + [sum(v in ({"not in A", "not in B"} if k == "single" else {k})
                               for v in [*r[5]["reason_a"].values(), *r[5]["reason_b"].values()])
                           for k in keys[1:]] for r in rows], float)
    rng = np.random.default_rng(0)
    boot = np.stack([per_img[rng.integers(0, len(per_img), len(per_img))].sum(0) for _ in range(2000)])
    shares = boot[:, 1:] / boot[:, :1]
    for k, (lo, hi) in zip(keys, np.percentile(shares, [2.5, 97.5], axis=0).T):
        print(f"  95% CI {k:12s} {lo:.1%} - {hi:.1%}")

    rows = [r for r in rows if min(r[5]["na"], r[5]["nb"]) >= args.min_filaments]
    if args.pick == "worst":
        rows.sort(key=lambda r: r[5]["f1"])
    else:
        med = np.median([r[5]["f1"] for r in rows])
        rows.sort(key=lambda r: abs(r[5]["f1"] - med))
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.jpg"):
        old.unlink()
    for rank, (fname, ida, idb, a, b, s) in enumerate(rows[: args.n], 1):
        base = Image.open(IMG_DIR / fname).convert("RGB")
        panels = [
            draw_set(base, a, COLOR_A, f"A: {ida}  ({s['na']} filaments)"),
            draw_set(base, b, COLOR_B, f"B: {idb}  ({s['nb']} filaments)"),
            draw_unmatched(base, a, b, s),
        ]
        P = args.panel
        sheet = Image.new("RGB", (3 * P, P + 50), (0, 0, 0))
        for k, pan in enumerate(panels):
            sheet.paste(pan.resize((P, P), Image.LANCZOS), (k * P, 50))
        ImageDraw.Draw(sheet).text(
            (10, 10),
            f"{fname}   matched F1={s['f1']:.2f}   pixel IoU={s['pix_iou']:.2f}   split/merge={s['splits']}",
            fill=(255, 255, 255), font_size=26)
        path = args.out / f"{rank}_{Path(fname).stem}.jpg"
        sheet.save(path, quality=88)
        print(f"{rank}. {fname}: A={s['na']} B={s['nb']} F1={s['f1']:.2f} pixIoU={s['pix_iou']:.2f} "
              f"split/merge={s['splits']} -> {path}")


if __name__ == "__main__":
    main()
