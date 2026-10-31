"""MAGFiLO loading, train/val splitting, mask rasterisation and PyTorch datasets.

An image may be annotated independently by several annotators. A `Record` groups those
annotation sets by file name, so each image appears once and the annotators can be either
sampled (`random_annotator`) or averaged (`soft`) into a training target.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .augment import IDENTITY_LUT, blur_and_noise, intensity_lut, lut_stats, random_crop_transform, warp

ORIG_SIZE = 2048

Polygon = list[float]
Filament = list[Polygon]


@dataclass
class Record:
    file_name: str
    annotations: list[list[Filament]]  # one list of filaments per annotator

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem


def load_records(json_path: str | Path) -> list[Record]:
    data = json.loads(Path(json_path).read_text())
    filaments_by_image = defaultdict(list)
    for ann in data["annotations"]:
        filaments_by_image[ann["image_id"]].append(ann["segmentation"])
    sets_by_file = defaultdict(list)
    for img in sorted(data["images"], key=lambda i: i["id"]):
        sets_by_file[img["file_name"]].append(filaments_by_image[img["id"]])
    return [Record(f, sets) for f, sets in sorted(sets_by_file.items())]


def split_records(
    records: list[Record], val_fraction: float, seed: int, split_by: str = "month"
) -> tuple[list[Record], list[Record]]:
    """Group-wise split. Filaments persist for days, so grouping by month avoids near-duplicate leakage."""
    key_len = {"month": 6, "day": 8, "image": None}[split_by]
    groups = defaultdict(list)
    for r in records:
        groups[r.file_name[:key_len] if key_len else r.file_name].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train, val = [], []
    target = val_fraction * len(records)
    for k in keys:
        (val if len(val) < target else train).extend(groups[k])
    return train, val


def rasterize(filaments: list[Filament], size: int, shift_bits: int = 4) -> np.ndarray:
    """Union of filament polygons as a uint8 mask at `size` x `size` (sub-pixel accurate)."""
    mask = np.zeros((size, size), np.uint8)
    scale = size / ORIG_SIZE * (1 << shift_bits)
    pts = [
        np.round(np.asarray(poly, np.float64).reshape(-1, 2) * scale).astype(np.int32)
        for fil in filaments
        for poly in fil
    ]
    if pts:
        cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=shift_bits)
    return mask


def filaments_to_rles(filaments: list[Filament], size: int = ORIG_SIZE) -> list[dict]:
    """One COCO RLE per filament, rasterised exactly as pycocotools (and the competition) does."""
    return [mask_utils.merge(mask_utils.frPyObjects(fil, size, size)) for fil in filaments]


def decode_instances(rles: list[dict], size: int = ORIG_SIZE) -> np.ndarray:
    if not rles:
        return np.zeros((0, size, size), bool)
    return mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)


def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray) -> np.ndarray:
    """Per-image standardisation over the whole frame; must match between training and inference."""
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-6)


def preprocess(path: str | Path, size: int) -> np.ndarray:
    return normalize(resize_image(load_gray(path), size).astype(np.float32) / 255.0)


class _ImageCache:
    def __init__(self, paths: list[Path], size: int, enabled: bool):
        self.paths, self.size = paths, size
        self.images = [resize_image(load_gray(p), size) for p in paths] if enabled else None

    def __getitem__(self, i: int) -> np.ndarray:
        if self.images is not None:
            return self.images[i]
        return resize_image(load_gray(self.paths[i]), self.size)


class FilamentTrainDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        image_dir: Path,
        image_size: int,
        crop_size: int | None,
        label_mode: str = "random_annotator",
        fg_crop_prob: float = 0.7,
        repeat: int = 1,
        augment: bool = True,
        cache_images: bool = True,
    ):
        if label_mode not in ("random_annotator", "soft"):
            raise ValueError(f"Unknown label_mode {label_mode!r}")
        self.records = records
        self.image_size, self.crop_size = image_size, crop_size
        self.label_mode, self.fg_crop_prob = label_mode, fg_crop_prob
        self.repeat, self.augment = repeat, augment
        self.images = _ImageCache([image_dir / r.file_name for r in records], image_size, cache_images)

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def _target(self, rec: Record) -> np.ndarray:
        if self.label_mode == "soft":
            masks = [rasterize(a, self.image_size) for a in rec.annotations]
            return np.mean(masks, axis=0).astype(np.float32)
        return rasterize(random.choice(rec.annotations), self.image_size).astype(np.float32)

    def __getitem__(self, idx: int):
        i = idx % len(self.records)
        # DataLoader reseeds `random` per worker, numpy's global RNG is not.
        rng = np.random.default_rng(random.getrandbits(64))
        img = self.images[i]
        target = self._target(self.records[i])
        size = self.crop_size or self.image_size

        lut = intensity_lut(rng) if self.augment else IDENTITY_LUT
        mean, std = lut_stats(img, lut)
        m = random_crop_transform(target, size, self.fg_crop_prob, rng, self.augment)
        img = lut[warp(img, m, size)]
        target = warp(target, m, size)
        if self.augment:
            img = blur_and_noise(img, rng)
        img = (img - mean) / (std + 1e-6)
        return torch.from_numpy(img.astype(np.float32))[None], torch.from_numpy(target)[None]


class FilamentEvalDataset(Dataset):
    """Full-disk images with soft targets (for val loss) and per-annotator GT RLEs at 2048 (for PQ)."""

    def __init__(self, records: list[Record], image_dir: Path, image_size: int, cache_images: bool = True):
        self.records, self.image_size = records, image_size
        self.images = _ImageCache([image_dir / r.file_name for r in records], image_size, cache_images)
        self.gt_rles = [[filaments_to_rles(a) for a in r.annotations] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        img = normalize(self.images[i].astype(np.float32) / 255.0)
        target = np.mean([rasterize(a, self.image_size) for a in self.records[i].annotations], axis=0)
        return torch.from_numpy(img)[None], torch.from_numpy(target.astype(np.float32))[None], i


class InferenceDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths, self.image_size = paths, image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return torch.from_numpy(preprocess(self.paths[i], self.image_size))[None], i
