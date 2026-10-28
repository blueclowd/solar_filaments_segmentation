from pathlib import Path

import yaml

ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply `dotted.key=value` overrides (values parsed as YAML)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def resolve_data_root(root: str | Path) -> Path:
    """Return the dataset root (the folder holding train/ and test/)."""
    root = Path(root)
    if (root / "train" / ANN_NAME).exists():
        return root
    for search in (root, Path("/kaggle/input")):
        if search.exists():
            hits = sorted(search.rglob(ANN_NAME))
            if hits:
                return hits[0].parent.parent
    raise FileNotFoundError(f"Could not find {ANN_NAME} under {root} or /kaggle/input")
