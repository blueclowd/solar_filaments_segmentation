import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(model_cfg: dict, pretrained: bool = True) -> nn.Module:
    """Single-logit output (filament vs background); 1 input channel unless `in_channels` is set."""
    arch = getattr(smp, model_cfg["arch"])
    return arch(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg.get("encoder_weights") if pretrained else None,
        in_channels=model_cfg.get("in_channels", 1),
        classes=1,
        **(model_cfg.get("kwargs") or {}),
    )
