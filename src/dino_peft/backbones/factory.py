from __future__ import annotations

from typing import Any, Mapping

from .dinov2 import DINOv2Adapter
from .dinov3 import DINOv3Adapter
from .openclip_adapter import OpenCLIPAdapter


def resolve_backbone_cfg(cfg: Mapping[str, Any]) -> dict:
    raw = dict(cfg.get("backbone") or {})
    name = str(raw.get("name") or cfg.get("backbone_name") or "dinov2").lower()
    variant = raw.get("variant") or cfg.get("dino_size") or cfg.get("variant")
    model = raw.get("model") or variant
    if name == "openclip":
        variant = model or variant or "ViT-L-14"
    if not variant:
        variant = "base"
    load_backend = raw.get("load_backend", "torchhub")
    weights = raw.get("weights") or cfg.get("weights")
    repo_dir = raw.get("repo_dir")
    num_register_tokens = raw.get("num_register_tokens")
    pretrained = raw.get("pretrained") or cfg.get("pretrained")
    preprocess = raw.get("preprocess") or {}
    return {
        "name": name,
        "variant": variant,
        "model": model or variant,
        "load_backend": load_backend,
        "weights": weights,
        "repo_dir": repo_dir,
        "num_register_tokens": num_register_tokens,
        "pretrained": pretrained,
        "preprocess": preprocess,
    }


def backbone_id(backbone_cfg: Mapping[str, Any]) -> str:
    name = backbone_cfg.get("name", "dinov2")
    variant = backbone_cfg.get("variant", "base")
    return f"{name}-{variant}"


def build_backbone(backbone_cfg: Mapping[str, Any], device: torch.device | str):
    name = str(backbone_cfg.get("name", "dinov2")).lower()
    if name == "dinov2":
        return DINOv2Adapter(variant=str(backbone_cfg.get("variant", "base")), device=device)
    if name == "dinov3":
        return DINOv3Adapter(
            variant=str(backbone_cfg.get("variant", "vits16")),
            device=device,
            load_backend=str(backbone_cfg.get("load_backend", "torchhub")),
            repo_dir=backbone_cfg.get("repo_dir"),
            weights=backbone_cfg.get("weights"),
            num_register_tokens=backbone_cfg.get("num_register_tokens"),
        )
    if name == "openclip":
        model_name = backbone_cfg.get("model") or backbone_cfg.get("variant") or "ViT-L-14"
        return OpenCLIPAdapter(
            model_name=str(model_name),
            device=device,
            pretrained=backbone_cfg.get("pretrained"),
            weights=backbone_cfg.get("weights"),
        )
    raise ValueError(f"Unsupported backbone name '{name}'.")
