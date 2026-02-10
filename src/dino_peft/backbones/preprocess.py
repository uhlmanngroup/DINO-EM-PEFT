from __future__ import annotations

from typing import Any, Mapping

from dino_peft.utils.transforms import em_dino_unsup_transforms, openclip_native_transforms


def resolve_preprocess_cfg(cfg: Mapping[str, Any], default_img_size: Any) -> dict:
    backbone_cfg = cfg.get("backbone") if isinstance(cfg, Mapping) else None
    backbone_cfg = backbone_cfg or {}
    preprocess = dict(backbone_cfg.get("preprocess") or {})
    preprocess.setdefault("preset", "em")
    if "img_size" not in preprocess:
        preprocess["img_size"] = default_img_size
    return preprocess


def build_preprocess_transform(
    preset: str,
    img_size: Any,
    *,
    backbone_cfg: Mapping[str, Any] | None = None,
):
    preset = (preset or "em").lower()
    backbone_name = None
    if backbone_cfg:
        backbone_name = str(backbone_cfg.get("name") or "").lower()
    if preset == "em":
        print(f"[preprocess] preset=em backbone={backbone_name or 'unknown'} img_size={img_size}")
        return em_dino_unsup_transforms(img_size=img_size)
    if preset == "openclip_native":
        print(
            "[preprocess] preset=openclip_native "
            f"backbone={backbone_name or 'unknown'} img_size={img_size}"
        )
        return openclip_native_transforms(img_size=img_size)
    raise ValueError(f"Unknown preprocess preset '{preset}'.")
