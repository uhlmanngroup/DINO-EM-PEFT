#!/usr/bin/env python3
"""Compute parameter counts for backbones + segmentation head under full, head-only, and LoRA regimes.

Example (local):
    python scripts/utils/param_count_table.py --cfg configs/cluster/param_counts.yaml

Example (cluster):
    sbatch slurm/param_counts.sbatch
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml

from dino_peft.backbones import build_backbone
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.models.lora import apply_peft


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "configs" / "cluster" / "param_counts.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build backbones + segmentation head and report parameter counts for "
            "full fine-tuning, head-only, and LoRA regimes."
        )
    )
    ap.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG,
        help=f"YAML config file (default: {DEFAULT_CFG}).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path override.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu/cuda/mps). Defaults to config or cpu.",
    )
    ap.add_argument(
        "--skip-failures",
        action="store_true",
        help="Continue after errors, recording a row with the error message.",
    )
    return ap.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as handle:
        cfg = yaml.safe_load(handle)
    if not cfg:
        raise ValueError(f"Config file {path} is empty or invalid.")
    return cfg


def _normalize_backbone_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if "backbone" in entry and isinstance(entry["backbone"], dict):
        merged = dict(entry["backbone"])
        for key, val in entry.items():
            if key != "backbone":
                merged.setdefault(key, val)
        return merged
    return dict(entry)


def _count_params(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _resolve_variant_name(backbone_cfg: Dict[str, Any]) -> str:
    name = str(backbone_cfg.get("name", "unknown"))
    if name == "openclip":
        return str(backbone_cfg.get("model") or backbone_cfg.get("variant") or "unknown")
    return str(backbone_cfg.get("variant") or backbone_cfg.get("model") or "unknown")


def _merge_lora_cfg(defaults: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(defaults)
    if override:
        merged.update(override)
    merged.setdefault("enabled", True)
    merged.setdefault("target_policy", "vit_attention_only")
    merged.setdefault("layer_selection", "all")
    merged.setdefault("exclude", ["head", "decoder", "seg_head"])
    merged.setdefault("rank", 16)
    merged.setdefault("alpha", 32)
    merged.setdefault("dropout", 0.0)
    merged.setdefault("compatibility_mode", True)
    return merged


def _pct(value: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * float(value) / float(denom)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _iter_backbones(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    backbones = cfg.get("backbones")
    if not isinstance(backbones, list) or not backbones:
        raise ValueError("Config must include a non-empty 'backbones' list.")
    for entry in backbones:
        if not isinstance(entry, dict):
            raise ValueError("Each backbone entry must be a mapping.")
        yield _normalize_backbone_entry(entry)


def compute_row(
    backbone_cfg: Dict[str, Any],
    *,
    device: str,
    num_classes: int,
    seg_head_cfg: Dict[str, Any],
    lora_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    backbone_name = str(backbone_cfg.get("name", "unknown"))
    backbone_variant = _resolve_variant_name(backbone_cfg)
    backbone_id = f"{backbone_name}-{backbone_variant}"

    backbone = build_backbone(backbone_cfg, device=device)
    backbone_total = _count_params(backbone.model)

    head = SegHeadDeconv(
        in_ch=int(backbone.embed_dim),
        num_classes=int(num_classes),
        n_ups=int(seg_head_cfg.get("n_ups", 4)),
        base_ch=int(seg_head_cfg.get("base_ch", 512)),
    ).to(device)
    head_params = _count_params(head)

    total_full = backbone_total + head_params

    lora_cfg = _merge_lora_cfg(lora_defaults, backbone_cfg.get("lora"))
    lora_payload = {"lora": lora_cfg, "use_lora": True}
    audit = apply_peft(
        backbone.model,
        lora_payload,
        run_dir=None,
        backbone_info=backbone_cfg,
        write_report=False,
    )
    lora_trainable = _count_params(backbone.model, trainable_only=True)
    lora_frozen = max(0, backbone_total - lora_trainable)

    row = {
        "backbone_name": backbone_name,
        "backbone_variant": backbone_variant,
        "backbone_id": backbone_id,
        "backbone_params": backbone_total,
        "seg_head_params": head_params,
        "total_params_full": total_full,
        "head_only_trainable_params": head_params,
        "head_only_pct_total": _pct(head_params, total_full),
        "lora_backbone_trainable_params": lora_trainable,
        "lora_backbone_frozen_params": lora_frozen,
        "lora_backbone_trainable_pct_backbone": _pct(lora_trainable, backbone_total),
        "lora_backbone_trainable_pct_total": _pct(lora_trainable, total_full),
        "lora_plus_head_trainable_params": lora_trainable + head_params,
        "lora_plus_head_pct_total": _pct(lora_trainable + head_params, total_full),
        "lora_rank": int(lora_cfg.get("rank", 0)),
        "lora_alpha": int(lora_cfg.get("alpha", 0)),
        "lora_policy": str(lora_cfg.get("target_policy", "")),
        "lora_dropout": float(lora_cfg.get("dropout", 0.0)),
        "lora_compatibility_mode": bool(lora_cfg.get("compatibility_mode", True)),
        "lora_targets": int(audit.total_targets) if audit else 0,
        "lora_blocks_targeted": int(audit.blocks_targeted) if audit else 0,
    }
    return row


def main() -> None:
    args = parse_args()
    cfg_path = args.cfg.expanduser()
    cfg = load_config(cfg_path)

    device = args.device or str(cfg.get("device", "cpu"))
    num_classes = int(cfg.get("num_classes", 2))
    seg_head_cfg = dict(cfg.get("seg_head") or {})
    lora_defaults = dict(cfg.get("lora") or {})

    output = args.output or cfg.get("output_csv") or "param_counts.csv"
    output_path = Path(output).expanduser()
    _ensure_parent(output_path)

    torch.set_grad_enabled(False)

    rows = []
    errors = 0
    for entry in _iter_backbones(cfg):
        try:
            row = compute_row(
                entry,
                device=device,
                num_classes=num_classes,
                seg_head_cfg=seg_head_cfg,
                lora_defaults=lora_defaults,
            )
            rows.append(row)
            print(
                "[params] "
                f"{row['backbone_id']}: total={row['total_params_full']:,} "
                f"head={row['seg_head_params']:,} "
                f"lora={row['lora_backbone_trainable_params']:,}"
            )
        except Exception as exc:
            if not args.skip_failures:
                raise
            errors += 1
            entry_name = f"{entry.get('name', 'unknown')}-{_resolve_variant_name(entry)}"
            rows.append(
                {
                    "backbone_name": entry.get("name", "unknown"),
                    "backbone_variant": _resolve_variant_name(entry),
                    "backbone_id": entry_name,
                    "error": str(exc),
                }
            )
            print(f"[error] {entry_name}: {exc}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No rows were produced. Check the config backbones list.")

    columns = [
        "backbone_name",
        "backbone_variant",
        "backbone_id",
        "backbone_params",
        "seg_head_params",
        "total_params_full",
        "head_only_trainable_params",
        "head_only_pct_total",
        "lora_backbone_trainable_params",
        "lora_backbone_frozen_params",
        "lora_backbone_trainable_pct_backbone",
        "lora_backbone_trainable_pct_total",
        "lora_plus_head_trainable_params",
        "lora_plus_head_pct_total",
        "lora_rank",
        "lora_alpha",
        "lora_policy",
        "lora_dropout",
        "lora_compatibility_mode",
        "lora_targets",
        "lora_blocks_targeted",
        "error",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[params] wrote {len(rows)} rows to {output_path}")
    if errors:
        print(f"[params] completed with {errors} errors (see error column).", file=sys.stderr)


if __name__ == "__main__":
    main()
