#!/usr/bin/env python3
"""Inspect backbone module tree and LoRA targets without training.

Example (local):
    python scripts/inspect_backbone.py --cfg configs/mac/paired_openclip_vitl14.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from dino_peft.backbones import build_backbone, resolve_backbone_cfg
from dino_peft.models.lora import apply_peft
from dino_peft.utils.paths import setup_run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect backbone modules and LoRA target discovery.")
    ap.add_argument("--cfg", required=True, help="Path to YAML config.")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument("--run-dir", default=None, help="Optional output directory override.")
    ap.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Max module depth to include in the tree summary.",
    )
    ap.add_argument("--image-size-h", type=int, default=None, help="Optional dummy input height.")
    ap.add_argument("--image-size-w", type=int, default=None, help="Optional dummy input width.")
    ap.add_argument(
        "--respect-config",
        action="store_true",
        help="Do not override lora.enabled for inspection.",
    )
    return ap.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file {path} is empty or invalid.")
    return cfg


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def summarize_module_tree(model: torch.nn.Module, max_depth: int) -> list[str]:
    lines = []
    for name, module in model.named_modules():
        if not name:
            continue
        if name.count(".") > max_depth:
            continue
        lines.append(f"{name}: {module.__class__.__name__}")
    return lines


def force_enable_lora(cfg: dict) -> dict:
    lora_cfg = dict(cfg.get("lora") or {})
    lora_cfg.setdefault("target_policy", "vit_attention_only")
    lora_cfg.setdefault("layer_selection", "all")
    lora_cfg.setdefault("exclude", ["head", "decoder", "seg_head"])
    lora_cfg["enabled"] = True
    cfg["lora"] = lora_cfg
    cfg["use_lora"] = True
    return cfg


def resolve_run_dir(cfg: dict, override: str | None) -> Path:
    if override:
        run_dir = Path(override).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    if "experiment_id" in cfg and "results_root" in cfg:
        task_type = cfg.get("task_type", "inspect")
        return setup_run_dir(cfg, task_type=task_type, save_config=True)
    run_dir = Path.cwd() / "runs" / "inspect"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg)
    cfg = load_config(cfg_path)
    if not args.respect_config:
        cfg = force_enable_lora(cfg)

    device = resolve_device(args.device)
    backbone_cfg = resolve_backbone_cfg(cfg)
    backbone = build_backbone(backbone_cfg, device=device)
    backbone.eval()

    run_dir = resolve_run_dir(cfg, args.run_dir)
    print(f"[inspect] run_dir={run_dir}")
    print(
        f"[inspect] backbone={backbone_cfg.get('name')}:{backbone_cfg.get('variant')} device={device}"
    )

    audit = apply_peft(
        backbone.model,
        cfg,
        run_dir=run_dir,
        backbone_info=backbone_cfg,
        write_report=True,
    )
    if audit is None:
        print("[inspect] LoRA disabled in config; no targets were generated.")
        return

    trainable_params = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    tree_lines = summarize_module_tree(backbone.model, max_depth=int(args.max_depth))
    tree_path = run_dir / "backbone_tree.txt"
    tree_path.write_text("\n".join(tree_lines))

    inspect_path = run_dir / "lora_inspect.txt"
    inspect_lines = [
        f"backbone: {backbone_cfg.get('name')}:{backbone_cfg.get('variant')}",
        f"model: {backbone_cfg.get('model')}",
        f"pretrained: {backbone_cfg.get('pretrained')}",
        f"trainable_params: {trainable_params}",
        f"lora_targets: {audit.total_targets}",
        "targets:",
        *[f"  - {name}" for name in audit.targets],
    ]
    inspect_path.write_text("\n".join(inspect_lines))

    print(f"[inspect] trainable_params={trainable_params:,} lora_targets={audit.total_targets}")
    print("[inspect] module tree:")
    for line in tree_lines:
        print(f"  {line}")
    print("[inspect] lora targets:")
    for name in audit.targets:
        print(f"  {name}")
    print(f"[inspect] module_tree={tree_path}")
    print(f"[inspect] lora_targets={inspect_path}")
    print(f"[inspect] lora_report={run_dir / 'lora_targets.json'}")

    if args.image_size_h and args.image_size_w:
        h = int(args.image_size_h)
        w = int(args.image_size_w)
        if h % backbone.patch_size != 0 or w % backbone.patch_size != 0:
            raise ValueError(
                f"Dummy input size must be divisible by patch_size={backbone.patch_size}."
            )
        dummy = torch.randn(1, 3, h, w, device=device)
        output = backbone(dummy)
        if output.patch_tokens.shape[1] != output.grid_size[0] * output.grid_size[1]:
            raise RuntimeError("Patch token count does not match grid size.")
        print(
            "[inspect] forward_ok "
            f"patch_tokens={tuple(output.patch_tokens.shape)} grid={output.grid_size}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[inspect] ERROR: {exc}", file=sys.stderr)
        raise
