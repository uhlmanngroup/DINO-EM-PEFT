#!/usr/bin/env python3
"""Smoke test LoRA injection for DINO backbones.

Example (local, DINOv2 only):
    python scripts/smoke_test_lora.py --device cpu

Example (local, include DINOv3; absolute paths are user-specific):
    python scripts/smoke_test_lora.py \
      --dinov3-weights /Users/cfuste/Documents/Models/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
      --dinov3-repo-dir /Users/cfuste/Documents/GitHub/dinov3 \
      --device cpu
"""

from __future__ import annotations

import argparse

import torch

from dino_peft.backbones import build_backbone, resolve_backbone_cfg
from dino_peft.models.lora import apply_peft


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _assert_lora_only_trainable(model: torch.nn.Module) -> None:
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise AssertionError("No trainable parameters found after LoRA injection.")
    bad = [n for n in trainable if "lora_" not in n]
    if bad:
        raise AssertionError(f"Unexpected trainable params (non-LoRA): {bad[:10]}")


def _run_backbone(name: str, variant: str, device: torch.device, weights: str | None, repo_dir: str | None) -> None:
    cfg = resolve_backbone_cfg(
        {
            "backbone": {
                "name": name,
                "variant": variant,
                "weights": weights,
                "repo_dir": repo_dir,
                "load_backend": "torchhub",
            }
        }
    )
    print(f"[lora_smoke] Loading {name}:{variant} on {device}...")
    backbone = build_backbone(cfg, device=device)

    lora_cfg = {
        "lora": {
            "enabled": True,
            "target_policy": "vit_attention_only",
            "layer_selection": "all",
            "exclude": [],
            "compatibility_mode": True,
        },
        "lora_rank": 2,
        "lora_alpha": 4,
    }
    audit = apply_peft(backbone.model, lora_cfg, backbone_info=cfg, write_report=False)
    if audit is None or audit.total_targets == 0:
        raise AssertionError("LoRA target discovery produced zero targets.")
    if audit.block_count is not None and audit.blocks_targeted != audit.block_count:
        raise AssertionError(
            f"Blocks targeted ({audit.blocks_targeted}) != backbone depth ({audit.block_count})."
        )
    _assert_lora_only_trainable(backbone.model)
    print(f"[lora_smoke] {name}:{variant} targets={audit.total_targets} blocks={audit.blocks_targeted}")
    print(f"[lora_smoke] sample targets: {audit.targets[:6]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test LoRA injection for DINO backbones.")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument("--dinov3-variant", default="vits16")
    ap.add_argument("--dinov3-weights", default=None)
    ap.add_argument("--dinov3-repo-dir", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)

    _run_backbone("dinov2", "small", device, weights=None, repo_dir=None)

    if args.dinov3_weights:
        _run_backbone(
            "dinov3",
            args.dinov3_variant,
            device,
            weights=args.dinov3_weights,
            repo_dir=args.dinov3_repo_dir,
        )
    else:
        print("[lora_smoke] DINOv3 weights not provided; skipping DINOv3 LoRA smoke test.")


if __name__ == "__main__":
    main()
