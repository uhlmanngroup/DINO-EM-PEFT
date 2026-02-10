"""Extract unsupervised DINO features and save them in the run directory layout.

Example (local):
    python scripts/extract_features.py --cfg configs/mac/em_unsupervised_features_mac.yaml

Example (cluster):
    sbatch slurm/feat_analysis_paired.sbatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from dino_peft.analysis.feature_extractor import extract_features_from_folder
from dino_peft.backbones import resolve_backbone_cfg
from dino_peft.utils.paths import setup_run_dir, update_metrics, write_run_info
from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG
from copy import deepcopy

DEFAULT_CFG_PATH = (
    Path(__file__).parent.parent / "configs" / "mac" / "em_unsupervised_features_mac.yaml"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract DINO features for EM datasets.")
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        default=str(DEFAULT_CFG_PATH),
        help="Path to YAML config.",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint (.pt) to load LoRA weights from.",
    )
    return ap.parse_args()


def load_config(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file {cfg_path} is empty or invalid.")
    return cfg


def resolve_device(device_str: str) -> str:
    device_str = device_str.lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but torch.cuda.is_available() is False.")
        return "cuda"

    if device_str == "mps":
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            raise RuntimeError("Requested device 'mps' but MPS backend is not available.")
        return "mps"

    if device_str == "cpu":
        return "cpu"

    raise ValueError(f"Invalid device '{device_str}', expected 'auto', 'cpu', 'cuda', or 'mps'.")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg)
    print(f"[extract_features] Using config file: {cfg_path}")

    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    checkpoint_arg = args.checkpoint
    checkpoint_cfg = model_cfg.get("checkpoint")
    checkpoint_path = checkpoint_arg or checkpoint_cfg
    checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
    if checkpoint_path and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if "img_size" not in runtime_cfg:
        img_size_cfg = deepcopy(DEFAULT_IMG_SIZE_CFG)
        print("[extract_features] img_size not set in config; defaulting to longest_edge=1022.")
    else:
        img_size_cfg = runtime_cfg["img_size"]
    data_dir = data_cfg.get("data_dir")
    if data_dir is None:
        raise ValueError("Config must define data.data_dir.")

    backbone_cfg = resolve_backbone_cfg(model_cfg)
    dino_size = backbone_cfg.get("variant")
    batch_size = int(runtime_cfg.get("batch_size", 16))
    num_workers = int(runtime_cfg.get("num_workers", 4))
    device = resolve_device(runtime_cfg.get("device", "auto"))

    run_dir = None
    out_path: Path
    task_type = cfg.get("task_type", "feats")
    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(cfg, task_type=task_type, subdirs=("plots",))
        out_path = run_dir / "features.npz"
        write_run_info(
            run_dir,
            {
                "task_type": task_type,
                "device": device,
                "img_size": img_size_cfg,
                "backbone_name": backbone_cfg.get("name"),
                "backbone_variant": backbone_cfg.get("variant"),
                "dino_size": dino_size,
                "data_dir": data_dir,
            },
        )
    else:
        output_path = data_cfg.get("output_path")
        if output_path is None:
            raise ValueError(
                "Config must define results_root/experiment_id or legacy data.output_path."
            )
        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract_features] data_dir   = {data_dir}")
    print(f"[extract_features] output     = {out_path}")
    print(f"[extract_features] backbone   = {backbone_cfg.get('name')}:{backbone_cfg.get('variant')}")
    print(f"[extract_features] batch_size = {batch_size}")
    print(f"[extract_features] num_workers= {num_workers}")
    print(f"[extract_features] device     = {device}")
    print(f"[extract_features] img_size   = {img_size_cfg}")
    if checkpoint_path:
        print(f"[extract_features] checkpoint= {checkpoint_path}")

    features_dict = extract_features_from_folder(
        data_dir=data_dir,
        dino_size=dino_size,
        img_size=img_size_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        checkpoint_path=checkpoint_path,
        backbone_cfg=backbone_cfg,
    )

    # Ensure everything is NumPy / Python-native before saving
    to_save = {}
    for k, v in features_dict.items():
        if isinstance(v, torch.Tensor):
            to_save[k] = v.detach().cpu().numpy()
        else:
            to_save[k] = v

    np.savez_compressed(out_path, **to_save)

    feats = features_dict["features"]
    print(
        f"[extract_features] Saved features to {out_path} "
        f"(N={feats.shape[0]}, dim={feats.shape[1]})"
    )
    print(f"[extract_features] Keys: {list(features_dict.keys())}")

    if run_dir is not None:
        update_metrics(
            run_dir,
            "features",
            {
                "num_samples": int(feats.shape[0]),
                "feature_dim": int(feats.shape[1]),
                "backbone_name": backbone_cfg.get("name"),
                "backbone_variant": backbone_cfg.get("variant"),
                "dino_size": dino_size,
                "data_dir": data_dir,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[extract_features] ERROR: {e}", file=sys.stderr)
        raise
