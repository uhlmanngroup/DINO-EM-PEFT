#!/usr/bin/env python3
"""Run the paired balance ablation for DINOv2 segmentation.

Example (local):
    python scripts/ablation_paired_balance_dinov2.py \
      --cfg configs/cluster/paired_dinov2_paired_balance.yaml \
      --seed 0 \
      --variant both

Example (cluster):
    sbatch slurm/ablation_paired_balance_dinov2.sbatch
"""
import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.models.lora import apply_peft
from dino_peft.trainers.seg_trainer import SegTrainer
from dino_peft.utils.paths import resolve_run_dir, update_metrics
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.backbones import build_backbone, resolve_backbone_cfg, patch_tokens_to_grid


class BalancedPairBatchSampler:
    """Yield 1 Lucchi + 1 Droso per batch; stop at the shorter list."""

    def __init__(self, lucchi_indices, droso_indices, seed=0):
        self.lucchi_indices = list(lucchi_indices)
        self.droso_indices = list(droso_indices)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        lucchi = self.lucchi_indices[:]
        droso = self.droso_indices[:]
        rng.shuffle(lucchi)
        rng.shuffle(droso)
        n = min(len(lucchi), len(droso))
        for i in range(n):
            yield [lucchi[i], droso[i]]
        self.epoch += 1

    def __len__(self):
        return min(len(self.lucchi_indices), len(self.droso_indices))


def _load_cfg(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _get_paired_params(cfg: dict):
    params = dict((cfg.get("dataset") or {}).get("params") or {})
    params.setdefault("pair_mode", "stem")
    params.setdefault("recursive", False)
    params.setdefault("mask_prefix", "")
    params.setdefault("mask_suffix", "")
    return params


def _build_paired_dataset(cfg: dict, split: str, transform):
    split = split.lower()
    img_key = f"{split}_img_dir"
    mask_key = f"{split}_mask_dir"
    if img_key not in cfg or mask_key not in cfg:
        raise KeyError(f"Config missing '{img_key}' or '{mask_key}'")
    params = _get_paired_params(cfg)
    return PairedDirsSegDataset(
        image_dir=cfg[img_key],
        mask_dir=cfg[mask_key],
        img_size=cfg.get("img_size"),
        to_rgb=True,
        transform=transform,
        binarize=bool(cfg.get("binarize", True)),
        binarize_threshold=int(cfg.get("binarize_threshold", 128)),
        pair_mode=params.get("pair_mode", "stem"),
        mask_prefix=params.get("mask_prefix", ""),
        mask_suffix=params.get("mask_suffix", ""),
        recursive=bool(params.get("recursive", False)),
    )


def _partition_indices_by_name(pairs):
    lucchi_idx = []
    droso_idx = []
    unknown = []
    for i, (img_path, _) in enumerate(pairs):
        name = img_path.stem.lower()
        if "lucchi" in name:
            lucchi_idx.append(i)
        elif "droso" in name:
            droso_idx.append(i)
        else:
            unknown.append(img_path.stem)
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            "Could not assign some training samples to lucchi/droso by name. "
            f"Examples: {preview}"
        )
    if not lucchi_idx or not droso_idx:
        raise ValueError(
            f"Balanced sampling requires both lucchi and droso samples; got lucchi={len(lucchi_idx)} droso={len(droso_idx)}"
        )
    return lucchi_idx, droso_idx


def _build_training_cfg(base_cfg, variant: str, seed: int, out_root: Path | None, epochs_override: int | None):
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    if epochs_override is not None:
        cfg["epochs"] = int(epochs_override)

    dataset_type = str((cfg.get("dataset") or {}).get("type", "paired")).lower()
    if dataset_type != "paired":
        raise ValueError("This ablation expects dataset.type: paired")

    base_exp = str(cfg.get("experiment_id", "ablation_paired_balance_dinov2"))
    task_type = str(cfg.get("task_type", "seg"))
    results_root = Path(out_root) if out_root is not None else Path(cfg.get("results_root", "results"))
    cfg["results_root"] = str(results_root)
    cfg["task_type"] = task_type
    cfg["experiment_id"] = f"{base_exp}/{variant}/seed_{seed}"

    run_dir = results_root / task_type / cfg["experiment_id"]
    cfg["out_dir"] = str(run_dir)
    return cfg, results_root / task_type / base_exp


def _load_eval_model(cfg, ckpt_path: Path, device):
    backbone_cfg = resolve_backbone_cfg(cfg)
    backbone = build_backbone(backbone_cfg, device=device)
    apply_peft(
        backbone.model,
        cfg,
        run_dir=None,
        backbone_info=backbone_cfg,
        write_report=False,
    )

    head = SegHeadDeconv(
        in_ch=backbone.embed_dim,
        num_classes=cfg["num_classes"],
        n_ups=4,
        base_ch=512,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(ckpt["head"])
    lora_dict = ckpt.get("backbone_lora", {})
    if lora_dict:
        bb_state = backbone.model.state_dict()
        for k, v in lora_dict.items():
            if k in bb_state:
                bb_state[k] = v
        backbone.model.load_state_dict(bb_state, strict=False)

    backbone.eval()
    head.eval()
    return backbone, head


def _eval_iouf_by_domain(backbone, head, dataset, device, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=SegTrainer._pad_collate,
    )

    total_inter = 0.0
    total_union = 0.0
    lucchi_inter = 0.0
    lucchi_union = 0.0
    droso_inter = 0.0
    droso_union = 0.0
    counts = {"lucchi": 0, "droso": 0, "unknown": 0, "total": 0}

    with torch.no_grad():
        for imgs, masks, names in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = backbone(imgs)
            feats = patch_tokens_to_grid(out)
            logits = head(feats, masks.shape[-2:])
            pred = logits.argmax(1)

            pk_fg = pred > 0
            mk_fg = masks > 0
            total_inter += (pk_fg & mk_fg).sum().item()
            total_union += (pk_fg | mk_fg).sum().item()

            for j, name in enumerate(names):
                name_l = str(name).lower()
                if "lucchi" in name_l:
                    lucchi_inter += (pk_fg[j] & mk_fg[j]).sum().item()
                    lucchi_union += (pk_fg[j] | mk_fg[j]).sum().item()
                    counts["lucchi"] += 1
                elif "droso" in name_l:
                    droso_inter += (pk_fg[j] & mk_fg[j]).sum().item()
                    droso_union += (pk_fg[j] | mk_fg[j]).sum().item()
                    counts["droso"] += 1
                else:
                    counts["unknown"] += 1
                counts["total"] += 1

    eps = 1e-7
    total_iouf = float(total_inter / (total_union + eps))
    lucchi_iouf = float(lucchi_inter / (lucchi_union + eps)) if counts["lucchi"] else float("nan")
    droso_iouf = float(droso_inter / (droso_union + eps)) if counts["droso"] else float("nan")
    macro_iouf = float((lucchi_iouf + droso_iouf) / 2.0) if counts["lucchi"] and counts["droso"] else float("nan")

    return {
        "iouf_overall": total_iouf,
        "iouf_lucchi": lucchi_iouf,
        "iouf_droso": droso_iouf,
        "iouf_macro": macro_iouf,
        "count_total": counts["total"],
        "count_lucchi": counts["lucchi"],
        "count_droso": counts["droso"],
        "count_unknown": counts["unknown"],
    }


def _write_run_metrics(run_dir: Path, metrics: dict):
    out_path = run_dir / "paired_balance_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    update_metrics(run_dir, "paired_balance", metrics)


def _write_summary_csv(summary_root: Path):
    rows = []
    for path in summary_root.glob("**/paired_balance_metrics.json"):
        rows.append(json.loads(path.read_text()))

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    summary_path = summary_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant",
            "n",
            "iouf_overall_mean",
            "iouf_overall_std",
            "iouf_lucchi_mean",
            "iouf_lucchi_std",
            "iouf_droso_mean",
            "iouf_droso_std",
            "iouf_macro_mean",
            "iouf_macro_std",
        ])
        for variant in sorted(by_variant.keys()):
            vals = by_variant[variant]
            overall = [v["iouf_overall"] for v in vals]
            lucchi = [v["iouf_lucchi"] for v in vals]
            droso = [v["iouf_droso"] for v in vals]
            macro = [v["iouf_macro"] for v in vals]
            writer.writerow([
                variant,
                len(vals),
                float(np.mean(overall)) if overall else float("nan"),
                float(np.std(overall, ddof=1)) if len(overall) > 1 else 0.0,
                float(np.mean(lucchi)) if lucchi else float("nan"),
                float(np.std(lucchi, ddof=1)) if len(lucchi) > 1 else 0.0,
                float(np.mean(droso)) if droso else float("nan"),
                float(np.std(droso, ddof=1)) if len(droso) > 1 else 0.0,
                float(np.mean(macro)) if macro else float("nan"),
                float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
            ])


def _run_variant(variant, seed, base_cfg, out_root: Path | None, epochs_override: int | None):
    cfg, summary_root = _build_training_cfg(base_cfg, variant, seed, out_root, epochs_override)
    run_dir = resolve_run_dir(cfg, cfg["task_type"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_ablation.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    trainer = SegTrainer(str(cfg_path))

    if variant == "balanced":
        lucchi_idx, droso_idx = _partition_indices_by_name(trainer.train_ds.pairs)
        sampler = BalancedPairBatchSampler(lucchi_idx, droso_idx, seed=seed)
        trainer.train_loader = DataLoader(
            trainer.train_ds,
            batch_sampler=sampler,
            num_workers=cfg["num_workers"],
            pin_memory=(trainer.device.type == "cuda"),
            collate_fn=trainer._pad_collate,
        )

    trainer.train()

    ckpt_path = run_dir / "ckpts" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = trainer.device
    backbone, head = _load_eval_model(cfg, ckpt_path, device)

    test_ds = _build_paired_dataset(cfg, "test", em_seg_transforms())
    metrics = _eval_iouf_by_domain(backbone, head, test_ds, device, cfg["num_workers"])
    metrics.update({"variant": variant, "seed": int(seed)})

    _write_run_metrics(run_dir, metrics)
    _write_summary_csv(summary_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/cluster/paired_dinov2_paired_balance.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", type=str, choices=["unbalanced", "balanced", "both"], default="both")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    base_cfg = _load_cfg(Path(args.cfg))
    out_root = Path(args.out_dir) if args.out_dir else None

    variants = ["unbalanced", "balanced"] if args.variant == "both" else [args.variant]
    for variant in variants:
        _run_variant(variant, args.seed, base_cfg, out_root, args.epochs)


if __name__ == "__main__":
    main()
