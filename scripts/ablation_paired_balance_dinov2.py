#!/usr/bin/env python3
import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from dino_peft.datasets.droso_seg import DrosoSegDataset
from dino_peft.datasets.lucchi_seg import LucchiSegDataset
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.models.lora import apply_peft
from dino_peft.trainers.seg_trainer import SegTrainer
from dino_peft.utils.paths import resolve_run_dir, update_metrics
from dino_peft.utils.transforms import em_seg_transforms
from dino_peft.backbones import build_backbone, resolve_backbone_cfg, patch_tokens_to_grid


LUCCHI_CFG_CANDIDATES = [
    Path("configs/cluster/lucchi_dinov2_cluster.yaml"),
    Path("configs/mac/lucchi_dinov2_lora_mac.yaml"),
]
VNC_CFG_CANDIDATES = [
    Path("configs/cluster/droso_cluster.yaml"),
    Path("configs/mac/droso_lora_mac.yaml"),
]


class BalancedPairBatchSampler:
    """Yield 1 Lucchi + 1 VNC per batch; stop at the shorter list."""

    def __init__(self, lucchi_indices, vnc_indices, seed=0):
        self.lucchi_indices = list(lucchi_indices)
        self.vnc_indices = list(vnc_indices)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        lucchi = self.lucchi_indices[:]
        vnc = self.vnc_indices[:]
        rng.shuffle(lucchi)
        rng.shuffle(vnc)
        n = min(len(lucchi), len(vnc))
        for i in range(n):
            yield [lucchi[i], vnc[i]]
        self.epoch += 1

    def __len__(self):
        return min(len(self.lucchi_indices), len(self.vnc_indices))


def _pick_cfg_path(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {label} config found in: {candidates}")


def _load_cfg(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _dataset_params(cfg, dataset_type: str):
    params = dict((cfg.get("dataset") or {}).get("params") or {})
    if dataset_type == "lucchi":
        params.setdefault("recursive", False)
        params.setdefault("zfill_width", 4)
        params.setdefault("image_prefix", "mask")
    elif dataset_type == "droso":
        params.setdefault("recursive", True)
    return params


def _build_dataset(
    dataset_type: str,
    img_dir: str,
    mask_dir: str,
    img_size_cfg,
    transform,
    binarize: bool,
    binarize_threshold: int,
    params: dict,
):
    common = dict(
        img_size=img_size_cfg,
        to_rgb=True,
        transform=transform,
        binarize=binarize,
        binarize_threshold=binarize_threshold,
    )
    if dataset_type == "lucchi":
        return LucchiSegDataset(
            img_dir,
            mask_dir,
            recursive=bool(params.get("recursive", False)),
            zfill_width=int(params.get("zfill_width", 4)),
            image_prefix=params.get("image_prefix", "mask"),
            **common,
        )
    if dataset_type == "droso":
        return DrosoSegDataset(
            img_dir,
            mask_dir,
            recursive=bool(params.get("recursive", True)),
            mask_prefix=params.get("mask_prefix", ""),
            mask_suffix=params.get("mask_suffix", ""),
            **common,
        )
    raise ValueError(f"Unsupported dataset_type='{dataset_type}'")


def _split_pairs(ds, val_ratio: float, seed: int):
    n = len(ds)
    if n == 0:
        raise ValueError("Cannot split empty dataset.")
    n_val = int(round(n * val_ratio))
    if n > 1:
        n_val = max(1, min(n_val, n - 1))
    else:
        n_val = 0
    n_train = n - n_val
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    train_pairs = [ds.pairs[i] for i in train_idx]
    val_pairs = [ds.pairs[i] for i in val_idx]
    return train_pairs, val_pairs


def _make_subset_dataset(dataset_type, img_dir, mask_dir, img_size_cfg, transform, binarize, binarize_threshold, params, pairs):
    ds = _build_dataset(
        dataset_type,
        img_dir,
        mask_dir,
        img_size_cfg,
        transform,
        binarize,
        binarize_threshold,
        params,
    )
    ds.pairs = pairs
    return ds


def _build_training_cfg(base_cfg, out_dir: Path, variant: str, seed: int, lucchi_cfg, vnc_cfg):
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["use_lora"] = False
    cfg["lora"] = {"enabled": False}
    cfg["batch_size"] = 2
    cfg["epochs"] = 1000
    cfg["patience"] = 20
    cfg["lr"] = 5e-5
    cfg["weight_decay"] = 1e-4
    cfg["loss"] = "dice"
    cfg["val_ratio"] = 0.1
    cfg["split_seed"] = 42
    cfg["device"] = cfg.get("device", "auto")
    cfg["amp"] = False
    cfg["num_workers"] = int(cfg.get("num_workers", 4))

    cfg["train_img_dir"] = lucchi_cfg["train_img_dir"]
    cfg["train_mask_dir"] = lucchi_cfg["train_mask_dir"]
    cfg["test_img_dir"] = lucchi_cfg["test_img_dir"]
    cfg["test_mask_dir"] = lucchi_cfg["test_mask_dir"]
    cfg["dataset"] = lucchi_cfg.get("dataset", {"type": "lucchi", "params": {}})

    cfg["paired_lucchi"] = {
        "train_img_dir": lucchi_cfg["train_img_dir"],
        "train_mask_dir": lucchi_cfg["train_mask_dir"],
        "test_img_dir": lucchi_cfg["test_img_dir"],
        "test_mask_dir": lucchi_cfg["test_mask_dir"],
        "dataset": lucchi_cfg.get("dataset", {"type": "lucchi", "params": {}}),
    }
    cfg["paired_vnc"] = {
        "train_img_dir": vnc_cfg["train_img_dir"],
        "train_mask_dir": vnc_cfg["train_mask_dir"],
        "test_img_dir": vnc_cfg["test_img_dir"],
        "test_mask_dir": vnc_cfg["test_mask_dir"],
        "dataset": vnc_cfg.get("dataset", {"type": "droso", "params": {}}),
    }

    cfg["results_root"] = str(out_dir.parent)
    cfg["task_type"] = out_dir.name
    cfg["experiment_id"] = f"{variant}/seed_{seed}"
    return cfg


def _prepare_datasets(cfg, lucchi_cfg, vnc_cfg):
    img_size_cfg = cfg.get("img_size")
    binarize = bool(cfg.get("binarize", True))
    bin_thresh = int(cfg.get("binarize_threshold", 128))
    split_seed = int(cfg.get("split_seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.1))

    t_train = em_seg_transforms()
    t_val = em_seg_transforms()

    lucchi_params = _dataset_params(lucchi_cfg, "lucchi")
    vnc_params = _dataset_params(vnc_cfg, "droso")

    lucchi_base = _build_dataset(
        "lucchi",
        lucchi_cfg["train_img_dir"],
        lucchi_cfg["train_mask_dir"],
        img_size_cfg,
        transform=None,
        binarize=binarize,
        binarize_threshold=bin_thresh,
        params=lucchi_params,
    )
    vnc_base = _build_dataset(
        "droso",
        vnc_cfg["train_img_dir"],
        vnc_cfg["train_mask_dir"],
        img_size_cfg,
        transform=None,
        binarize=binarize,
        binarize_threshold=bin_thresh,
        params=vnc_params,
    )

    lucchi_train_pairs, lucchi_val_pairs = _split_pairs(lucchi_base, val_ratio, split_seed)
    vnc_train_pairs, vnc_val_pairs = _split_pairs(vnc_base, val_ratio, split_seed)

    lucchi_train_ds = _make_subset_dataset(
        "lucchi",
        lucchi_cfg["train_img_dir"],
        lucchi_cfg["train_mask_dir"],
        img_size_cfg,
        t_train,
        binarize,
        bin_thresh,
        lucchi_params,
        lucchi_train_pairs,
    )
    vnc_train_ds = _make_subset_dataset(
        "droso",
        vnc_cfg["train_img_dir"],
        vnc_cfg["train_mask_dir"],
        img_size_cfg,
        t_train,
        binarize,
        bin_thresh,
        vnc_params,
        vnc_train_pairs,
    )
    lucchi_val_ds = _make_subset_dataset(
        "lucchi",
        lucchi_cfg["train_img_dir"],
        lucchi_cfg["train_mask_dir"],
        img_size_cfg,
        t_val,
        binarize,
        bin_thresh,
        lucchi_params,
        lucchi_val_pairs,
    )
    vnc_val_ds = _make_subset_dataset(
        "droso",
        vnc_cfg["train_img_dir"],
        vnc_cfg["train_mask_dir"],
        img_size_cfg,
        t_val,
        binarize,
        bin_thresh,
        vnc_params,
        vnc_val_pairs,
    )

    train_ds = ConcatDataset([lucchi_train_ds, vnc_train_ds])
    val_ds = ConcatDataset([lucchi_val_ds, vnc_val_ds])
    return train_ds, val_ds, lucchi_train_ds, vnc_train_ds


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


def _eval_iouf(backbone, head, dataset, device, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=SegTrainer._pad_collate,
    )
    fg_inter = 0.0
    fg_union = 0.0
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            out = backbone(imgs)
            feats = patch_tokens_to_grid(out)
            logits = head(feats, masks.shape[-2:])
            pred = logits.argmax(1)
            pk_fg = pred > 0
            mk_fg = masks > 0
            fg_inter += (pk_fg & mk_fg).sum().item()
            fg_union += (pk_fg | mk_fg).sum().item()
    eps = 1e-7
    return float(fg_inter / (fg_union + eps))


def _write_run_metrics(run_dir: Path, metrics: dict):
    out_path = run_dir / "paired_balance_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    update_metrics(run_dir, "paired_balance", metrics)


def _write_summary_csv(out_dir: Path):
    out_dir = Path(out_dir)
    rows = []
    for path in out_dir.glob("*/*/paired_balance_metrics.json"):
        data = json.loads(path.read_text())
        rows.append(data)

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    summary_path = out_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant",
            "n",
            "lucchi_iouf_mean",
            "lucchi_iouf_std",
            "vnc_iouf_mean",
            "vnc_iouf_std",
            "macro_iouf_mean",
            "macro_iouf_std",
        ])
        for variant in sorted(by_variant.keys()):
            vals = by_variant[variant]
            lucchi = [v["lucchi_iouf"] for v in vals]
            vnc = [v["vnc_iouf"] for v in vals]
            macro = [v["macro_iouf"] for v in vals]
            writer.writerow([
                variant,
                len(vals),
                float(np.mean(lucchi)) if lucchi else float("nan"),
                float(np.std(lucchi, ddof=1)) if len(lucchi) > 1 else 0.0,
                float(np.mean(vnc)) if vnc else float("nan"),
                float(np.std(vnc, ddof=1)) if len(vnc) > 1 else 0.0,
                float(np.mean(macro)) if macro else float("nan"),
                float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
            ])


def _run_variant(variant, seed, out_dir: Path, base_cfg, lucchi_cfg, vnc_cfg):
    cfg = _build_training_cfg(base_cfg, out_dir, variant, seed, lucchi_cfg, vnc_cfg)
    run_dir = resolve_run_dir(cfg, cfg["task_type"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_ablation.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    trainer = SegTrainer(str(cfg_path))

    train_ds, val_ds, lucchi_train_ds, vnc_train_ds = _prepare_datasets(cfg, lucchi_cfg, vnc_cfg)
    pin = trainer.device.type == "cuda"

    if variant == "balanced":
        offset = len(lucchi_train_ds)
        lucchi_indices = list(range(len(lucchi_train_ds)))
        vnc_indices = [i + offset for i in range(len(vnc_train_ds))]
        sampler = BalancedPairBatchSampler(lucchi_indices, vnc_indices, seed=seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=cfg["num_workers"],
            pin_memory=pin,
            collate_fn=trainer._pad_collate,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=pin,
            collate_fn=trainer._pad_collate,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        collate_fn=trainer._pad_collate,
    )

    trainer.train_ds = train_ds
    trainer.val_ds = val_ds
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader

    trainer.train()

    ckpt_path = run_dir / "ckpts" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = trainer.device
    backbone, head = _load_eval_model(cfg, ckpt_path, device)

    img_size_cfg = cfg.get("img_size")
    binarize = bool(cfg.get("binarize", True))
    bin_thresh = int(cfg.get("binarize_threshold", 128))
    num_workers = int(cfg.get("num_workers", 4))
    t_eval = em_seg_transforms()

    lucchi_params = _dataset_params(lucchi_cfg, "lucchi")
    vnc_params = _dataset_params(vnc_cfg, "droso")

    lucchi_test = _build_dataset(
        "lucchi",
        lucchi_cfg["test_img_dir"],
        lucchi_cfg["test_mask_dir"],
        img_size_cfg,
        t_eval,
        binarize,
        bin_thresh,
        lucchi_params,
    )
    vnc_test = _build_dataset(
        "droso",
        vnc_cfg["test_img_dir"],
        vnc_cfg["test_mask_dir"],
        img_size_cfg,
        t_eval,
        binarize,
        bin_thresh,
        vnc_params,
    )

    lucchi_iouf = _eval_iouf(backbone, head, lucchi_test, device, num_workers)
    vnc_iouf = _eval_iouf(backbone, head, vnc_test, device, num_workers)
    macro_iouf = 0.5 * (lucchi_iouf + vnc_iouf)

    metrics = {
        "variant": variant,
        "seed": int(seed),
        "lucchi_iouf": float(lucchi_iouf),
        "vnc_iouf": float(vnc_iouf),
        "macro_iouf": float(macro_iouf),
    }
    _write_run_metrics(run_dir, metrics)
    _write_summary_csv(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results/ablation_paired_balance_dinov2")
    ap.add_argument("--variant", type=str, choices=["unbalanced", "balanced", "both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lucchi_cfg_path = _pick_cfg_path(LUCCHI_CFG_CANDIDATES, "lucchi")
    vnc_cfg_path = _pick_cfg_path(VNC_CFG_CANDIDATES, "vnc")
    lucchi_cfg = _load_cfg(lucchi_cfg_path)
    vnc_cfg = _load_cfg(vnc_cfg_path)

    base_cfg = _load_cfg(lucchi_cfg_path)

    variants = ["unbalanced", "balanced"] if args.variant == "both" else [args.variant]
    for variant in variants:
        _run_variant(variant, args.seed, out_dir, base_cfg, lucchi_cfg, vnc_cfg)


if __name__ == "__main__":
    main()
