#!/usr/bin/env python3
"""Evaluate EM segmentation checkpoints.

Example (local):
    python scripts/eval_em_seg.py --cfg configs/mac/lucchi_dinov2_lora_mac.yaml

Example (cluster):
    sbatch slurm/single_lucchi_dinov2.sbatch configs/cluster/lucchi_dinov2_cluster.yaml
"""

import argparse
import csv
import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_peft.datasets.paired_dirs_seg import PairedDirsSegDataset
from dino_peft.datasets.lucchi_seg import LucchiSegDataset
from dino_peft.datasets.droso_seg import DrosoSegDataset
from dino_peft.utils.transforms import em_seg_transforms, denorm_imagenet
from dino_peft.utils.viz import colorize_mask
from dino_peft.utils.plots import save_triptych_grid
from dino_peft.backbones import build_backbone, patch_tokens_to_grid, resolve_backbone_cfg
from dino_peft.models.head_seg1x1 import SegHeadDeconv
from dino_peft.models.lora import apply_peft
from dino_peft.utils.paths import setup_run_dir, update_metrics
from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG

def _filter_dataset_params(dataset_class, dataset_params: dict, dataset_type: str) -> dict:
    sig = inspect.signature(dataset_class.__init__)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    dropped = sorted(set(dataset_params) - allowed)
    if dropped:
        print(
            f"[eval_em_seg] Ignoring unsupported dataset.params for type='{dataset_type}': {dropped}"
        )
    return {k: v for k, v in dataset_params.items() if k in allowed}

def pad_collate(batch):
    imgs, masks, names = zip(*batch)
    max_h = max(img.shape[-2] for img in imgs)
    max_w = max(img.shape[-1] for img in imgs)
    padded_imgs = []
    padded_masks = []
    for img, mask in zip(imgs, masks):
        c = img.shape[0]
        h, w = img.shape[-2], img.shape[-1]
        pad_img = img.new_zeros((c, max_h, max_w))
        pad_img[:, :h, :w] = img
        padded_imgs.append(pad_img)

        mh, mw = mask.shape[-2], mask.shape[-1]
        pad_mask = mask.new_full((max_h, max_w), 0)
        pad_mask[:mh, :mw] = mask
        padded_masks.append(pad_mask)
    return torch.stack(padded_imgs), torch.stack(padded_masks), list(names)


def _resolve_img_size(cfg: dict):
    img_size_cfg = cfg.get("img_size")
    if img_size_cfg is None:
        img_size_cfg = deepcopy(DEFAULT_IMG_SIZE_CFG)
        cfg["img_size"] = deepcopy(img_size_cfg)
        print("[eval_em_seg] img_size not set in config; defaulting to longest_edge=1022.")
    return img_size_cfg


def build_dataset_from_cfg(cfg, split: str, transform):
    img_size_cfg = _resolve_img_size(cfg)
    dataset_cfg = cfg.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "lucchi")).lower()
    dataset_params = dict(dataset_cfg.get("params", {}))
    dataset_map = {
        "lucchi": LucchiSegDataset,
        "droso": DrosoSegDataset,
        "paired": PairedDirsSegDataset,
    }
    DatasetClass = dataset_map.get(dataset_type)
    if DatasetClass is None:
        raise ValueError(
            f"Unsupported dataset.type '{dataset_type}'. "
            "Use 'lucchi', 'droso', or 'paired'."
        )
    if dataset_type == "lucchi":
        dataset_params.setdefault("recursive", False)
        dataset_params.setdefault("zfill_width", 4)
        dataset_params.setdefault("image_prefix", "mask")
    elif dataset_type == "droso":
        dataset_params.setdefault("recursive", True)
    dataset_params = _filter_dataset_params(DatasetClass, dataset_params, dataset_type)

    kwargs = {
        "img_size": img_size_cfg,
        "to_rgb": True,
        "transform": transform,
        "binarize": bool(cfg.get("binarize", True)),
        "binarize_threshold": int(cfg.get("binarize_threshold", 128)),
    }
    kwargs.update(dataset_params)

    img_dir_key = f"{split}_img_dir"
    mask_dir_key = f"{split}_mask_dir"
    if img_dir_key not in cfg or mask_dir_key not in cfg:
        raise KeyError(f"Missing {img_dir_key}/{mask_dir_key} in config for split '{split}'")

    return DatasetClass(cfg[img_dir_key], cfg[mask_dir_key], **kwargs)


@torch.no_grad()
def eval_loop(
    backbone, head, loader, device, num_classes,
    out_dir: Path | None = None, ckpt_tag: str = "model",
    preview_n: int = 4,
    preview_seed: int | None = None,
    preview_mode: str = "random",
    preview_cols: int = 4,
):
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    gt_pix = np.zeros(num_classes, dtype=np.float64)
    pr_pix = np.zeros(num_classes, dtype=np.float64)

    # Foreground-collapsed accumulators (bg=0, fg=1..K-1)
    fg_inter = 0.0
    fg_union = 0.0
    fg_tp = 0.0
    fg_fp = 0.0
    fg_fn = 0.0
    fg_gt_pix = 0.0
    fg_pr_pix = 0.0

    prev_dir = None
    sample_indices = None
    if out_dir is not None:
        prev_dir = Path(out_dir) / "figs" / "eval_previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        preview_mode = (preview_mode or "random").lower()
        if preview_mode == "random":
            total = len(getattr(loader, "dataset", []))
            if total and preview_n > 0:
                import random

                rng = random.Random(preview_seed) if preview_seed is not None else random
                k = min(int(preview_n), int(total))
                sample_indices = set(rng.sample(range(total), k))

    preview_samples = []
    preview_group_idx = 0
    global_idx = 0
    for b, (imgs, masks, names) in enumerate(tqdm(loader, desc="eval")):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out = backbone(imgs)
        feats = patch_tokens_to_grid(out)
        logits = head(feats, masks.shape[-2:])
        pred = logits.argmax(1)

        # Per-class stats
        for k in range(num_classes):
            pk = (pred == k)
            mk = (masks == k)
            i = (pk & mk).sum().item()
            u = (pk | mk).sum().item()
            inter[k] += i
            union[k] += u
            tp[k] += i
            fp[k] += (pk & ~mk).sum().item()
            fn[k] += (~pk & mk).sum().item()
            gt_pix[k] += mk.sum().item()
            pr_pix[k] += pk.sum().item()

        # Foreground-collapsed (k>0)
        pk_fg = (pred > 0)
        mk_fg = (masks > 0)
        fg_inter += (pk_fg & mk_fg).sum().item()
        fg_union += (pk_fg | mk_fg).sum().item()
        fg_tp    += (pk_fg & mk_fg).sum().item()
        fg_fp    += (pk_fg & ~mk_fg).sum().item()
        fg_fn    += (~pk_fg & mk_fg).sum().item()
        fg_gt_pix += mk_fg.sum().item()
        fg_pr_pix += pk_fg.sum().item()
    
        # ---------------- Preview triptychs ----------------
        if prev_dir is not None:
            imgs_vis = denorm_imagenet(imgs.detach().cpu()).clamp(0, 1)
            gts_cpu = masks.detach().cpu()
            preds_cp = logits.detach().argmax(1).cpu()

            H, W = gts_cpu.shape[-2:]
            if imgs_vis.shape[-2:] != (H, W):
                imgs_vis = F.interpolate(
                    imgs_vis, size=(H, W), mode="bilinear", align_corners=False
                )

            pred_rgb = colorize_mask(preds_cp, num_classes)
            gt_rgb = colorize_mask(gts_cpu, num_classes)

            if preview_mode == "all":
                for j in range(imgs_vis.size(0)):
                    preview_samples.append(
                        {
                            "image": imgs_vis[j],
                            "gt": gt_rgb[j],
                            "pred": pred_rgb[j],
                            "name": str(names[j]),
                        }
                    )
                while len(preview_samples) >= preview_cols:
                    group = preview_samples[:preview_cols]
                    preview_samples = preview_samples[preview_cols:]
                    out_png = Path(prev_dir) / f"eval_triptych_{ckpt_tag}_g{preview_group_idx:04d}.png"
                    save_triptych_grid(
                        group,
                        out_path=str(out_png),
                        title=f"Evaluation — {ckpt_tag} — group {preview_group_idx}",
                    )
                    preview_group_idx += 1
            elif sample_indices:
                batch_size = imgs.size(0)
                capture = [
                    j for j in range(batch_size)
                    if (global_idx + j) in sample_indices
                ]
                for j in capture:
                    preview_samples.append(
                        {
                            "image": imgs_vis[j],
                            "gt": gt_rgb[j],
                            "pred": pred_rgb[j],
                            "name": str(names[j]),
                        }
                    )
        global_idx += imgs.size(0)

    if prev_dir is not None and preview_samples:
        if preview_mode == "all":
            out_png = Path(prev_dir) / f"eval_triptych_{ckpt_tag}_g{preview_group_idx:04d}.png"
            save_triptych_grid(
                preview_samples,
                out_path=str(out_png),
                title=f"Evaluation — {ckpt_tag} — group {preview_group_idx}",
            )
        else:
            out_png = Path(prev_dir) / f"eval_triptych_{ckpt_tag}_random.png"
            save_triptych_grid(
                preview_samples,
                out_path=str(out_png),
                title=f"Evaluation — {ckpt_tag} — {len(preview_samples)} samples",
            )

    eps = 1e-7
    iou  = inter / (union + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    iou_f  = fg_inter / (fg_union + eps)
    dice_f = (2 * fg_tp) / (2 * fg_tp + fg_fp + fg_fn + eps)

    # Sanity notes (optional)
    if fg_gt_pix == 0:
        print("[WARN] test set has ZERO foreground pixels — IoU_f/Dice_f not meaningful.")

    return iou, dice, iou_f, dice_f


# --- replace your `main()` with this version ---

def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _build_dataset(cfg, split: str, transform):
    img_size_cfg = _resolve_img_size(cfg)
    split = split.lower()
    img_key = f"{split}_img_dir"
    mask_key = f"{split}_mask_dir"
    if img_key not in cfg or mask_key not in cfg:
        raise KeyError(f"Config missing '{img_key}' or '{mask_key}' for eval split='{split}'")

    dataset_cfg = cfg.get("dataset", {})
    dataset_type = str(dataset_cfg.get("type", "lucchi")).lower()
    params = dataset_cfg.get("params") or {}
    common = dict(
        image_dir=cfg[img_key],
        mask_dir=cfg[mask_key],
        img_size=img_size_cfg,
        to_rgb=True,
        transform=transform,
        binarize=bool(cfg.get("binarize", True)),
        binarize_threshold=int(cfg.get("binarize_threshold", 128)),
    )
    if dataset_type == "paired":
        return PairedDirsSegDataset(
            **common,
            pair_mode=params.get("pair_mode", "stem"),
            mask_prefix=params.get("mask_prefix", ""),
            mask_suffix=params.get("mask_suffix", ""),
            recursive=bool(params.get("recursive", False)),
        )
    elif dataset_type == "lucchi":
        return LucchiSegDataset(
            **common,
            recursive=bool(params.get("recursive", False)),
            zfill_width=int(params.get("zfill_width", 4)),
            image_prefix=params.get("image_prefix", "mask"),
        )
    elif dataset_type == "droso":
        return DrosoSegDataset(
            **common,
            recursive=bool(params.get("recursive", True)),
            mask_prefix=params.get("mask_prefix", ""),
            mask_suffix=params.get("mask_suffix", ""),
        )
    else:
        raise ValueError(
            f"Unknown dataset.type '{dataset_type}'. "
            "Use 'lucchi', 'droso', or 'paired' and attach details under dataset.params."
        )

def best_checkpoint(run_dir) -> Path:
    """Return ckpts/best_model.pt or raise if missing."""
    p = Path(run_dir) / "ckpts" / "best_model.pt"
    if not p.exists():
        raise FileNotFoundError(f"best_model.pt not found in {p.parent}")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        required=True,
        help="Training YAML or saved config_used.yaml / config_runtime.yaml",
    )
    ap.add_argument("--ckpt", default="", help="Optional checkpoint path; if empty, auto-pick latest in run directory")
    ap.add_argument("--out_csv", default="", help="Optional metrics output; if empty, write run_dir/metrics_test.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    task_type = cfg.get("task_type", "seg")
    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(
            cfg,
            task_type=task_type,
            subdirs=("figs", "figs/eval_previews"),
            save_config=False,
        )
    else:
        run_dir = Path(cfg["out_dir"]).expanduser()
        (run_dir / "figs" / "eval_previews").mkdir(parents=True, exist_ok=True)

    # auto-pick checkpoint if not provided
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else best_checkpoint(run_dir)
    if args.ckpt:
        run_dir = ckpt_path.parent.parent
        (run_dir / "figs" / "eval_previews").mkdir(parents=True, exist_ok=True)
    out_csv   = Path(args.out_csv).expanduser() if args.out_csv else (run_dir / "metrics_test.csv")

    print(f"[eval_em_seg] run_dir={run_dir}")
    print(f"[eval_em_seg] ckpt_path={ckpt_path}")

    device = _pick_device()

    # load checkpoint (LoRA + head)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("cfg", {}) or {}
    eval_cfg = ckpt_cfg if ckpt_cfg else cfg
    if ckpt_cfg:
        print("[eval_em_seg] using cfg from checkpoint for model/data settings.")
    lora_cfg_source = ckpt_cfg if ckpt_cfg else cfg

    # model
    backbone_cfg = resolve_backbone_cfg(eval_cfg)
    bb = build_backbone(backbone_cfg, device=device)
    audit = apply_peft(
        bb.model,
        lora_cfg_source,
        run_dir=run_dir,
        backbone_info=backbone_cfg,
        write_report=False,
    )
    bb.to(device)

    # dataset (test split from eval cfg)
    t = em_seg_transforms()
    ds = build_dataset_from_cfg(eval_cfg, split="test", transform=t)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    head = SegHeadDeconv(
        in_ch=bb.embed_dim,
        num_classes=eval_cfg["num_classes"],
        n_ups=4,
        base_ch=512,
    ).to(device)
    head.load_state_dict(ckpt["head"])
    backbone_full = ckpt.get("backbone")
    if backbone_full:
        result = bb.model.load_state_dict(backbone_full, strict=False)
        if result.missing_keys:
            print(f"[eval_em_seg][warn] Missing backbone keys: {result.missing_keys[:10]}")
        if result.unexpected_keys:
            print(f"[eval_em_seg][warn] Unexpected backbone keys: {result.unexpected_keys[:10]}")
    else:
        bb_state = bb.model.state_dict()
        lora_dict = ckpt.get("backbone_lora", {})
        if audit is None and lora_dict:
            print("[eval_em_seg][warn] Checkpoint has LoRA weights, but LoRA is disabled in eval config.")
        if audit is not None and not lora_dict:
            print("[eval_em_seg][warn] LoRA enabled but checkpoint missing backbone_lora weights.")
        matched = 0
        for k, v in lora_dict.items():
            if k in bb_state:
                bb_state[k] = v
                matched += 1
        if lora_dict and matched == 0:
            print("[eval_em_seg][warn] LoRA weights did not match any backbone keys; check backbone variant/config.")
        bb.model.load_state_dict(bb_state, strict=False)

    bb.eval(); head.eval()
    preview_n = int(eval_cfg.get("eval_preview_n", 4))
    preview_seed = eval_cfg.get("eval_preview_seed")
    if preview_seed is not None:
        preview_seed = int(preview_seed)
    preview_mode = str(eval_cfg.get("eval_preview_mode", "random")).lower()
    preview_cols = int(eval_cfg.get("eval_preview_cols", 4))
    if preview_cols < 1:
        preview_cols = 4

    # --- compute metrics ---
    iou, dice, iou_f, dice_f = eval_loop(
        bb,
        head,
        loader,
        device,
        eval_cfg["num_classes"],
        out_dir=run_dir,
        preview_n=preview_n,
        preview_seed=preview_seed,
        preview_mode=preview_mode,
        preview_cols=preview_cols,
    )

    # --- write CSV before updating metrics.json ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","IoU","Dice"])
        for k,(i,d) in enumerate(zip(iou, dice)):
            w.writerow([k, f"{i:.6f}", f"{d:.6f}"])
        w.writerow(["foreground", f"{iou_f:.6f}", f"{dice_f:.6f}"])
        w.writerow(["mean", f"{iou.mean():.6f}", f"{dice.mean():.6f}"])
    print(f"Saved metrics → {out_csv}")
    update_metrics(
        run_dir,
        "eval",
        {
            "mean_iou": float(iou.mean()),
            "mean_dice": float(dice.mean()),
            "foreground_iou": float(iou_f),
            "foreground_dice": float(dice_f),
            "num_classes": int(eval_cfg["num_classes"]),
        },
    )

if __name__ == "__main__":
    main()
