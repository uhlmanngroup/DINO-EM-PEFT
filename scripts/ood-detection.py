#!/usr/bin/env python3
"""Mahalanobis OOD detection on DINO/PEFT embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # force headless backend for batch use
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:  # pragma: no cover - optional dependency
    from sklearn.covariance import LedoitWolf, OAS

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    LedoitWolf = None
    OAS = None
    _HAVE_SKLEARN = False

from dino_peft.analysis.dimred import l2_normalize, run_pca
from dino_peft.models.backbone_dinov2 import DINOv2FeatureExtractor
from dino_peft.models.lora import inject_lora
from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG
from dino_peft.utils.paths import setup_run_dir, update_metrics, write_run_info
from dino_peft.utils.transforms import em_dino_unsup_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # load incomplete TIFFs instead of crashing

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CACHE_VERSION = 1


# Small containers describing dataset runtime settings.
@dataclass
class DatasetSpec:
    label: str
    root: Path
    paths: List[str]
    img_size: Any
    batch_size: int
    num_workers: int
    recursive: bool
    filters: Dict[str, Any]
    sequence_regex: Optional[str]
    sequence_group: Optional[str]


# Pack embeddings + metadata to keep arrays/paths aligned.
@dataclass
class DatasetEmbeddings:
    label: str
    embeddings: np.ndarray
    paths: List[str]
    sequence_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="OOD detection with Mahalanobis distances on DINO embeddings."
    )
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config.")
    return ap.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file {path} is empty or invalid.")
    return cfg


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str | None) -> torch.device:
    device_str = (device_str or "auto").lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_str == "mps":
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but backend is unavailable.")
        return torch.device("mps")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Invalid device '{device_str}'.")


def _normalize_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _read_file_list(list_path: str | None, root: Path) -> Optional[List[Path]]:
    if not list_path:
        return None
    entries = []
    file_path = Path(list_path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File list not found: {file_path}")
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (root / candidate).expanduser()
        entries.append(candidate)
    return entries


# Collect candidate image paths with optional glob/regex filters.
def discover_images(entry: Dict[str, Any], label: str) -> Tuple[List[str], Dict[str, Any]]:
    path_key = entry.get("path") or entry.get("data_dir") or entry.get("root")
    if not path_key:
        raise ValueError(f"Config for dataset '{label}' must define path/data_dir.")
    root = Path(path_key).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root not found for {label}: {root}")

    recursive = bool(entry.get("recursive", True))
    pattern = "**/*" if recursive else "*"
    listed = _read_file_list(entry.get("file_list"), root)
    if listed is not None:
        files = [p for p in listed if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        files = [
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]

    globs = _normalize_list(entry.get("file_glob") or entry.get("glob"))
    if globs:
        import fnmatch

        def _match_glob(p: Path) -> bool:
            rel = None
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.name
            for pat in globs:
                if fnmatch.fnmatch(rel, pat):
                    return True
            return False

        files = [p for p in files if _match_glob(p)]

    regex_pattern = entry.get("file_regex")
    if regex_pattern:
        regex = re.compile(regex_pattern)
        files = [p for p in files if regex.search(p.as_posix())]

    limit = entry.get("max_images") or entry.get("limit")
    if limit:
        limit = int(limit)
        if limit > 0:
            files = files[:limit]

    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found for dataset '{label}' under {root}.")

    filters = {
        "recursive": recursive,
        "file_list": entry.get("file_list"),
        "file_glob": globs,
        "file_regex": regex_pattern,
        "limit": limit,
    }
    return [str(p) for p in files], filters


# Thin Dataset wrapper that loads one image per path.
class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path)
        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
        except Exception:
            pass
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path


# Pad batch to max height/width (preserves original aspect).
def pad_collate(batch):
    images, paths = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded.append(F.pad(img, (0, pad_w, 0, pad_h)))
    return torch.stack(padded, dim=0), list(paths)


# Normalize config inputs into a DatasetSpec.
def build_dataset_spec(
    label: str,
    entry: Dict[str, Any],
    default_img_size: Any,
    default_batch: int,
    default_workers: int,
) -> DatasetSpec:
    paths, filters = discover_images(entry, label=label)
    img_size = entry.get("img_size", default_img_size)
    batch_size = int(entry.get("batch_size", default_batch))
    num_workers = int(entry.get("num_workers", default_workers))
    sequence_regex = entry.get("sequence_regex")
    sequence_group = entry.get("sequence_group")
    recursive = bool(entry.get("recursive", True))
    root_path = Path(entry.get("path") or entry.get("data_dir") or entry.get("root")).expanduser()
    return DatasetSpec(
        label=label,
        root=root_path,
        paths=paths,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        recursive=recursive,
        filters=filters,
        sequence_regex=sequence_regex,
        sequence_group=sequence_group,
    )


# Build DINO backbone and optionally inject LoRA weights.
def load_backbone(model_cfg: Dict[str, Any], device: torch.device) -> DINOv2FeatureExtractor:
    dino_size = model_cfg.get("dino_size", "base")
    checkpoint = model_cfg.get("checkpoint")
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    if checkpoint_path and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    backbone = DINOv2FeatureExtractor(size=dino_size, device=device)
    backbone.eval()

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        use_lora = bool(ckpt_cfg.get("use_lora", ckpt_cfg.get("enable_lora", True)))
        lora_rank = int(ckpt_cfg.get("lora_rank", 0) or 0)
        lora_alpha = int(ckpt_cfg.get("lora_alpha", lora_rank))
        lora_targets = ckpt_cfg.get("lora_targets", ["attn.qkv", "attn.proj"])
        if use_lora and lora_rank > 0:
            inject_lora(
                backbone.vit,
                target_substrings=lora_targets,
                r=lora_rank,
                alpha=lora_alpha if lora_alpha > 0 else lora_rank,
            )
            lora_state = ckpt.get("backbone_lora") or {}
            if not lora_state:
                raise RuntimeError("Checkpoint missing backbone_lora weights.")
            state = backbone.state_dict()
            for key, tensor in lora_state.items():
                if key in state:
                    state[key] = tensor
            backbone.load_state_dict(state, strict=False)
        else:
            print("[ood] Checkpoint has no LoRA weights; using base backbone.")
    return backbone


def _compute_paths_digest(paths: Sequence[str]) -> str:
    joined = "\n".join(paths).encode("utf-8")
    return md5(joined).hexdigest()


# Helper for stable cache metadata serialization.
def _metadata_repr(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


# Describe inputs so cached embeddings can be reused safely.
def build_cache_metadata(
    spec: DatasetSpec,
    model_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    paths: Sequence[str],
) -> Dict[str, Any]:
    checkpoint = model_cfg.get("checkpoint")
    pooling = model_cfg.get("pooling", "mean")
    meta = {
        "cache_version": CACHE_VERSION,
        "dataset_label": spec.label,
        "dataset_root": str(spec.root),
        "dino_size": model_cfg.get("dino_size", "base"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "img_size": _metadata_repr(spec.img_size),
        "pooling": pooling,
        "l2_normalize": bool(model_cfg.get("l2_normalize", False)),
        "paths_digest": _compute_paths_digest(paths),
        "filters": _metadata_repr(spec.filters),
        "transform": "em_dino_unsup_transforms",
        "device": runtime_cfg.get("device", "auto"),
    }
    return meta


# Embedding caches live under run_dir/embeddings.
def _cache_paths(run_dir: Path) -> Path:
    cache_dir = run_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Fast path: reuse saved embeddings when metadata still matches.
def load_cached_embeddings(cache_dir: Path, label: str, expected_meta: Dict[str, Any]) -> Optional[Tuple[np.ndarray, List[str]]]:
    emb_path = cache_dir / f"{label}_embeddings.npz"
    meta_path = cache_dir / f"{label}_meta.json"
    if not emb_path.is_file() or not meta_path.is_file():
        return None
    try:
        stored_meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    for key, value in expected_meta.items():
        if stored_meta.get(key) != value:
            return None
    with np.load(emb_path, allow_pickle=True) as data:
        embeddings = np.asarray(data["embeddings"]).astype(np.float32, copy=False)
        paths = list(data["paths"].tolist())
    if embeddings.shape[0] != len(paths):
        return None
    return embeddings, paths


# Persist embeddings + metadata for future runs.
def save_cached_embeddings(
    cache_dir: Path,
    label: str,
    embeddings: np.ndarray,
    paths: Sequence[str],
    metadata: Dict[str, Any],
) -> None:
    emb_path = cache_dir / f"{label}_embeddings.npz"
    meta_path = cache_dir / f"{label}_meta.json"
    np.savez_compressed(
        emb_path,
        embeddings=embeddings.astype(np.float32, copy=False),
        paths=np.array(paths, dtype=object),
    )
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


# Optional per-sample sequence metadata extracted via regex.
def _extract_sequence_ids(paths: Sequence[str], pattern: Optional[str], group: Optional[str]) -> np.ndarray:
    seq = np.full(len(paths), np.nan, dtype=float)
    if not pattern:
        return seq
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid sequence_regex '{pattern}': {exc}") from exc
    for idx, path in enumerate(paths):
        match = regex.search(path)
        if not match:
            continue
        if group is None:
            capture = match.group(1) if match.groups() else match.group(0)
        else:
            capture = match.group(group)
        try:
            seq[idx] = float(capture)
        except Exception:
            continue
    return seq


# Full embedding pass with caching + transforms.
@torch.no_grad()
def extract_embeddings_for_dataset(
    spec: DatasetSpec,
    model: DINOv2FeatureExtractor,
    device: torch.device,
    model_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    run_dir: Path,
    cache_enabled: bool,
) -> DatasetEmbeddings:
    cache_dir = _cache_paths(run_dir)
    metadata = build_cache_metadata(spec, model_cfg, runtime_cfg, spec.paths)
    if cache_enabled:
        cached = load_cached_embeddings(cache_dir, spec.label, metadata)
        if cached is not None:
            embeddings, cached_paths = cached
            print(f"[ood] Using cached embeddings for {spec.label} ({len(cached_paths)} samples).")
            seq = _extract_sequence_ids(cached_paths, spec.sequence_regex, spec.sequence_group)
            return DatasetEmbeddings(
                label=spec.label,
                embeddings=embeddings,
                paths=cached_paths,
                sequence_ids=seq,
            )

    transform = em_dino_unsup_transforms(img_size=spec.img_size)
    dataset = ImagePathDataset(spec.paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=spec.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate,
    )

    pooling = model_cfg.get("pooling", "mean").lower()
    l2_flag = bool(model_cfg.get("l2_normalize", False))
    desc = f"{spec.label}_embed"
    vectors = []
    ordered_paths: List[str] = []
    for imgs, paths in tqdm(loader, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)
        if feats.ndim != 4:
            raise RuntimeError(f"Backbone output expected 4D, got {tuple(feats.shape)}")
        if pooling == "mean":
            pooled = feats.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unsupported pooling '{pooling}'. Use 'mean'.")
        vectors.append(pooled.detach().cpu().numpy())
        ordered_paths.extend(paths)

    embeddings = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    if l2_flag:
        embeddings = l2_normalize(embeddings)
    seq = _extract_sequence_ids(ordered_paths, spec.sequence_regex, spec.sequence_group)
    if cache_enabled:
        save_cached_embeddings(cache_dir, spec.label, embeddings, ordered_paths, metadata)
    return DatasetEmbeddings(
        label=spec.label,
        embeddings=embeddings,
        paths=ordered_paths,
        sequence_ids=seq,
    )


# Estimate covariance/precision with shrinkage or diag regularization.
def fit_covariance(
    embeddings: np.ndarray,
    method: str,
    reg_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float64)
    method = method.lower()
    if method == "ledoit-wolf" and _HAVE_SKLEARN:
        estimator = LedoitWolf(store_precision=True, assume_centered=False)
        estimator.fit(x)
        cov = estimator.covariance_
        precision = estimator.precision_
        return cov.astype(np.float64), precision.astype(np.float64)
    if method == "oas" and _HAVE_SKLEARN:
        estimator = OAS(store_precision=True, assume_centered=False)
        estimator.fit(x)
        cov = estimator.covariance_
        precision = estimator.precision_
        return cov.astype(np.float64), precision.astype(np.float64)
    cov = np.cov(x, rowvar=False)
    if reg_eps > 0.0:
        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * reg_eps
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov)
    return cov.astype(np.float64), precision.astype(np.float64)


# Classic Mahalanobis distance computation in embedding space.
def mahalanobis_distances(points: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centered = points - mean
    m = np.einsum("ij,jk,ik->i", centered, precision, centered, optimize=True)
    m = np.maximum(m, 0.0)
    return np.sqrt(m.astype(np.float64))


# Convenience stats for summary.json.
def summarize_distances(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "min", "max", "q95", "q99")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def _top_k(indices: np.ndarray, paths: Sequence[str], distances: np.ndarray, squared: np.ndarray, dataset_labels: Sequence[str], k: int) -> List[Dict[str, Any]]:
    top = []
    for idx in indices[:k]:
        top.append(
            {
                "filepath": paths[idx],
                "dataset": dataset_labels[idx],
                "distance": float(distances[idx]),
                "distance_squared": float(squared[idx]),
            }
        )
    return top


# Write summary JSON (indent + stable ordering).
def save_summary(
    out_path: Path,
    payload: Dict[str, Any],
) -> None:
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# Persist per-sample metrics (uses pandas if available, else csv).
def write_distances_csv(
    csv_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if pd is not None:
        df = pd.DataFrame(rows)
        df.sort_values(by="distance", ascending=False, inplace=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.to_csv(csv_path, index=False)
        return
    rows.sort(key=lambda r: r["distance"], reverse=True)
    header = ["rank", "filepath", "dataset", "distance", "distance_squared", "is_outlier", "sequence_id"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "filepath": row["filepath"],
                    "dataset": row["dataset"],
                    "distance": f"{row['distance']:.8f}",
                    "distance_squared": f"{row['distance_squared']:.8f}",
                    "is_outlier": int(row["is_outlier"]),
                    "sequence_id": "" if np.isnan(row["sequence_id"]) else row["sequence_id"],
                }
            )


# Create plots directory once.
def ensure_plot_dir(run_dir: Path) -> Path:
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return plots


# ID vs target histogram with threshold marker.
def plot_histogram(id_dist: np.ndarray, target_dist: np.ndarray, threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = max(20, min(80, int(math.sqrt(len(id_dist) + len(target_dist)))))
    ax.hist(id_dist, bins=bins, alpha=0.6, label="ID", color="tab:blue")
    ax.hist(target_dist, bins=bins, alpha=0.6, label="Target", color="tab:orange")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Distance histogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# Violin/box hybrid across the two splits.
def plot_violin(id_dist: np.ndarray, target_dist: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [id_dist, target_dist]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc, color in zip(parts["bodies"], ("tab:blue", "tab:orange")):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax.set_xticks([1, 2], ["ID", "Target"])
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Distance distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# 2-view PCA projection: dataset colors + distance coloring.
def plot_pca_scatter(
    embeddings: np.ndarray,
    split_index: int,
    distances: np.ndarray,
    outliers: np.ndarray,
    run_dir: Path,
    seed: int,
    mean_vec: np.ndarray,
) -> None:
    total = embeddings.shape[0]
    if total < 3 or embeddings.shape[1] < 2:
        return
    try:
        pca, coords = run_pca(embeddings, n_components=2, random_state=seed, l2norm=False)
    except ValueError:
        return
    plots_dir = ensure_plot_dir(run_dir)
    id_coords = coords[:split_index]
    target_coords = coords[split_index:]

    def _plot_dataset():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(id_coords[:, 0], id_coords[:, 1], c="tab:blue", s=14, alpha=0.7, label="ID")
        ax.scatter(target_coords[:, 0], target_coords[:, 1], c="tab:orange", s=14, alpha=0.7, label="Target")
        out_xy = coords[outliers]
        if out_xy.size > 0:
            ax.scatter(
                out_xy[:, 0],
                out_xy[:, 1],
                facecolors="none",
                edgecolors="red",
                s=48,
                linewidths=1.2,
                label="Outliers",
            )
        mean_proj = pca.transform(mean_vec.reshape(1, -1))
        ax.scatter(mean_proj[:, 0], mean_proj[:, 1], c="black", marker="*", s=120, label="ID mean")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("PCA scatter (dataset labels)")
        fig.tight_layout()
        fig.savefig(plots_dir / "pca_dataset.png", dpi=320)
        plt.close(fig)

    def _plot_distance():
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=distances, cmap="viridis", s=18, alpha=0.85)
        fig.colorbar(sc, ax=ax, label="Mahalanobis distance")
        out_xy = coords[outliers]
        if out_xy.size > 0:
            ax.scatter(
                out_xy[:, 0],
                out_xy[:, 1],
                facecolors="none",
                edgecolors="red",
                s=52,
                linewidths=1.2,
                label="Outliers",
            )
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA scatter (distance coloring)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "pca_distance.png", dpi=320)
        plt.close(fig)

    _plot_dataset()
    _plot_distance()


# Visualize largest target outliers as horizontal bars.
def plot_topk_bar(paths: Sequence[str], distances: np.ndarray, k: int, out_path: Path) -> None:
    if not paths:
        return
    k = min(k, len(paths))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * k)))
    labels = [Path(p).name for p in paths[:k]]
    vals = distances[:k]
    y = np.arange(k)
    ax.barh(y, vals, color="tab:red", alpha=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mahalanobis distance")
    ax.set_title("Top target outliers")
    for idx, val in enumerate(vals):
        ax.text(val, idx, f"{val:.2f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


# Distance trend along dataset iteration order.
def plot_distance_index(
    id_dist: np.ndarray,
    target_dist: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(id_dist)), id_dist, label="ID", alpha=0.9)
    ax.plot(np.arange(len(target_dist)), target_dist, label="Target", alpha=0.9)
    ax.set_xlabel("Dataset index")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Distance vs. order")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


# Show relation between regex-derived sequence ids and distances.
def plot_distance_vs_sequence(
    sequence_ids: np.ndarray,
    distances: np.ndarray,
    out_path: Path,
) -> None:
    valid = ~np.isnan(sequence_ids)
    if not valid.any():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sequence_ids[valid], distances[valid], c=distances[valid], cmap="viridis", s=20, alpha=0.9)
    ax.set_xlabel("Sequence id")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Distance vs. sequence id")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


# Save an inspection grid of top-K outlier images.
def save_outlier_grid(paths: Sequence[str], distances: np.ndarray, out_path: Path, cols: int = 4) -> None:
    if not paths:
        return
    cols = max(1, cols)
    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    for idx, ax in enumerate(axes.flat):
        if idx >= len(paths):
            ax.axis("off")
            continue
        path = paths[idx]
        try:
            img = Image.open(path)
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception:
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{Path(path).name}\n{distances[idx]:.3f}", fontsize=9)
    fig.suptitle("Top target outliers", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg).expanduser()
    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    ood_cfg = cfg.get("ood", cfg.get("ood_detection", {}))

    seed = int(runtime_cfg.get("seed", 0))
    set_global_seed(seed)
    device = resolve_device(runtime_cfg.get("device", "auto"))

    img_size_cfg = runtime_cfg.get("img_size", DEFAULT_IMG_SIZE_CFG)
    batch_size = int(runtime_cfg.get("batch_size", 8))
    num_workers = int(runtime_cfg.get("num_workers", 2))

    id_entry = data_cfg.get("id")
    target_entry = data_cfg.get("target")
    if id_entry is None or target_entry is None:
        raise ValueError("Config must define data.id and data.target sections.")

    # Resolve dataset configs and keep run metadata consistent.
    id_spec = build_dataset_spec("id", id_entry, img_size_cfg, batch_size, num_workers)
    target_spec = build_dataset_spec("target", target_entry, img_size_cfg, batch_size, num_workers)

    task_type = cfg.get("task_type", "ood-detection")
    run_dir = setup_run_dir(cfg, task_type=task_type, subdirs=("plots", "embeddings"))
    write_run_info(
        run_dir,
        {
            "task_type": task_type,
            "device": str(device),
            "seed": seed,
            "id_root": str(id_spec.root),
            "target_root": str(target_spec.root),
            "checkpoint": model_cfg.get("checkpoint"),
        },
    )

    print(f"[ood] Run directory: {run_dir}")
    print(f"[ood] Device: {device}")
    print(f"[ood] ID samples: {len(id_spec.paths)} | Target samples: {len(target_spec.paths)}")

    backbone = load_backbone(model_cfg, device)
    cache_enabled = bool(ood_cfg.get("cache_embeddings", True))

    # Embed each dataset (re-using caches when possible).
    id_emb = extract_embeddings_for_dataset(
        id_spec,
        backbone,
        device,
        model_cfg,
        runtime_cfg,
        run_dir,
        cache_enabled=cache_enabled,
    )
    target_emb = extract_embeddings_for_dataset(
        target_spec,
        backbone,
        device,
        model_cfg,
        runtime_cfg,
        run_dir,
        cache_enabled=cache_enabled,
    )

    if id_emb.embeddings.shape[1] != target_emb.embeddings.shape[1]:
        raise ValueError("ID and target embeddings have mismatched dimensions.")

    mean_vec = id_emb.embeddings.mean(axis=0)
    cov_cfg = ood_cfg.get("covariance", {})
    method = cov_cfg.get("method", "ledoit-wolf")
    reg_eps = float(cov_cfg.get("reg_eps", 1e-4))
    cov, precision = fit_covariance(id_emb.embeddings, method, reg_eps)

    id_dist = mahalanobis_distances(id_emb.embeddings, mean_vec, precision)
    target_dist = mahalanobis_distances(target_emb.embeddings, mean_vec, precision)
    id_sq = id_dist ** 2
    target_sq = target_dist ** 2

    threshold_cfg = ood_cfg.get("threshold", {})
    quantile = float(threshold_cfg.get("quantile", 0.95))
    threshold = float(np.quantile(id_dist, quantile))
    id_outliers = id_dist > threshold
    target_outliers = target_dist > threshold

    summary = {
        "embedding_dim": int(id_emb.embeddings.shape[1]),
        "num_id": int(len(id_dist)),
        "num_target": int(len(target_dist)),
        "covariance_method": method,
        "cov_reg_eps": reg_eps,
        "threshold": {
            "method": "quantile",
            "quantile": quantile,
            "value": threshold,
        },
        "id_outlier_rate": float(id_outliers.mean()) if len(id_outliers) > 0 else 0.0,
        "target_outlier_rate": float(target_outliers.mean()) if len(target_outliers) > 0 else 0.0,
        "id_stats": summarize_distances(id_dist),
        "target_stats": summarize_distances(target_dist),
    }

    combined_paths = id_emb.paths + target_emb.paths
    combined_dist = np.concatenate([id_dist, target_dist], axis=0)
    combined_sq = np.concatenate([id_sq, target_sq], axis=0)
    combined_labels = ["id"] * len(id_dist) + ["target"] * len(target_dist)
    combined_outliers = np.concatenate([id_outliers, target_outliers], axis=0)
    order = np.argsort(combined_dist)[::-1]
    top_k = int(ood_cfg.get("topk", 24))
    summary["top_outliers_overall"] = _top_k(order, combined_paths, combined_dist, combined_sq, combined_labels, top_k)
    target_order = np.argsort(target_dist)[::-1]
    summary["top_outliers_target"] = _top_k(
        target_order,
        target_emb.paths,
        target_dist,
        target_sq,
        ["target"] * len(target_dist),
        top_k,
    )

    summary_path = run_dir / "summary.json"
    save_summary(summary_path, summary)

    rows: List[Dict[str, Any]] = []
    for label, dist, dist_sq, paths, seq, flags in (
        ("id", id_dist, id_sq, id_emb.paths, id_emb.sequence_ids, id_outliers),
        ("target", target_dist, target_sq, target_emb.paths, target_emb.sequence_ids, target_outliers),
    ):
        for idx, path in enumerate(paths):
            rows.append(
                {
                    "filepath": path,
                    "dataset": label,
                    "distance": float(dist[idx]),
                    "distance_squared": float(dist_sq[idx]),
                    "is_outlier": bool(flags[idx]),
                    "sequence_id": seq[idx],
                }
            )
    write_distances_csv(run_dir / "distances.csv", rows)

    plots_dir = ensure_plot_dir(run_dir)
    plot_histogram(id_dist, target_dist, threshold, plots_dir / "distance_hist.png")
    plot_violin(id_dist, target_dist, plots_dir / "distance_violin.png")
    plot_distance_index(id_dist, target_dist, plots_dir / "distance_vs_index.png")
    plot_distance_vs_sequence(target_emb.sequence_ids, target_dist, plots_dir / "target_distance_vs_sequence.png")
    plot_pca_scatter(
        np.concatenate([id_emb.embeddings, target_emb.embeddings], axis=0),
        split_index=len(id_emb.embeddings),
        distances=combined_dist,
        outliers=combined_outliers,
        run_dir=run_dir,
        seed=seed,
        mean_vec=mean_vec,
    )

    k_grid = min(int(ood_cfg.get("outlier_grid_topk", 12)), len(target_emb.paths))
    top_target_paths = [target_emb.paths[i] for i in target_order[:k_grid]]
    top_target_distances = target_dist[target_order][:k_grid]
    plot_topk_bar(top_target_paths, top_target_distances, top_k, plots_dir / "top_target_outliers_bar.png")
    save_outlier_grid(
        top_target_paths,
        top_target_distances,
        plots_dir / "top_target_outliers_grid.png",
        cols=int(ood_cfg.get("outlier_grid_cols", 4)),
    )

    if run_dir is not None:
        update_metrics(
            run_dir,
            "ood_detection",
            {
                "threshold": threshold,
                "id_outlier_rate": summary["id_outlier_rate"],
                "target_outlier_rate": summary["target_outlier_rate"],
                "num_id": summary["num_id"],
                "num_target": summary["num_target"],
                "embedding_dim": summary["embedding_dim"],
            },
        )

    print(f"[ood] Threshold ({quantile:.2f} quantile): {threshold:.4f}")
    print(f"[ood] ID outlier rate: {summary['id_outlier_rate'] * 100:.2f}%")
    print(f"[ood] Target outlier rate: {summary['target_outlier_rate'] * 100:.2f}%")
    print(f"[ood] Results saved to: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[ood] ERROR: {exc}")
        raise
