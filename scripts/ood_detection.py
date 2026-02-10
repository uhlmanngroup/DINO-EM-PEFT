#!/usr/bin/env python3
"""Mahalanobis OOD detection on DINO/PEFT embeddings.

Example (local):
    python scripts/ood_detection.py --cfg configs/mac/ood_detection_mac.yaml
"""

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
from dino_peft.backbones import (
    build_backbone,
    build_preprocess_transform,
    resolve_backbone_cfg,
    resolve_preprocess_cfg,
)
from dino_peft.models.lora import apply_peft
from dino_peft.utils.image_size import DEFAULT_IMG_SIZE_CFG
from dino_peft.utils.paths import setup_run_dir, update_metrics, write_run_info

ImageFile.LOAD_TRUNCATED_IMAGES = True  # load incomplete TIFFs instead of crashing

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CACHE_VERSION = 3


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
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
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
    """Expand globs/file lists into absolute image paths + record filters."""
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


# Thin Dataset wrapper that loads one image per path (applies spatial preprocess + transforms).
# Thin Dataset wrapper that loads one image per path (applies spatial preprocess + transforms).
class ImagePathDataset(Dataset):
    """Load PIL images, run spatial preprocessing, then feed repo transforms."""

    def __init__(self, paths: Sequence[str], transform, *, spatial_preprocessor=None, dataset_label: str = "id"):
        self.paths = list(paths)
        self.transform = transform
        self.spatial_preprocessor = spatial_preprocessor  # hook for center-crops/tiles
        self.dataset_label = dataset_label

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
        # Run the geometry harmonization (native / crop / etc.) before transforms.
        if self.spatial_preprocessor is not None:
            img = self.spatial_preprocessor.apply(img, self.dataset_label)
        if self.transform:
            img = self.transform(img)
        return img, path


# Helper to inspect dataset geometry without preloading everything.
def _probe_first_image_size(paths: Sequence[str]) -> Tuple[int, int]:
    """Inspect the first loadable image to determine dataset-native geometry."""
    for path in paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w > 0 and h > 0:
                    return int(w), int(h)
        except Exception:
            continue
    raise RuntimeError("Unable to determine image size for dataset (no readable images).")


def _center_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Simple center crop that preserves aspect ratio (no stretching)."""
    target_w, target_h = target_size
    w, h = image.size
    crop_w = min(target_w, w)
    crop_h = min(target_h, h)
    if crop_w == w and crop_h == h:
        return image
    left = max(0, int(round((w - crop_w) / 2)))
    top = max(0, int(round((h - crop_h) / 2)))
    right = left + crop_w
    bottom = top + crop_h
    return image.crop((left, top, right, bottom))


# SpatialPreprocessor: central switchboard for native/crop/tile logic.
class SpatialPreprocessor:
    """Centralizes how we crop/keep images so model + plots share identical views."""
    """Centralizes how we crop/keep images so model + plots share identical views."""

    def __init__(
        self,
        mode: str,
        multiple_of: Optional[int],
        id_paths: Sequence[str],
        target_paths: Sequence[str],
    ) -> None:
        self.mode = mode
        self.multiple_of = multiple_of if multiple_of and multiple_of > 0 else None
        self.id_probe_size = _probe_first_image_size(id_paths)
        self.target_probe_size = _probe_first_image_size(target_paths)
        self.crop_map: Dict[str, Optional[Tuple[int, int]]] = {"id": None, "target": None}
        self._warned_small: set[Tuple[str, Tuple[int, int]]] = set()
        self._resolve()

    def _snap(self, value: int) -> int:
        if not self.multiple_of:
            return value
        snapped = (value // self.multiple_of) * self.multiple_of
        return snapped if snapped > 0 else value

    def _snap_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        w, h = size
        return self._snap(w), self._snap(h)

    def _resolve(self) -> None:
        mode = self.mode.lower()
        if mode == "native":
            # No-op: keep original geometry as provided on disk
            return
        if mode == "crop_match_id":
            ref_w, ref_h = self._snap_size(self.id_probe_size)
            if ref_w <= 0 or ref_h <= 0:
                ref_w, ref_h = self.id_probe_size
            self.crop_map["target"] = (ref_w, ref_h)
        elif mode == "crop_common_square":
            id_min = min(self.id_probe_size)
            target_min = min(self.target_probe_size)
            side = min(id_min, target_min)
            side = self._snap(side)
            if side <= 0:
                side = min(id_min, target_min)
            self.crop_map["id"] = (side, side)
            self.crop_map["target"] = (side, side)
        else:
            raise ValueError(f"Unknown spatial preprocessing mode '{self.mode}'")

    def apply(self, image: Image.Image, dataset_label: str) -> Image.Image:
        """Apply resolved crop for the given dataset (if any)."""
        label = dataset_label.lower()
        crop_size = self.crop_map.get(label)
        if not crop_size:
            return image
        w, h = image.size
        target_w, target_h = crop_size
        adj_w = min(target_w, w)
        adj_h = min(target_h, h)
        adjusted = (adj_w, adj_h)
        if adjusted != crop_size:
            key = (label, adjusted)
            if key not in self._warned_small:
                print(
                    f"[ood][warn] Image smaller than desired crop for '{label}'. "
                    f"Requested {crop_size}, available {(w, h)}; using {adjusted}."
                )
                self._warned_small.add(key)
        return _center_crop(image, adjusted)

    def describe(self) -> str:
        parts = [self.mode]
        if self.multiple_of:
            parts.append(f"multiple_of={self.multiple_of}")
        if self.crop_map["id"]:
            parts.append(f"id_crop={self.crop_map['id']}")
        if self.crop_map["target"]:
            parts.append(f"target_crop={self.crop_map['target']}")
        return ", ".join(parts)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "multiple_of": self.multiple_of,
            "id_probe_size": list(self.id_probe_size),
            "target_probe_size": list(self.target_probe_size),
            "id_crop_size": list(self.crop_map["id"]) if self.crop_map["id"] else None,
            "target_crop_size": list(self.crop_map["target"]) if self.crop_map["target"] else None,
        }

    def fingerprint_for(self, label: str) -> Dict[str, Any]:
        info = self.summary_dict()
        info["dataset_label"] = label
        return info


def create_spatial_preprocessor(
    id_spec: DatasetSpec,
    target_spec: DatasetSpec,
    runtime_cfg: Dict[str, Any],
) -> SpatialPreprocessor:
    # Accept either runtime.spatial_preprocess or runtime.spatial (legacy) knobs.
    # We resolve cropping once so dataloaders, caching, and plots stay consistent.
    spatial_cfg = (
        runtime_cfg.get("spatial_preprocess")
        or runtime_cfg.get("spatial")
        or {}
    )
    if not isinstance(spatial_cfg, dict):
        raise ValueError("runtime.spatial_preprocess must be a mapping if provided.")
    mode = str(spatial_cfg.get("mode", "native")).lower()
    multiple = spatial_cfg.get("multiple_of")
    try:
        multiple_int = int(multiple)
    except (TypeError, ValueError):
        multiple_int = None
    return SpatialPreprocessor(mode, multiple_int, id_spec.paths, target_spec.paths)


# Pad batch to max height/width (preserves original aspect).
def pad_collate(batch):
    images, paths = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # Dynamically pad to the largest tensor in the batch; avoids distortions.
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
def load_backbone(model_cfg: Dict[str, Any], device: torch.device):
    """Instantiate backbone adapter and optionally restore LoRA adapters."""
    backbone_cfg = resolve_backbone_cfg(model_cfg)
    checkpoint = model_cfg.get("checkpoint")
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
    if checkpoint_path and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    backbone = build_backbone(backbone_cfg, device=device)
    backbone.eval()

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_cfg = ckpt.get("cfg", {}) or {}
        ckpt_backbone = resolve_backbone_cfg(ckpt_cfg)
        audit = apply_peft(
            backbone.model,
            ckpt_cfg,
            backbone_info=ckpt_backbone,
            write_report=False,
        )
        if audit is not None:
            lora_state = ckpt.get("backbone_lora") or {}
            if not lora_state:
                raise RuntimeError("Checkpoint missing backbone_lora weights.")
            state = backbone.model.state_dict()
            for key, tensor in lora_state.items():
                if key in state:
                    state[key] = tensor
            backbone.model.load_state_dict(state, strict=False)
        else:
            print("[ood] Checkpoint has no LoRA weights; using base backbone.")
    return backbone


def _compute_paths_digest(paths: Sequence[str]) -> str:
    """Fingerprint path ordering so caches invalidate when files change."""
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
    spatial_fingerprint: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    checkpoint = model_cfg.get("checkpoint")
    pooling = model_cfg.get("pooling", "mean")
    backbone_cfg = resolve_backbone_cfg(model_cfg)
    # Encode everything that affects the embedding values so cache hits are safe.
    meta = {
        "cache_version": CACHE_VERSION,
        "dataset_label": spec.label,
        "dataset_root": str(spec.root),
        "backbone_name": backbone_cfg.get("name"),
        "backbone_variant": backbone_cfg.get("variant"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "img_size": _metadata_repr(spec.img_size),
        "pooling": pooling,
        "l2_normalize": bool(model_cfg.get("l2_normalize", False)),
        "paths_digest": _compute_paths_digest(paths),
        "filters": _metadata_repr(spec.filters),
        "transform": "em_dino_unsup_transforms",
        "preprocess_preset": (backbone_cfg.get("preprocess") or {}).get("preset", "em"),
        "device": runtime_cfg.get("device", "auto"),
        "runtime_cfg": _metadata_repr(runtime_cfg),
        "spatial_preprocess": _metadata_repr(spatial_fingerprint or {}),
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
    model,
    device: torch.device,
    model_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
    run_dir: Path,
    cache_enabled: bool,
    spatial_preprocessor: Optional[SpatialPreprocessor],
) -> DatasetEmbeddings:
    cache_dir = _cache_paths(run_dir)
    metadata = build_cache_metadata(
        spec,
        model_cfg,
        runtime_cfg,
        spec.paths,
        spatial_preprocessor.fingerprint_for(spec.label) if spatial_preprocessor else None,
    )
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

    backbone_cfg = resolve_backbone_cfg(model_cfg)
    preprocess_cfg = resolve_preprocess_cfg(model_cfg, default_img_size=spec.img_size)
    transform = build_preprocess_transform(
        preprocess_cfg["preset"],
        preprocess_cfg["img_size"],
        backbone_cfg=backbone_cfg,
    )
    # Wrap dataset so spatial preprocessor is applied right after PIL load.
    dataset = ImagePathDataset(
        spec.paths,
        transform,
        spatial_preprocessor=spatial_preprocessor,
        dataset_label=spec.label,
    )
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
    # Single pass over dataset—no gradients—just feature extraction.
    for imgs, paths in tqdm(loader, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        output = model(imgs)
        if pooling == "mean":
            pooled = output.global_embedding
        elif pooling == "cls":
            if output.cls_token is None:
                raise ValueError("CLS pooling requested but backbone did not return cls_token.")
            pooled = output.cls_token.squeeze(1)
        else:
            raise ValueError(f"Unsupported pooling '{pooling}'. Use 'mean' or 'cls'.")
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


def log10_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "min", "max")}
    log_vals = np.log10(np.maximum(arr, 1e-12))
    return {
        "mean": float(np.mean(log_vals)),
        "std": float(np.std(log_vals)),
        "median": float(np.median(log_vals)),
        "min": float(np.min(log_vals)),
        "max": float(np.max(log_vals)),
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
    header = [
        "rank",
        "filepath",
        "dataset",
        "distance",
        "distance_squared",
        "threshold",
        "distance_over_threshold",
        "distance_minus_threshold",
        "id_percentile",
        "tail_prob",
        "ood_score",
        "is_outlier",
        "sequence_id",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            seq_val = row.get("sequence_id", np.nan)
            writer.writerow(
                {
                    "rank": rank,
                    "filepath": row["filepath"],
                    "dataset": row["dataset"],
                    "distance": f"{row['distance']:.8f}",
                    "distance_squared": f"{row['distance_squared']:.8f}",
                    "threshold": f"{row['threshold']:.8f}",
                    "distance_over_threshold": f"{row['distance_over_threshold']:.8f}",
                    "distance_minus_threshold": f"{row['distance_minus_threshold']:.8f}",
                    "id_percentile": f"{row['id_percentile']:.8f}",
                    "tail_prob": f"{row['tail_prob']:.8e}",
                    "ood_score": f"{row['ood_score']:.8f}",
                    "is_outlier": int(row["is_outlier"]),
                    "sequence_id": "" if (isinstance(seq_val, float) and np.isnan(seq_val)) else seq_val,
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
    # Reuse the same distance arrays across multiple plot types.
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


def plot_histogram_log(id_dist: np.ndarray, target_dist: np.ndarray, threshold: float, out_path: Path) -> None:
    id_pos = id_dist[id_dist > 0]
    tgt_pos = target_dist[target_dist > 0]
    combined = np.concatenate([id_pos, tgt_pos])
    if combined.size == 0:
        return
    low = combined.min()
    high = combined.max()
    if low <= 0 or not np.isfinite(low) or not np.isfinite(high):
        return
    bins = np.logspace(np.log10(low), np.log10(high), num=max(20, min(80, int(math.sqrt(combined.size)))))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_pos, bins=bins, alpha=0.6, label="ID", color="tab:blue")
    ax.hist(tgt_pos, bins=bins, alpha=0.6, label="Target", color="tab:orange")
    if threshold > 0:
        ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Mahalanobis distance (log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Distance histogram (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_histogram_id_zoom(id_dist: np.ndarray, threshold: float, out_path: Path) -> None:
    if id_dist.size == 0:
        return
    lower = float(np.min(id_dist))
    upper = float(np.quantile(id_dist, 0.99) * 1.1)
    if upper <= lower:
        upper = lower * 1.05 + 1e-6
    bins = np.linspace(lower, upper, num=50)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_dist, bins=bins, alpha=0.7, color="tab:blue")
    ax.axvline(threshold, color="red", linestyle="--", label="threshold")
    ax.set_xlim(lower, upper)
    ax.set_xlabel("Mahalanobis distance (ID zoom)")
    ax.set_ylabel("Count")
    ax.set_title("Distance histogram — ID zoom")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_normalized_hist(
    id_ratio: np.ndarray,
    target_ratio: np.ndarray,
    out_path: Path,
) -> None:
    combined = np.concatenate([id_ratio, target_ratio])
    if combined.size == 0:
        return
    upper = float(np.quantile(combined, 0.995))
    upper = max(upper, 1.5)
    bins = np.linspace(0, upper, num=60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_ratio, bins=bins, alpha=0.6, label="ID", color="tab:blue")
    ax.hist(target_ratio, bins=bins, alpha=0.6, label="Target", color="tab:orange")
    ax.axvline(1.0, color="red", linestyle="--", label="distance = threshold")
    ax.set_xlabel("Distance / threshold")
    ax.set_ylabel("Count")
    ax.set_title("Distance / threshold histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_normalized_violin(
    id_ratio: np.ndarray,
    target_ratio: np.ndarray,
    out_path: Path,
) -> None:
    if id_ratio.size == 0 and target_ratio.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [id_ratio, target_ratio]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc, color in zip(parts["bodies"], ("tab:blue", "tab:orange")):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xticks([1, 2], ["ID", "Target"])
    ax.set_ylabel("Distance / threshold")
    ax.set_title("Normalized distance distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_ood_score_hist(
    id_scores: np.ndarray,
    target_scores: np.ndarray,
    out_path: Path,
) -> None:
    combined = np.concatenate([id_scores, target_scores])
    if combined.size == 0:
        return
    upper = float(np.quantile(combined, 0.995))
    upper = max(upper, 1.0)
    bins = np.linspace(0, upper, num=60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_scores, bins=bins, alpha=0.6, label="ID", color="tab:blue")
    ax.hist(target_scores, bins=bins, alpha=0.6, label="Target", color="tab:orange")
    ax.set_xlabel("-log10(tail probability)")
    ax.set_ylabel("Count")
    ax.set_title("OOD score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def _load_example_image(path: str, spatial_preprocessor: Optional[SpatialPreprocessor], label: str) -> Optional[Tuple[Image.Image, Path]]:
    try:
        img = Image.open(path)
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if spatial_preprocessor is not None:
            img = spatial_preprocessor.apply(img, label)
        return img, Path(path)
    except Exception:
        return None


def plot_spatial_examples(
    id_spec: DatasetSpec,
    target_spec: DatasetSpec,
    spatial_preprocessor: Optional[SpatialPreprocessor],
    out_path: Path,
) -> None:
    """Visual sanity check showing how crops affect ID vs target geometry."""
    if not id_spec.paths or not target_spec.paths:
        return
    examples = [
        ("ID", id_spec.paths[0], "id"),
        ("Target", target_spec.paths[0], "target"),
    ]
    loaded = []
    for title, path, label in examples:
        sample = _load_example_image(path, spatial_preprocessor, label)
        if sample is None:
            continue
        loaded.append((title, *sample))
    if len(loaded) < 2:
        return
    fig, axes = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 6))
    axes = np.atleast_1d(axes)
    for ax, (title, img, path) in zip(axes, loaded):
        width, height = img.size
        ax.imshow(img)
        ax.set_title(f"{title}: {path.name}\n{width}×{height}px", fontsize=11)
        ax.set_xticks([0, width // 2, width])
        ax.set_yticks([0, height // 2, height])
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
    fig.suptitle("Spatial preprocessing preview", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# Save an inspection grid of top-K outlier images.
def save_outlier_grid(
    paths: Sequence[str],
    distances: np.ndarray,
    out_path: Path,
    cols: int = 4,
    spatial_preprocessor: Optional[SpatialPreprocessor] = None,
    dataset_label: Optional[str] = None,
) -> None:
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
            if spatial_preprocessor is not None and dataset_label is not None:
                img = spatial_preprocessor.apply(img, dataset_label)
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

    # Each dataset has independent filters/globs, so keep both blocks explicit.
    id_entry = data_cfg.get("id")
    target_entry = data_cfg.get("target")
    if id_entry is None or target_entry is None:
        raise ValueError("Config must define data.id and data.target sections.")

    # Resolve dataset configs and keep run metadata consistent.
    # Normalize dataset entries into reusable specs (paths, transforms, regex, etc.).
    id_spec = build_dataset_spec("id", id_entry, img_size_cfg, batch_size, num_workers)
    target_spec = build_dataset_spec("target", target_entry, img_size_cfg, batch_size, num_workers)

    # Determine how to harmonize spatial geometry (native vs crop modes).
    spatial_preprocessor = create_spatial_preprocessor(id_spec, target_spec, runtime_cfg)

    task_type = cfg.get("task_type", "ood-detection")
    run_dir = setup_run_dir(cfg, task_type=task_type, subdirs=("plots", "embeddings"))
    plots_dir = ensure_plot_dir(run_dir)
    write_run_info(
        run_dir,
        {
            "task_type": task_type,
            "device": str(device),
            "seed": seed,
            "id_root": str(id_spec.root),
            "target_root": str(target_spec.root),
            "checkpoint": model_cfg.get("checkpoint"),
            "backbone_name": resolve_backbone_cfg(model_cfg).get("name"),
            "backbone_variant": resolve_backbone_cfg(model_cfg).get("variant"),
        },
    )

    # Quick sanity log so users know geometry + hardware before extraction begins.
    print(f"[ood] Run directory: {run_dir}")
    print(f"[ood] Device: {device}")
    print(f"[ood] ID samples: {len(id_spec.paths)} | Target samples: {len(target_spec.paths)}")
    print(f"[ood] Spatial preprocessing: {spatial_preprocessor.describe()}")

    plot_spatial_examples(
        id_spec,
        target_spec,
        spatial_preprocessor,
        plots_dir / "spatial_preprocessing_preview.png",
    )

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
        spatial_preprocessor=spatial_preprocessor,
    )
    target_emb = extract_embeddings_for_dataset(
        target_spec,
        backbone,
        device,
        model_cfg,
        runtime_cfg,
        run_dir,
        cache_enabled=cache_enabled,
        spatial_preprocessor=spatial_preprocessor,
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
    safe_threshold = threshold if threshold > 1e-12 else 1e-12

    sorted_id = np.sort(id_dist)
    denom = max(len(sorted_id), 1)

    def _percentiles(vals: np.ndarray) -> np.ndarray:
        ranks = np.searchsorted(sorted_id, vals, side="right")
        return ranks / denom

    id_percentiles = _percentiles(id_dist)
    target_percentiles = _percentiles(target_dist)
    id_tail = np.clip(1.0 - id_percentiles, 0.0, 1.0)
    target_tail = np.clip(1.0 - target_percentiles, 0.0, 1.0)
    id_ood_scores = -np.log10(np.maximum(id_tail, 1e-12))
    target_ood_scores = -np.log10(np.maximum(target_tail, 1e-12))
    id_ratio = id_dist / safe_threshold
    target_ratio = target_dist / safe_threshold

    # Collect run-level diagnostics for JSON + CLI printouts.
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
        "threshold_quantile": quantile,
        "threshold_value": threshold,
        "spatial_preprocess": spatial_preprocessor.summary_dict(),
    }
    def _ratio_stat(arr: np.ndarray, fn) -> float:
        return float(fn(arr)) if arr.size > 0 else float("nan")

    summary["id_min_over_threshold"] = _ratio_stat(id_ratio, np.min)
    summary["id_median_over_threshold"] = _ratio_stat(id_ratio, np.median)
    summary["id_max_over_threshold"] = _ratio_stat(id_ratio, np.max)
    summary["target_min_over_threshold"] = _ratio_stat(target_ratio, np.min)
    summary["target_median_over_threshold"] = _ratio_stat(target_ratio, np.median)
    summary["target_max_over_threshold"] = _ratio_stat(target_ratio, np.max)
    summary["id_dist_log10_stats"] = log10_stats(id_dist)
    summary["target_dist_log10_stats"] = log10_stats(target_dist)
    # Persist spatial preprocessing metadata for downstream comparisons.
    summary["spatial_preprocess"] = spatial_preprocessor.summary_dict()

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

    # Flatten per-image info (paths, distances, normalized scores) for CSV export.
    dataset_rows: List[Dict[str, Any]] = []
    dataset_info = {
        "id": {
            "dist": id_dist,
            "dist_sq": id_sq,
            "paths": id_emb.paths,
            "seq": id_emb.sequence_ids,
            "outliers": id_outliers,
            "ratio": id_ratio,
            "delta": id_dist - threshold,
            "percentile": id_percentiles,
            "tail": id_tail,
            "score": id_ood_scores,
        },
        "target": {
            "dist": target_dist,
            "dist_sq": target_sq,
            "paths": target_emb.paths,
            "seq": target_emb.sequence_ids,
            "outliers": target_outliers,
            "ratio": target_ratio,
            "delta": target_dist - threshold,
            "percentile": target_percentiles,
            "tail": target_tail,
            "score": target_ood_scores,
        },
    }
    for label, info in dataset_info.items():
        dist = info["dist"]
        for idx, path in enumerate(info["paths"]):
            dataset_rows.append(
                {
                    "filepath": path,
                    "dataset": label,
                    "distance": float(dist[idx]),
                    "distance_squared": float(info["dist_sq"][idx]),
                    "threshold": float(threshold),
                    "distance_over_threshold": float(info["ratio"][idx]),
                    "distance_minus_threshold": float(info["delta"][idx]),
                    "id_percentile": float(info["percentile"][idx]),
                    "tail_prob": float(info["tail"][idx]),
                    "ood_score": float(info["score"][idx]),
                    "is_outlier": bool(info["outliers"][idx]),
                    "sequence_id": info["seq"][idx],
                }
            )
    csv_path = run_dir / "distances.csv"
    write_distances_csv(csv_path, dataset_rows)

    plot_histogram(id_dist, target_dist, threshold, plots_dir / "distance_hist.png")
    plot_histogram_log(id_dist, target_dist, threshold, plots_dir / "distance_hist_log.png")
    plot_histogram_id_zoom(id_dist, threshold, plots_dir / "distance_hist_id_zoom.png")
    plot_violin(id_dist, target_dist, plots_dir / "distance_violin.png")
    plot_normalized_hist(id_ratio, target_ratio, plots_dir / "distance_over_threshold_hist.png")
    plot_normalized_violin(id_ratio, target_ratio, plots_dir / "distance_over_threshold_violin.png")
    plot_ood_score_hist(id_ood_scores, target_ood_scores, plots_dir / "ood_score_hist.png")
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
        spatial_preprocessor=spatial_preprocessor,
        dataset_label="target",
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

    # Final console recap mirrors summary.json so quick runs don't need file inspection.
    target_median_ratio = summary["target_median_over_threshold"]
    target_ratio_txt = (
        f"{target_median_ratio:.3f}"
        if target_median_ratio is not None and math.isfinite(target_median_ratio)
        else "n/a"
    )
    print(
        "[ood] Threshold "
        f"(quantile={quantile:.2f}): value={threshold:.4f}"
    )
    print(
        "[ood] Outlier rates → "
        f"ID={summary['id_outlier_rate'] * 100:.2f}% "
        f"(expected ~{(1.0 - quantile) * 100:.2f}%), "
        f"Target={summary['target_outlier_rate'] * 100:.2f}%"
    )
    print(f"[ood] Target median distance/threshold: {target_ratio_txt}")
    print(f"[ood] Artifacts: CSV={csv_path}, summary={summary_path}, plots={plots_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[ood] ERROR: {exc}")
        raise
