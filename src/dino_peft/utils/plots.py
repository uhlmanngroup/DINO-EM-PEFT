# utils/plots.py
# This file contains multiple plots from differen part of the pipeline

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Dict


try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]

def _to_numpy_img(x: ArrayLike) -> np.ndarray:
    """
    Accepts torch/numpy; shapes: HxW, CxHxW, HxWxC.
    Returns float np.ndarray in [0,1] with shape (H,W) or (H,W,3).
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    else:
        x = np.asarray(x)

    # move channels-last if needed
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] < x.shape[1]:
        x = np.transpose(x, (1, 2, 0))

    # grayscale -> (H,W)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[..., 0]

    # normalize to [0,1] safely
    x = x.astype(np.float32)
    if x.size > 0:
        vmin, vmax = np.nanmin(x), np.nanmax(x)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)

def save_triptych_grid(
    samples: Iterable[dict],
    out_path: str,
    *,
    title: Optional[str] = None,
    row_labels: Sequence[str] = ("Original", "Ground truth", "Prediction"),
    dpi: int = 180,
    border_lw: float = 1.25,
    figsize_per_col: float = 3.2,
    figsize_per_row: float = 3.2,
) -> str:
    """
    Save a 3-row (orig/gt/pred) × N-column grid of triptychs.
    Each item in `samples` must be a dict with:
        {
          "image": ArrayLike,         # original image
          "gt": ArrayLike,            # ground-truth mask/image
          "pred": ArrayLike,          # prediction mask/image
          "name": str                 # column title
        }

    Returns the output path.
    """
    samples = list(samples)
    if len(samples) == 0:
        raise ValueError("No samples provided to save_triptych_grid()")

    n_cols = len(samples)
    n_rows = 3

    fig_w = figsize_per_col * n_cols
    fig_h = figsize_per_row * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    # Ensure axes is 2D even if n_cols == 1
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    plt.subplots_adjust(
        left=0.12, right=0.98, top=0.88, bottom=0.08,
        wspace=0.25, hspace=0.22
    )

    for c, s in enumerate(samples):
        name = s.get("name", f"sample_{c}")
        # Column header
        axes[0, c].set_title(str(name), fontsize=12, pad=8)

        # Extract arrays
        img  = _to_numpy_img(s["image"])
        gt   = _to_numpy_img(s["gt"])
        pred = _to_numpy_img(s["pred"])

        # Draw three rows
        tiles = (img, gt, pred)
        for r in range(n_rows):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            arr = tiles[r]

            # Choose cmap automatically (RGB shown as-is; 2D uses gray)
            if arr.ndim == 2:
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr, vmin=0, vmax=1)

            # Thin border for clear separation
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_lw)

    # Y-axis labels on the left-most column
    for r, lab in enumerate(row_labels):
        axes[r, 0].set_ylabel(lab, fontsize=13, rotation=90, labelpad=18)

    if title:
        fig.suptitle(title, fontsize=14)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def scatter_2d(
    xy: np.ndarray,
    labels: Optional[np.ndarray],
    label_names: Optional[Dict[int, str]],
    out_path: Path | str,
    title: str = "",
    figsize=(6, 6),
    alpha=0.7,
    s=12,
):
    """
    xy: (N, 2) array of PCA coords.
    labels: (N,) ints or None. If None, single-color plot.
    label_names: mapping {id: name} for legend; optional.
    """
    xy = np.asarray(xy)
    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], c="C0", alpha=alpha, s=s, label="data")
    else:
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        # pick a palette big enough
        cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
        for i, lbl in enumerate(uniq):
            mask = labels == lbl
            name = label_names.get(int(lbl), str(lbl)) if label_names else str(lbl)
            ax.scatter(xy[mask, 0], xy[mask, 1], color=cmap(i % cmap.N), alpha=alpha, s=s, label=name)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if title:
        ax.set_title(title)
    if labels is not None:
        ax.legend(loc="best", fontsize=8, markerscale=1.2, frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path