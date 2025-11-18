#!/usr/bin/env python3
# scripts/run_pca.py

import sys
import yaml
import numpy as np
from pathlib import Path

from dino_peft.analysis.dimred import load_feature_npz, run_pca, run_umap
from dino_peft.utils.plots import scatter_2d

def load_cfg(path: Path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty config")
    return cfg

def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/em_pca_mac.yaml")
    cfg = load_cfg(cfg_path)

    data_cfg = cfg.get("data", {})
    pca_cfg = cfg.get("pca", {})

    input_path = Path(data_cfg["input_path"])
    output_dir = Path(data_cfg["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    pca_out = output_dir / f"{stem}_pca.png"
    umap_out = output_dir / f"{stem}_umap.png"

    # PCA config
    n_components = int(pca_cfg.get("n_components", 2))
    plot_dims = pca_cfg.get("plot_dims", [0, 1])
    whiten = bool(pca_cfg.get("whiten", False))
    l2norm = bool(pca_cfg.get("l2norm", False))
    seed = int(pca_cfg.get("seed", 0))

    # UMAP Config
    umap_cfg = cfg.get("umap", {})
    umap_enabled = bool(umap_cfg.get("enabled", False))
    umap_output = Path(umap_cfg.get("output_path", input_path.with_name(input_path.stem + "_umap.png")))
    pre_umap_dim = int(umap_cfg.get("pca_dim", 50))  # PCA dim before UMAP
    umap_params = {
        "n_neighbors": int(umap_cfg.get("n_neighbors", 15)),
        "min_dist": float(umap_cfg.get("min_dist", 0.1)),
        "n_components": int(umap_cfg.get("n_components", 2)),
        "metric": umap_cfg.get("metric", "euclidean"),
        "random_state": int(umap_cfg.get("seed", seed)),
    }

    # Load features/labels
    bundle = load_feature_npz(input_path)
    feats = np.asarray(bundle.features)


    # ======================== Run PCA ====================================
    pca, emb = run_pca(
        feats,
        n_components=n_components,
        whiten=whiten,
        random_state=seed,
        l2norm=l2norm,
    )

    # Select dimensions to plot
    i, j = plot_dims
    xy = emb[:, [i, j]]

    # Build label names from meta if available for title and legends
    label_names = None
    if bundle.meta and bundle.meta.get("dataset_name_to_id"):
        label_names = {v: k for k, v in bundle.meta["dataset_name_to_id"].items()}

    dino_size = None
    if bundle.meta:
        dino_size = bundle.meta.get("dino_size")
    if dino_size is None and hasattr(bundle, "dino_size"):
        dino_size = getattr(bundle, "dino_size")
    if isinstance(dino_size, (list, np.ndarray)):
        dino_size = dino_size[0]

    # Title with DINO size and explained variance of plotted PCs
    evr = pca.explained_variance_ratio_
    title = f"DINO {dino_size} PCA (PC1 {evr[i]:.1%}, PC2 {evr[j]:.1%})" if dino_size else f"DINO PCA (PC1 {evr[i]:.1%}, PC2 {evr[j]:.1%})"

    # Plot
    scatter_2d(
        xy=xy,
        labels=bundle.dataset_ids,
        label_names=label_names,
        out_path=pca_out,
        title=title,
    )

    print(f"Saved PCA scatter to {pca_out}")
    print(f"N={emb.shape[0]}, original_dim={feats.shape[1]}, pca_dim={n_components}")
    print(f"Explained variance (first components): {evr[: min(5, len(evr))]}")

    # ======================== Run UMAP ====================================
    if umap_enabled:
        pca_pre, feats_pre = run_pca(
            feats,
            n_components=pre_umap_dim,
            whiten=whiten,
            random_state=seed,
            l2norm=l2norm,
        )
        _, umap_emb = run_umap(feats_pre, **umap_params)
        title_umap = f"DINO {dino_size} PCA{pre_umap_dim}→UMAP (nn={umap_params['n_neighbors']}, md={umap_params['min_dist']})" if dino_size else f"PCA{pre_umap_dim}→UMAP (nn={...})"
        scatter_2d(
            xy=umap_emb[:, :2],
            labels=bundle.dataset_ids,
            label_names=label_names,
            out_path=umap_out,
            title=title_umap,
        )
        print(f"Saved UMAP scatter to {umap_out}")


if __name__ == "__main__":
    main()
