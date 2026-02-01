#!/usr/bin/env python3
# scripts/run_pca.py

import argparse
import re
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from dino_peft.analysis.dimred import load_feature_npz, run_pca, run_umap
from dino_peft.utils.plots import scatter_2d
from dino_peft.utils.paths import setup_run_dir, update_metrics, write_run_info

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None

DISPLAY_NAME_MAP = {
    "droso": "VNC",
    "lucchi": "Lucchi++",
}

def _display_name(value):
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    return DISPLAY_NAME_MAP.get(key, value)


def _map_display_names(values):
    if values is None:
        return None
    if isinstance(values, dict):
        return {key: _display_name(val) for key, val in values.items()}
    if isinstance(values, (list, tuple, np.ndarray)):
        return np.array([_display_name(val) for val in values], dtype=object)
    return _display_name(values)

def load_cfg(path: Path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty config")
    return cfg


def _extract_sequence_and_z(path: str) -> tuple[int | None, int | None]:
    """Return (sequence_number, z_plane) inferred from a filename."""
    stem = Path(path).stem
    seq = None
    z_plane = None

    seq_match = re.search(r"(droso|lucchi)[^0-9]*([0-9]+)", stem, flags=re.IGNORECASE)
    if seq_match:
        seq = int(seq_match.group(2))
    else:
        digits = re.findall(r"(\d+)", stem)
        if digits:
            seq = int(digits[-1])

    z_match = re.search(r"(?:^|[_-])(?:z|plane)(\d+)", stem, flags=re.IGNORECASE)
    if z_match:
        z_plane = int(z_match.group(1))

    return seq, z_plane


def _collect_image_metadata(image_paths, total_items: int):
    seq_values = np.full(total_items, np.nan, dtype=float)
    z_planes = np.full(total_items, np.nan, dtype=float)
    names = ["" for _ in range(total_items)]
    if not image_paths:
        return seq_values, z_planes, names

    for idx in range(min(total_items, len(image_paths))):
        path = str(image_paths[idx])
        seq, z_plane = _extract_sequence_and_z(path)
        if seq is not None:
            seq_values[idx] = seq
        if z_plane is not None:
            z_planes[idx] = z_plane
        names[idx] = Path(path).name
    return seq_values, z_planes, names


def _plot_sequence_scatter(xy: np.ndarray, seq_values: np.ndarray, title: str, out_path: Path | str):
    seq = np.asarray(seq_values, dtype=float)
    valid = ~np.isnan(seq)
    if not valid.any():
        return None

    xy = np.asarray(xy)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(xy[valid, 0], xy[valid, 1], c=seq[valid], cmap="viridis", s=16, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sequence #", rotation=270, labelpad=12)

    missing = ~valid
    if missing.any():
        ax.scatter(xy[missing, 0], xy[missing, 1], c="lightgray", alpha=0.35, s=12, label="No sequence id")
        ax.legend(loc="best", fontsize=8, frameon=False)

    ax.set_xlabel("D1")
    ax.set_ylabel("D2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _build_hover_text(
    n_points: int,
    dataset_labels,
    dataset_ids,
    label_names,
    seq_values,
    z_planes,
    image_names,
):
    hover = []
    dataset_labels = np.asarray(dataset_labels) if dataset_labels is not None else None
    dataset_ids = np.asarray(dataset_ids) if dataset_ids is not None else None
    for idx in range(n_points):
        dataset = None
        if dataset_labels is not None and idx < len(dataset_labels):
            dataset = dataset_labels[idx]
        elif dataset_ids is not None and idx < len(dataset_ids):
            dataset = label_names.get(int(dataset_ids[idx]), str(dataset_ids[idx])) if label_names else str(dataset_ids[idx])

        seq_val = seq_values[idx]
        z_val = z_planes[idx]
        seq_txt = "n/a" if np.isnan(seq_val) else f"{int(seq_val)}"
        z_txt = "n/a" if np.isnan(z_val) else f"{int(z_val)}"
        img_txt = image_names[idx] if idx < len(image_names) and image_names[idx] else ""

        parts = [f"Dataset: {dataset or 'n/a'}", f"Seq #: {seq_txt}", f"Z plane: {z_txt}"]
        if img_txt:
            parts.append(f"Image: {img_txt}")
        hover.append("<br>".join(parts))
    return np.asarray(hover)


def _sorted_unique(values):
    cleaned = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        cleaned.append(v)
    if not cleaned:
        return []
    if all(isinstance(v, (int, float)) for v in cleaned):
        return sorted(set(cleaned))
    return sorted(set(cleaned), key=str)


def _save_plotly_scatter(xy, hover_text, groupings, title: str, out_path: Path | str):
    if go is None:
        print(f"[run_pca] Plotly not installed; skipping {out_path}")
        return None

    xy = np.asarray(xy)
    hover = np.asarray(hover_text)
    traces = []
    group_to_indices = []

    for group_label, values in groupings:
        if values is None:
            continue
        values = list(values)
        unique_vals = _sorted_unique(values)
        if not unique_vals:
            continue
        trace_ids = []
        for val in unique_vals:
            mask = np.array([v == val for v in values], dtype=bool)
            if not mask.any():
                continue
            trace = go.Scattergl(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                marker=dict(size=7),
                name=str(val),
                hovertext=hover[mask],
                hoverinfo="text",
                visible=False,
            )
            trace_ids.append(len(traces))
            traces.append(trace)
        if trace_ids:
            group_to_indices.append((group_label, trace_ids))

    if not traces:
        trace = go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(size=7),
            hovertext=hover,
            hoverinfo="text",
        )
        fig = go.Figure(data=[trace])
    else:
        # Enable first grouping by default
        if group_to_indices:
            for idx in group_to_indices[0][1]:
                traces[idx].visible = True
        fig = go.Figure(data=traces)
        if len(group_to_indices) > 1:
            buttons = []
            total = len(traces)
            for group_label, trace_ids in group_to_indices:
                visible = [False] * total
                for idx in trace_ids:
                    visible[idx] = True
                buttons.append(
                    dict(
                        label=group_label,
                        method="update",
                        args=[
                            {"visible": visible},
                            {"legend": {"title": {"text": group_label}}},
                        ],
                    )
                )
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="dropdown",
                        buttons=buttons,
                        x=1.02,
                        y=1.15,
                    )
                ]
            )
        if group_to_indices:
            fig.update_layout(legend=dict(title=group_to_indices[0][0]))

    fig.update_layout(
        title=title,
        xaxis_title="D1",
        yaxis_title="D2",
        template="plotly_white",
        dragmode="pan",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path

DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "mac" / "em_pca_mac.yaml"

def parse_args() -> Path:
    ap = argparse.ArgumentParser(description="Run PCA/UMAP on extracted DINO features.")
    ap.add_argument("cfg", nargs="?", default=None, help="Path to YAML config.")
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg_flag",
        default=None,
        help="Path to YAML config.",
    )
    args = ap.parse_args()
    cfg_path = args.cfg_flag or args.cfg or str(DEFAULT_CFG)
    return Path(cfg_path).expanduser()


def main():
    cfg_path = parse_args()
    cfg = load_cfg(cfg_path)

    data_cfg = cfg.get("data", {})
    pca_cfg = cfg.get("pca", {})

    input_path = Path(data_cfg["input_path"]).expanduser()
    run_dir = None
    task_type = cfg.get("task_type", "feats")
    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(cfg, task_type=task_type, subdirs=("plots",))
        plots_dir = run_dir / "plots"
        write_run_info(
            run_dir,
            {
                "task_type": task_type,
                "input_features": input_path,
                "n_components": int(pca_cfg.get("n_components", 2)),
            },
        )
    else:
        output_dir = Path(data_cfg.get("output_path", ".")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir

    stem = input_path.stem
    pca_out = plots_dir / f"{stem}_pca.png"
    umap_out = plots_dir / f"{stem}_umap.png"
    pca_seq_out = plots_dir / f"{stem}_pca_sequence.png"
    umap_seq_out = plots_dir / f"{stem}_umap_sequence.png"
    pca_html_out = plots_dir / f"{stem}_pca_interactive.html"
    umap_html_out = plots_dir / f"{stem}_umap_interactive.html"

    # PCA config
    n_components = int(pca_cfg.get("n_components", 2))
    plot_dims = pca_cfg.get("plot_dims", [0, 1])
    whiten = bool(pca_cfg.get("whiten", False))
    l2norm = bool(pca_cfg.get("l2norm", False))
    seed = int(pca_cfg.get("seed", 0))

    # UMAP Config
    umap_cfg = cfg.get("umap", {})
    umap_enabled = bool(umap_cfg.get("enabled", False))
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
    label_names = _map_display_names(label_names)

    dataset_label_per_sample = None
    if getattr(bundle, "dataset_names", None) is not None:
        dataset_label_per_sample = np.asarray(bundle.dataset_names)
    elif bundle.dataset_ids is not None and label_names:
        dataset_label_per_sample = np.array(
            [label_names.get(int(idx), str(idx)) for idx in bundle.dataset_ids],
            dtype=object,
        )
    dataset_label_per_sample = _map_display_names(dataset_label_per_sample)

    dino_size = None
    backbone_name = None
    backbone_variant = None
    if bundle.meta:
        dino_size = bundle.meta.get("dino_size")
        backbone_name = bundle.meta.get("backbone_name")
        backbone_variant = bundle.meta.get("backbone_variant")
    if dino_size is None and hasattr(bundle, "dino_size"):
        dino_size = getattr(bundle, "dino_size")
    if isinstance(dino_size, (list, np.ndarray)):
        dino_size = dino_size[0]
    if isinstance(backbone_name, (list, np.ndarray)):
        backbone_name = backbone_name[0]
    if isinstance(backbone_variant, (list, np.ndarray)):
        backbone_variant = backbone_variant[0]

    # Title with backbone label and explained variance of plotted PCs
    evr = pca.explained_variance_ratio_
    if backbone_name or backbone_variant:
        label = f"{backbone_name or 'backbone'} {backbone_variant or ''}".strip()
    elif dino_size:
        label = f"DINO {dino_size}"
    else:
        label = "DINO"
    title = f"{label} PCA (PC1 {evr[i]:.1%}, PC2 {evr[j]:.1%})"

    seq_values, z_planes, image_names = _collect_image_metadata(getattr(bundle, "image_paths", None), xy.shape[0])
    hover_text = _build_hover_text(
        xy.shape[0],
        dataset_label_per_sample,
        getattr(bundle, "dataset_ids", None),
        label_names,
        seq_values,
        z_planes,
        image_names,
    )

    groupings = []
    dataset_group_values = dataset_label_per_sample
    if dataset_group_values is None and getattr(bundle, "dataset_ids", None) is not None:
        dataset_group_values = np.asarray(bundle.dataset_ids)
    if dataset_group_values is not None:
        groupings.append(("Dataset", dataset_group_values))
    if np.any(~np.isnan(z_planes)):
        z_group = np.array(
            [None if np.isnan(v) else int(v) for v in z_planes],
            dtype=object,
        )
        groupings.append(("Z plane", z_group))

    # Plot
    scatter_2d(
        xy=xy,
        labels=bundle.dataset_ids,
        label_names=label_names,
        out_path=pca_out,
        title=title,
    )
    print(f"Saved PCA scatter to {pca_out}")
    seq_plot_path = _plot_sequence_scatter(
        xy,
        seq_values,
        f"{title} (sequence order)",
        pca_seq_out,
    )
    if seq_plot_path:
        print(f"Saved PCA sequence scatter to {seq_plot_path}")

    html_path = _save_plotly_scatter(
        xy,
        hover_text,
        groupings,
        f"{title} (interactive)",
        pca_html_out,
    )
    if html_path:
        print(f"Saved PCA interactive scatter to {html_path}")

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
        if backbone_name or backbone_variant:
            label_umap = f"{backbone_name or 'backbone'} {backbone_variant or ''}".strip()
        elif dino_size:
            label_umap = f"DINO {dino_size}"
        else:
            label_umap = "PCA"
        title_umap = (
            f"{label_umap} PCA{pre_umap_dim}→UMAP "
            f"(nn={umap_params['n_neighbors']}, md={umap_params['min_dist']})"
        )
        umap_xy = umap_emb[:, :2]
        scatter_2d(
            xy=umap_emb[:, :2],
            labels=bundle.dataset_ids,
            label_names=label_names,
            out_path=umap_out,
            title=title_umap,
        )
        print(f"Saved UMAP scatter to {umap_out}")
        seq_plot_path = _plot_sequence_scatter(
            umap_xy,
            seq_values,
            f"{title_umap} (sequence order)",
            umap_seq_out,
        )
        if seq_plot_path:
            print(f"Saved UMAP sequence scatter to {seq_plot_path}")
        html_path = _save_plotly_scatter(
            umap_xy,
            hover_text,
            groupings,
            f"{title_umap} (interactive)",
            umap_html_out,
        )
        if html_path:
            print(f"Saved UMAP interactive scatter to {html_path}")

    if run_dir is not None:
        update_metrics(
            run_dir,
            "pca",
            {
                "pca_components": int(n_components),
                "pca_plot_dims": list(plot_dims),
                "umap_enabled": bool(umap_enabled),
                "input_features": str(input_path),
            },
        )


if __name__ == "__main__":
    main()
