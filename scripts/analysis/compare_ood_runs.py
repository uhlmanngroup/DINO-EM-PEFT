#!/usr/bin/env python3
"""Compare multiple OOD-detection runs (fine-tuned vs frozen, etc.).

Example (local; absolute paths are user-specific):
    python scripts/analysis/compare_ood_runs.py \
      /Users/cfuste/Documents/Results/DINO-LoRA/ood-detection/2025-12-16_crop-common-square_droso-vs-testlucchi_dinov2-base_nolora_ood \
      /Users/cfuste/Documents/Results/DINO-LoRA/ood-detection/2025-12-16_crop-common-square_droso-vs-testlucchi_dinov2-base_lora_ood \
      --labels "Frozen DINO" "PEFT DINO"
"""


# from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

try:  # optional convenience dependency
    import pandas as pd
except Exception:  # pragma: no cover - fallback path
    pd = None


@dataclass
class RunInfo:
    label: str
    path: Path
    summary: Dict[str, object]
    threshold: float
    quantile: float
    id_outlier_rate: float
    target_outlier_rate: float
    target_ratio: np.ndarray
    id_ratio: np.ndarray
    target_scores: np.ndarray
    id_scores: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Render comparison plots for multiple ood-detection runs. "
            "Point it at the run directories produced by scripts/ood_detection.py."
        )
    )
    ap.add_argument(
        "runs",
        nargs="+",
        help="One or more ood-detection run directories.",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        help="Optional human-readable labels (same order as runs).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="Destination directory for comparison artifacts (default: sibling of the first run).",
    )
    return ap.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_distances(run_dir: Path):
    csv_path = run_dir / "distances.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"distances.csv not found in {run_dir}")
    if pd is not None:
        return pd.read_csv(csv_path)
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _parse_column(row, key, default=None):
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _collect_arrays(data, dataset: str, threshold: float):
    rows: Iterable
    if pd is not None and isinstance(data, pd.DataFrame):
        subset = data[data["dataset"].astype(str).str.lower() == dataset]
        if subset.empty:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        ratio_col = (
            subset["distance_over_threshold"]
            if "distance_over_threshold" in subset.columns
            else subset["distance"] / max(threshold, 1e-12)
        )
        score_col = (
            subset["ood_score"]
            if "ood_score" in subset.columns
            else -np.log10(
                np.maximum(
                    1e-12,
                    1.0
                    - subset["distance"].rank(pct=True, method="max").to_numpy(dtype=float),
                )
            )
        )
        return ratio_col.to_numpy(dtype=np.float64), score_col.to_numpy(dtype=np.float64)

    rows = [
        row
        for row in data
        if str(row.get("dataset", "")).strip().lower() == dataset
    ]
    ratios = []
    scores = []
    safe_thresh = max(threshold, 1e-12)
    for row in rows:
        dist = _parse_column(row, "distance", 0.0)
        if dist is None:
            continue
        ratio = _parse_column(row, "distance_over_threshold")
        if ratio is None:
            ratio = dist / safe_thresh
        score = _parse_column(row, "ood_score")
        if score is None:
            # fallback: approximate from tail prob if present
            tail = _parse_column(row, "tail_prob")
            if tail is None:
                tail = max(1e-12, 1.0 - (dist / (safe_thresh + 1e-12)))
            score = -math.log10(max(tail, 1e-12))
        ratios.append(ratio)
        scores.append(score)
    return _to_numpy(ratios), _to_numpy(scores)


def load_run(run_dir: Path, label: str) -> RunInfo:
    summary = load_summary(run_dir)
    threshold_meta = summary.get("threshold") or {}
    threshold = float(summary.get("threshold_value") or threshold_meta.get("value") or 0.0)
    quantile = float(summary.get("threshold_quantile") or threshold_meta.get("quantile") or 0.95)
    id_rate = float(summary.get("id_outlier_rate", 0.0))
    target_rate = float(summary.get("target_outlier_rate", 0.0))

    data = _load_distances(run_dir)
    ratio_id, score_id = _collect_arrays(data, "id", threshold)
    ratio_target, score_target = _collect_arrays(data, "target", threshold)

    return RunInfo(
        label=label,
        path=run_dir,
        summary=summary,
        threshold=threshold,
        quantile=quantile,
        id_outlier_rate=id_rate,
        target_outlier_rate=target_rate,
        target_ratio=ratio_target,
        id_ratio=ratio_id,
        target_scores=score_target,
        id_scores=score_id,
    )


def plot_outlier_rates(runs: Sequence[RunInfo], out_path: Path) -> None:
    labels = [r.label for r in runs]
    id_vals = [r.id_outlier_rate * 100 for r in runs]
    tgt_vals = [r.target_outlier_rate * 100 for r in runs]
    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, id_vals, width, label="ID")
    ax.bar(x + width / 2, tgt_vals, width, label="Target")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Outlier rate (%)")
    ax.set_title("Outlier rates per run")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_ratio_violin(runs: Sequence[RunInfo], out_path: Path) -> None:
    data = [r.target_ratio for r in runs if r.target_ratio.size > 0]
    labels = [r.label for r in runs if r.target_ratio.size > 0]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    palette = plt.get_cmap("tab10")
    for idx, body in enumerate(parts.get("bodies", [])):
        body.set_facecolor(palette(idx % palette.N))
        body.set_alpha(0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="distance = threshold")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("Target distance / threshold")
    ax.set_title("Target normalized distance distribution")
    if len(labels) > 1:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_ratio_density(runs: Sequence[RunInfo], out_path: Path) -> None:
    max_ratio = max(
        (float(np.quantile(r.target_ratio, 0.995)) for r in runs if r.target_ratio.size),
        default=0.0,
    )
    if max_ratio == 0.0:
        return
    bins = np.linspace(0, max(max_ratio, 1.5), 80)
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        if run.target_ratio.size == 0:
            continue
        hist, edges = np.histogram(run.target_ratio, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, label=run.label, color=palette(idx % palette.N))
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Target distance / threshold")
    ax.set_ylabel("Density")
    ax.set_title("Target normalized distance density")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_ood_score_hist(runs: Sequence[RunInfo], out_path: Path) -> None:
    max_score = max(
        (float(np.quantile(r.target_scores, 0.995)) for r in runs if r.target_scores.size),
        default=0.0,
    )
    if max_score == 0.0:
        return
    bins = np.linspace(0, max(max_score, 1.0), 70)
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        if run.target_scores.size == 0:
            continue
        ax.hist(
            run.target_scores,
            bins=bins,
            alpha=0.5,
            label=run.label,
            color=palette(idx % palette.N),
        )
    ax.set_xlabel("-log10(tail probability)")
    ax.set_ylabel("Count")
    ax.set_title("Target OOD score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def plot_ratio_box(run_infos: Sequence[RunInfo], out_path: Path) -> None:
    data = [r.summary.get("target_median_over_threshold") for r in run_infos]
    if not any(isinstance(val, (int, float)) and math.isfinite(val) for val in data):
        return
    labels = [r.label for r in run_infos]
    values = [float(val) if isinstance(val, (int, float)) else float("nan") for val in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#4c72b0")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Target median (distance / threshold)")
    ax.set_title("Target median normalized distance per run")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def write_summary(runs: Sequence[RunInfo], out_path: Path) -> None:
    payload = {
        "runs": [
            {
                "label": run.label,
                "path": str(run.path),
                "threshold_value": run.threshold,
                "threshold_quantile": run.quantile,
                "id_outlier_rate": run.id_outlier_rate,
                "target_outlier_rate": run.target_outlier_rate,
                "target_median_over_threshold": run.summary.get("target_median_over_threshold"),
                "target_min_over_threshold": run.summary.get("target_min_over_threshold"),
                "target_max_over_threshold": run.summary.get("target_max_over_threshold"),
            }
            for run in runs
        ]
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if len(args.runs) < 2:
        raise SystemExit("Provide at least two run directories to compare.")

    run_paths = [Path(p).expanduser().resolve() for p in args.runs]
    labels = args.labels or []
    if labels and len(labels) != len(run_paths):
        raise SystemExit("Number of labels must match number of runs.")
    if not labels:
        labels = [path.name for path in run_paths]

    if args.output:
        out_dir = ensure_dir(args.output.expanduser())
    else:
        parent = run_paths[0].parent
        slug = "vs".join(p.name for p in run_paths[:2])
        out_dir = ensure_dir(parent / f"comparison_{slug}")

    runs = [load_run(path, label) for path, label in zip(run_paths, labels)]
    write_summary(runs, out_dir / "comparison_summary.json")

    plot_outlier_rates(runs, out_dir / "outlier_rates.png")
    plot_ratio_violin(runs, out_dir / "target_ratio_violin.png")
    plot_ratio_density(runs, out_dir / "target_ratio_density.png")
    plot_ratio_box(runs, out_dir / "target_ratio_medians.png")
    plot_ood_score_hist(runs, out_dir / "target_ood_scores.png")

    print(f"[compare_ood] Saved comparison plots to {out_dir}")


if __name__ == "__main__":
    main()
