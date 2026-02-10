#!/usr/bin/env python3
"""Create publication-ready plots from segmentation summary exports.

Examples:

    # Uses ~/Downloads/summary by default and writes into ./plots
    python scripts/analysis/plot_seg_summary.py

    # Custom locations
    python scripts/analysis/plot_seg_summary.py \\
        --summary-dir /path/to/summary \\
        --output-dir reports/plots \\
        --dpi 250
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "matplotlib is required. Install it via `pip install matplotlib`."
    ) from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit("pandas is required. Install it via `pip install pandas`.") from exc

try:
    import seaborn as sns
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit("seaborn is required. Install it via `pip install seaborn`.") from exc

from pandas.api.types import CategoricalDtype

BASE_DINO_ORDER: List[str] = ["small", "base", "large", "giant"]
VIT_SIZE_ORDER: List[str] = ["s", "b", "l", "g"]
USE_LORA_LABELS: List[str] = ["No LoRA", "LoRA"]
PALETTE: Dict[str, str] = {"No LoRA": "#355c7d", "LoRA": "#c06c84"}

METRIC_META: Dict[str, Dict[str, object]] = {
    "mean_iou": {"label": "Mean IoU", "ylim": (0.0, 1.0)},
    "foreground_iou": {"label": "Foreground IoU", "ylim": (0.0, 1.0)},
    "mean_dice": {"label": "Mean Dice", "ylim": (0.0, 1.0)},
}

SUMMARY_METRICS = tuple((key, meta["label"]) for key, meta in METRIC_META.items())
RUN_METRICS = tuple((key, meta["label"]) for key, meta in METRIC_META.items() if key != "mean_dice")
TREND_METRICS = (("foreground_iou", METRIC_META["foreground_iou"]["label"]),)


@dataclass
class SummaryBundle:
    summary: pd.DataFrame
    runs: pd.DataFrame
    metadata: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render visualization-ready plots from summarize_seg_results outputs. "
            "Both summary.csv and run_metrics.csv files are consumed."
        )
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path.home() / "Downloads" / "summary",
        help="Folder that contains run_metrics.csv, summary.csv, and summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "plots",
        help="Destination directory for the generated figure PNGs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution (dots per inch) for saved figures.",
    )
    parser.add_argument(
        "--interactive-html",
        type=Path,
        help="Optional path to save an interactive Plotly dashboard (requires plotly).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_dino_order(values: Iterable[str]) -> List[str]:
    unique: List[str] = []
    for raw in values:
        v = str(raw).strip().lower()
        if not v or v in unique:
            continue
        unique.append(v)

    index_map = {v: idx for idx, v in enumerate(unique)}

    def sort_key(v: str) -> Tuple[int, int, int, int]:
        if v in BASE_DINO_ORDER:
            return (0, BASE_DINO_ORDER.index(v), 0, index_map[v])
        match = re.match(r"vit([sblg])(\d+)?", v)
        if match:
            letter = match.group(1)
            patch = int(match.group(2) or 0)
            letter_rank = VIT_SIZE_ORDER.index(letter) if letter in VIT_SIZE_ORDER else len(VIT_SIZE_ORDER)
            return (1, letter_rank, patch, index_map[v])
        return (2, 0, 0, index_map[v])

    return sorted(unique, key=sort_key)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = (
        series.astype(str).str.strip().str.lower().map(mapping, na_action="ignore")
    )
    return normalized.fillna(False).astype(bool)


def prepare_frames(summary_path: Path, runs_path: Path, meta_path: Path) -> SummaryBundle:
    if not summary_path.is_file() or not runs_path.is_file():
        raise FileNotFoundError(
            "Both summary.csv and run_metrics.csv must exist inside the summary directory."
        )

    summary_df = pd.read_csv(summary_path)
    runs_df = pd.read_csv(runs_path)
    metadata = read_json(meta_path)

    dino_values = (
        summary_df["dino_size"].astype(str).str.lower().dropna().unique().tolist()
    )
    dino_values.extend(
        runs_df["dino_size"].astype(str).str.lower().dropna().unique().tolist()
    )
    dino_order = build_dino_order(dino_values)
    dino_dtype = CategoricalDtype(categories=dino_order, ordered=True)
    lora_dtype = CategoricalDtype(categories=USE_LORA_LABELS, ordered=True)

    summary_df["dino_size"] = (
        summary_df["dino_size"].astype(str).str.lower().astype(dino_dtype)
    )
    summary_df["dataset_type"] = summary_df["dataset_type"].astype(str).str.lower()
    summary_df["dataset_title"] = summary_df["dataset_type"].str.title()
    summary_df["use_lora"] = normalize_bool_series(summary_df["use_lora"])
    summary_df["use_lora_label"] = summary_df["use_lora"].map(
        {True: "LoRA", False: "No LoRA"}
    )
    summary_df["use_lora_label"] = summary_df["use_lora_label"].astype(lora_dtype)

    runs_df["dino_size"] = runs_df["dino_size"].astype(str).str.lower().astype(dino_dtype)
    runs_df["dataset_type"] = runs_df["dataset_type"].astype(str).str.lower()
    runs_df["dataset_title"] = runs_df["dataset_type"].str.title()
    runs_df["use_lora"] = normalize_bool_series(runs_df["use_lora"])
    runs_df["use_lora_label"] = runs_df["use_lora"].map(
        {True: "LoRA", False: "No LoRA"}
    )
    runs_df["use_lora_label"] = runs_df["use_lora_label"].astype(lora_dtype)

    metadata = dict(metadata)
    metadata["dino_order"] = dino_order
    return SummaryBundle(summary=summary_df, runs=runs_df, metadata=metadata)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def metric_limits(metric: str) -> Optional[Tuple[float, float]]:
    meta = METRIC_META.get(metric)
    if not meta:
        return None
    ylim = meta.get("ylim")
    if isinstance(ylim, Iterable):
        return tuple(ylim)  # type: ignore[return-value]
    return None


def stylize_axis(ax: plt.Axes, metric: str, label: str, x_label: str = "DINO size") -> None:
    limits = metric_limits(metric)
    if limits:
        ax.set_ylim(*limits)
    ax.set_ylabel(label)
    ax.set_xlabel(x_label)
    ax.grid(axis="y", color="#d0d0d0", linestyle="--", linewidth=0.8)


def add_error_bars(ax: plt.Axes, subset: pd.DataFrame, std_col: str) -> None:
    if std_col not in subset:
        return
    ordered_subset = subset.sort_values(["dino_size", "use_lora_label"])
    for patch, (_, row) in zip(ax.patches, ordered_subset.iterrows()):
        std_val = row.get(std_col)
        if pd.isna(std_val) or std_val == 0:
            continue
        ax.errorbar(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            yerr=std_val,
            ecolor="#333333",
            elinewidth=1.3,
            capsize=6,
        )


def plot_summary_metric(
    summary_df: pd.DataFrame,
    metric_prefix: str,
    label: str,
    output_path: Path,
    metadata: Dict,
    dpi: int,
) -> None:
    mean_col = f"{metric_prefix}_mean"
    std_col = f"{metric_prefix}_std"
    data = summary_df.dropna(subset=[mean_col]).copy()
    if data.empty:
        print(f"[warn] No summary rows available for {metric_prefix}, skipping plot.")
        return

    size_order = metadata.get("dino_order") or build_dino_order(data["dino_size"].unique())
    hue_order = [v for v in USE_LORA_LABELS if v in data["use_lora_label"].unique().tolist()]
    if not hue_order:
        hue_order = data["use_lora_label"].unique().tolist()

    g = sns.catplot(
        data=data,
        kind="bar",
        x="dino_size",
        y=mean_col,
        hue="use_lora_label",
        col="dataset_title",
        order=size_order,
        hue_order=hue_order,
        palette=PALETTE,
        height=4.2,
        aspect=1.1,
        legend_out=False,
        sharey=True,
        edgecolor="black",
    )
    for ax, title in zip(g.axes.flat, g.col_names):
        subset = data[data["dataset_title"] == title]
        add_error_bars(ax, subset, std_col)
        stylize_axis(ax, metric_prefix, label)
        ax.set_title(title)
    total_runs = metadata.get("num_runs")
    created_at = metadata.get("created_at")
    sub_title_bits = []
    if total_runs:
        sub_title_bits.append(f"{total_runs} runs")
    if created_at:
        sub_title_bits.append(f"summary {created_at}")
    subtitle = " | ".join(sub_title_bits)
    g.fig.suptitle(
        f"{label} by DINO size and LoRA" + (f" ({subtitle})" if subtitle else ""),
        y=0.98,
        fontsize=15,
    )
    g.add_legend(title="LoRA usage")
    if g._legend:
        g._legend.set_frame_on(False)
    g.fig.subplots_adjust(top=0.9, bottom=0.15)
    g.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[info] Wrote {output_path}")


def plot_run_metric_distribution(
    run_df: pd.DataFrame,
    metric: str,
    label: str,
    output_path: Path,
    dpi: int,
    dino_order: Optional[List[str]] = None,
) -> None:
    data = run_df.dropna(subset=[metric]).copy()
    if data.empty:
        print(f"[warn] No run-level rows available for {metric}, skipping plot.")
        return

    size_order = dino_order or build_dino_order(data["dino_size"].unique())
    if not size_order:
        size_order = data["dino_size"].unique().tolist()
    hue_order = [v for v in USE_LORA_LABELS if v in data["use_lora_label"].unique().tolist()]
    if not hue_order:
        hue_order = data["use_lora_label"].unique().tolist()

    try:
        g = sns.catplot(
            data=data,
            kind="box",
            x="dino_size",
            y=metric,
            hue="use_lora_label",
            col="dataset_title",
            order=size_order,
            hue_order=hue_order,
            palette=PALETTE,
            height=4.2,
            aspect=1.1,
            legend=False,
            sharey=True,
            linewidth=1.1,
            width=0.6,
            fliersize=0,
        )
        for ax, title in zip(g.axes.flat, g.col_names):
            subset = data[data["dataset_title"] == title]
            sns.stripplot(
                data=subset,
                x="dino_size",
                y=metric,
                hue="use_lora_label",
                order=size_order,
                hue_order=hue_order,
                dodge=True,
                palette=PALETTE,
                alpha=0.55,
                linewidth=0.3,
                edgecolor="white",
                ax=ax,
                legend=False,
                marker="o",
                size=4.5,
            )
            stylize_axis(ax, metric, label)
            ax.set_title(title)
        g.fig.suptitle(f"Run-level {label} distribution", y=0.98, fontsize=15)
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                label=label_name,
                markersize=10,
                markerfacecolor=PALETTE.get(label_name, "#666666"),
                markeredgecolor="#444444",
            )
            for label_name in hue_order
        ]
        g.fig.legend(
            handles=handles,
            loc="upper center",
            ncol=max(1, len(handles)),
            title="LoRA usage",
            frameon=False,
        )
        g.fig.subplots_adjust(top=0.88, bottom=0.17)
        g.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(g.fig)
        print(f"[info] Wrote {output_path}")
        return
    except UnboundLocalError as exc:
        if "boxprops" not in str(exc):
            raise
        print(f"[warn] seaborn boxplot failed ({exc}); falling back to manual plots.")

    titles = data["dataset_title"].unique().tolist()
    ncols = max(1, len(titles))
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, title in zip(axes, titles):
        subset = data[data["dataset_title"] == title]
        sns.boxplot(
            data=subset,
            x="dino_size",
            y=metric,
            hue="use_lora_label",
            order=size_order,
            hue_order=hue_order,
            palette=PALETTE,
            linewidth=1.1,
            width=0.6,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="dino_size",
            y=metric,
            hue="use_lora_label",
            order=size_order,
            hue_order=hue_order,
            dodge=True,
            palette=PALETTE,
            alpha=0.55,
            linewidth=0.3,
            edgecolor="white",
            ax=ax,
            marker="o",
            size=4.5,
        )
        if ax.legend_:
            ax.legend_.remove()
        stylize_axis(ax, metric, label)
        ax.set_title(title)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            label=label_name,
            markersize=10,
            markerfacecolor=PALETTE.get(label_name, "#666666"),
            markeredgecolor="#444444",
        )
        for label_name in hue_order
    ]
    fig.suptitle(f"Run-level {label} distribution", y=0.98, fontsize=15)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=max(1, len(handles)),
        title="LoRA usage",
        frameon=False,
    )
    fig.subplots_adjust(top=0.88, bottom=0.17)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Wrote {output_path}")


def plot_replicate_trend(
    run_df: pd.DataFrame,
    metric: str,
    label: str,
    output_path: Path,
    dpi: int,
    dino_order: Optional[List[str]] = None,
) -> None:
    data = run_df.dropna(subset=[metric, "replicate"]).copy()
    if data.empty:
        print(f"[warn] No replicate trend data for {metric}, skipping plot.")
        return

    data["replicate"] = data["replicate"].astype(int)
    size_order = dino_order or build_dino_order(data["dino_size"].unique())
    hue_order = [v for v in USE_LORA_LABELS if v in data["use_lora_label"].unique().tolist()]
    if not hue_order:
        hue_order = data["use_lora_label"].unique().tolist()
    g = sns.relplot(
        data=data,
        kind="line",
        x="replicate",
        y=metric,
        hue="use_lora_label",
        style="dino_size",
        markers=True,
        dashes=False,
        col="dataset_title",
        hue_order=hue_order,
        style_order=size_order,
        palette=PALETTE,
        height=4.0,
        aspect=1.2,
        legend=False,
        facet_kws={"sharey": True},
    )
    for ax in g.axes.flat:
        stylize_axis(ax, metric, label, x_label="Replicate")
    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[label_name],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=label_name,
        )
        for label_name in hue_order
    ]
    g.fig.legend(
        handles=handles,
        loc="upper center",
        title="LoRA usage",
        ncol=len(handles),
        frameon=False,
    )
    g.fig.suptitle(f"{label} vs replicate", y=0.98, fontsize=15)
    g.fig.subplots_adjust(top=0.9, bottom=0.15)
    g.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[info] Wrote {output_path}")


def export_interactive_dashboard(bundle: SummaryBundle, output_path: Path) -> None:
    try:
        import plotly.express as px
        from plotly.io import to_html
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "--interactive-html requested but plotly is not installed. "
            "Install it via `pip install plotly>=5`."
        ) from exc

    html_sections: List[str] = [
        "<h1 style='font-family:Helvetica,Arial,sans-serif;'>DINO-EM summary dashboard</h1>"
    ]
    dino_order = bundle.metadata.get("dino_order") or build_dino_order(bundle.summary["dino_size"].unique())
    category_orders = {"dino_size": dino_order, "use_lora_label": USE_LORA_LABELS}

    for metric, label in (("foreground_iou", "Foreground IoU"), ("mean_iou", "Mean IoU")):
        fig = px.bar(
            bundle.summary,
            x="dino_size",
            y=f"{metric}_mean",
            error_y=f"{metric}_std",
            color="use_lora_label",
            facet_col="dataset_title",
            barmode="group",
            category_orders=category_orders,
            color_discrete_map=PALETTE,
            title=f"{label} summary (interactive)",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(legend_title="LoRA usage", height=450)
        html_sections.append(to_html(fig, include_plotlyjs="cdn", full_html=False))

    run_fig = px.box(
        bundle.runs,
        x="dino_size",
        y="foreground_iou",
        color="use_lora_label",
        facet_col="dataset_title",
        points="all",
        category_orders=category_orders,
        color_discrete_map=PALETTE,
        title="Run-level Foreground IoU (interactive)",
    )
    run_fig.update_yaxes(range=[0, 1])
    run_fig.update_layout(legend_title="LoRA usage", height=450)
    html_sections.append(to_html(run_fig, include_plotlyjs=False, full_html=False))

    if "replicate" in bundle.runs.columns and bundle.runs["replicate"].notna().any():
        trend_fig = px.line(
            bundle.runs.dropna(subset=["replicate"]),
            x="replicate",
            y="foreground_iou",
            color="use_lora_label",
            line_dash="dino_size",
            markers=True,
            facet_col="dataset_title",
            category_orders=category_orders,
            color_discrete_map=PALETTE,
            title="Foreground IoU vs replicate (interactive)",
        )
        trend_fig.update_yaxes(range=[0, 1])
        trend_fig.update_layout(legend_title="LoRA usage", height=450)
        html_sections.append(to_html(trend_fig, include_plotlyjs=False, full_html=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = "\n".join(html_sections)
    shell = f"<!doctype html><html><head><meta charset='utf-8'><title>DINO-EM plots</title></head><body>{html}</body></html>"
    output_path.write_text(shell, encoding="utf-8")
    print(f"[info] Wrote interactive dashboard to {output_path}")


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.expanduser().resolve()
    output_dir = ensure_output_dir(args.output_dir.expanduser().resolve())

    bundle = prepare_frames(
        summary_dir / "summary.csv",
        summary_dir / "run_metrics.csv",
        summary_dir / "summary.json",
    )
    sns.set_theme(style="ticks", context="talk", font_scale=0.95)

    for metric, label in SUMMARY_METRICS:
        plot_summary_metric(
            bundle.summary,
            metric,
            label,
            output_dir / f"summary_{metric}.png",
            bundle.metadata,
            args.dpi,
        )

    for metric, label in RUN_METRICS:
        plot_run_metric_distribution(
            bundle.runs,
            metric,
            label,
            output_dir / f"runs_{metric}.png",
            args.dpi,
            bundle.metadata.get("dino_order"),
        )

    for metric, label in TREND_METRICS:
        plot_replicate_trend(
            bundle.runs,
            metric,
            label,
            output_dir / f"trend_{metric}.png",
            args.dpi,
            bundle.metadata.get("dino_order"),
        )

    if args.interactive_html:
        export_interactive_dashboard(bundle, args.interactive_html.expanduser().resolve())

    print(f"[done] Plots saved under {output_dir}")


if __name__ == "__main__":
    main()
