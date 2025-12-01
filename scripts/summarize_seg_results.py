#!/usr/bin/env python3
"""Summarize segmentation experiment runs into CSV/JSON reports.

Cluster reminder (plain CLI):

    conda activate dino-peft
    python scripts/summarize_seg_results.py --root /scratch/$USER/Results/DINO-LoRA/seg

Tweak `--root` if the runs live elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as stats
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

METRIC_KEYS: Tuple[str, ...] = (
    "mean_dice",
    "mean_iou",
    "foreground_dice",
    "foreground_iou",
)


@dataclass
class RunRecord:
    """Container for per-run metrics and metadata."""

    run_dir: Path
    rel_run_dir: str
    experiment_id: str
    dino_size: str
    use_lora: bool
    replicate: Optional[int]
    seed: Optional[int]
    loss: Optional[str]
    dataset_type: Optional[str]
    pair_mode: Optional[str]
    mean_dice: Optional[float]
    mean_iou: Optional[float]
    foreground_dice: Optional[float]
    foreground_iou: Optional[float]
    best_val_loss: Optional[float]
    best_epoch: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate segmentation experiment metrics across DINO sizes "
            "and LoRA configurations."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Experiment directory that contains run folders.",
    )
    parser.add_argument(
        "--summary-dir-name",
        default="summary",
        help="Name of the folder that will store the generated reports.",
    )
    parser.add_argument(
        "--config-candidates",
        nargs="+",
        default=("config_used.yaml", "config.yaml"),
        help="Config filenames to probe in each run directory.",
    )
    parser.add_argument(
        "--metrics-filename",
        default="metrics.json",
        help="Name of the metrics file inside each run directory.",
    )
    return parser.parse_args()


def safe_relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_config(run_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        cfg_path = run_dir / candidate
        if cfg_path.is_file():
            return cfg_path
    return None


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def extract_replicate(name: str) -> Optional[int]:
    match = re.search(r"rep(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def size_rank(size: Optional[str]) -> int:
    order = {"tiny": 0, "small": 1, "base": 2, "large": 3, "giant": 4, "huge": 5}
    return order.get((size or "").lower(), len(order))


def collect_runs(
    root: Path, summary_dir: Path, metrics_filename: str, config_candidates: Sequence[str]
) -> List[RunRecord]:
    run_records: List[RunRecord] = []
    for metrics_path in sorted(root.rglob(metrics_filename)):
        if summary_dir in metrics_path.parents:
            continue
        run_dir = metrics_path.parent
        cfg_path = find_config(run_dir, config_candidates)
        if cfg_path is None:
            print(f"[warn] config missing for {run_dir}", flush=True)
            continue
        try:
            metrics = read_json(metrics_path)
        except json.JSONDecodeError as err:
            print(f"[warn] skipping {metrics_path}: {err}", flush=True)
            continue
        try:
            cfg = read_yaml(cfg_path) or {}
        except yaml.YAMLError as err:
            print(f"[warn] skipping {cfg_path}: {err}", flush=True)
            continue

        eval_metrics = metrics.get("eval", {})
        train_metrics = metrics.get("train", {})

        run_records.append(
            RunRecord(
                run_dir=run_dir,
                rel_run_dir=safe_relative_to(run_dir, root),
                experiment_id=cfg.get("experiment_id", run_dir.name),
                dino_size=str(cfg.get("dino_size", "unknown")).lower(),
                use_lora=bool(cfg.get("use_lora", False)),
                replicate=extract_replicate(run_dir.name),
                seed=train_metrics.get("seed"),
                loss=cfg.get("loss"),
                dataset_type=(cfg.get("dataset") or {}).get("type"),
                pair_mode=((cfg.get("dataset") or {}).get("params") or {}).get(
                    "pair_mode"
                ),
                mean_dice=eval_metrics.get("mean_dice"),
                mean_iou=eval_metrics.get("mean_iou"),
                foreground_dice=eval_metrics.get("foreground_dice"),
                foreground_iou=eval_metrics.get("foreground_iou"),
                best_val_loss=train_metrics.get("best_val_loss"),
                best_epoch=train_metrics.get("best_epoch"),
            )
        )
    return run_records


def compute_stats(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(stats.mean(vals)), float(stats.stdev(vals))


def build_aggregates(run_records: List[RunRecord]) -> List[Dict]:
    grouped: Dict[Tuple[str, bool, str], Dict[str, List[float]]] = {}
    counts: Dict[Tuple[str, bool, str], int] = {}

    for record in run_records:
        ds_type = (record.dataset_type or "unknown").lower()
        key = (ds_type, record.use_lora, record.dino_size)
        if key not in grouped:
            grouped[key] = {metric: [] for metric in METRIC_KEYS}
            counts[key] = 0
        counts[key] += 1
        for metric in METRIC_KEYS:
            value = getattr(record, metric)
            if isinstance(value, (int, float)):
                grouped[key][metric].append(float(value))

    def _sort_key(item: Tuple[Tuple[str, bool, str], Dict[str, List[float]]]):
        (ds_type, use_lora, size), _ = item
        return (ds_type, int(use_lora), size_rank(size), size)

    aggregates: List[Dict] = []
    for key, metric_lists in sorted(grouped.items(), key=_sort_key):
        dataset_type, use_lora, size = key
        row: Dict[str, Optional[float]] = {
            "dataset_type": dataset_type,
            "use_lora": use_lora,
            "dino_size": size,
            "num_runs": counts.get(key, 0),
        }
        for metric, values in metric_lists.items():
            mean_val, std_val = compute_stats(values)
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
        aggregates.append(row)
    return aggregates


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def serialize_run_record(record: RunRecord) -> Dict[str, object]:
    return {
        "run_dir": str(record.run_dir),
        "rel_run_dir": record.rel_run_dir,
        "experiment_id": record.experiment_id,
        "dino_size": record.dino_size,
        "use_lora": record.use_lora,
        "replicate": record.replicate,
        "seed": record.seed,
        "loss": record.loss,
        "dataset_type": record.dataset_type,
        "pair_mode": record.pair_mode,
        "mean_dice": record.mean_dice,
        "mean_iou": record.mean_iou,
        "foreground_dice": record.foreground_dice,
        "foreground_iou": record.foreground_iou,
        "best_val_loss": record.best_val_loss,
        "best_epoch": record.best_epoch,
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().absolute()
    if not root.exists():
        raise SystemExit(f"root directory not found: {root}")
    summary_dir = root / args.summary_dir_name
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] scanning runs under {root}")
    run_records = collect_runs(
        root=root,
        summary_dir=summary_dir,
        metrics_filename=args.metrics_filename,
        config_candidates=args.config_candidates,
    )
    if not run_records:
        print("[warn] no runs discovered, nothing to summarize.")
        return
    aggregates = build_aggregates(run_records)

    run_fieldnames = [
        "run_dir",
        "rel_run_dir",
        "experiment_id",
        "use_lora",
        "dino_size",
        "replicate",
        "seed",
        "loss",
        "dataset_type",
        "pair_mode",
        "mean_dice",
        "mean_iou",
        "foreground_dice",
        "foreground_iou",
        "best_val_loss",
        "best_epoch",
    ]
    summary_fieldnames = [
        "dataset_type",
        "use_lora",
        "dino_size",
        "num_runs",
        "mean_dice_mean",
        "mean_dice_std",
        "mean_iou_mean",
        "mean_iou_std",
        "foreground_dice_mean",
        "foreground_dice_std",
        "foreground_iou_mean",
        "foreground_iou_std",
    ]

    run_rows = [serialize_run_record(record) for record in run_records]
    write_csv(summary_dir / "run_metrics.csv", run_rows, run_fieldnames)
    write_csv(summary_dir / "summary.csv", aggregates, summary_fieldnames)

    manifest = {
        "root": str(root),
        "summary_dir": str(summary_dir),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_runs": len(run_records),
        "run_metrics_csv": str(summary_dir / "run_metrics.csv"),
        "summary_csv": str(summary_dir / "summary.csv"),
        "aggregate_metrics": aggregates,
    }
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"[info] wrote {summary_dir / 'run_metrics.csv'}")
    print(f"[info] wrote {summary_dir / 'summary.csv'}")
    print(f"[info] wrote {summary_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
