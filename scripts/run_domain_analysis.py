#!/usr/bin/env python3
"""Compute FID + linear separability metrics between two feature domains.

Example (local):
    python scripts/run_domain_analysis.py --cfg configs/mac/domain_analysis.yaml

Example (cluster):
    sbatch slurm/domain_analysis.sbatch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from dino_peft.analysis.domain_metrics import (
    compute_fid_from_features,
    fit_pca_on_concat,
    load_features,
    lr_domain_separability,
)
from dino_peft.utils.paths import setup_run_dir, update_metrics, write_run_info

DEFAULT_CFG = Path(__file__).parent.parent / "configs" / "mac" / "domain_analysis.yaml"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Domain gap analysis based on DINO features.")
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        default=str(DEFAULT_CFG),
        help="Path to YAML config.",
    )
    return ap.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as handle:
        cfg = yaml.safe_load(handle)
    if not cfg:
        raise ValueError(f"Config file {path} is empty or invalid.")
    return cfg


def _resolve_output_paths(
    domain_cfg: Dict[str, Any],
    domain1_path: Path,
) -> tuple[Path, Path, Path]:
    """Return (output_dir, metrics_json_path, metrics_csv_path)."""
    legacy_out = domain_cfg.get("out_json")
    if legacy_out:
        json_path = Path(legacy_out)
        if not json_path.is_absolute():
            json_path = domain1_path.parent / json_path
        json_path = json_path.expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_name = domain_cfg.get("csv_name")
        csv_path = (
            json_path.with_suffix(".csv")
            if not csv_name
            else (json_path.parent / csv_name)
        )
        csv_path = csv_path.expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return json_path.parent, json_path, csv_path

    out_dir = domain_cfg.get("out_dir")
    if out_dir:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = domain1_path.parent / out_dir
    else:
        subdir = domain_cfg.get("output_subdir", "domain-analysis")
        out_dir = domain1_path.parent / subdir
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_name = domain_cfg.get("json_name", "metrics.json")
    csv_name = domain_cfg.get("csv_name", "metrics.csv")
    json_path = (out_dir / json_name).expanduser()
    csv_path = (out_dir / csv_name).expanduser()
    return out_dir, json_path, csv_path


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg).expanduser()
    print(f"[domain_analysis] Using config file: {cfg_path}")
    cfg = load_config(cfg_path)
    domain_cfg = cfg.get("domain_analysis") or {}
    required_keys = ["domain1", "domain2", "n_components", "seed"]
    missing = [k for k in required_keys if k not in domain_cfg]
    if missing:
        raise ValueError(f"Config domain_analysis missing keys: {missing}")

    domain1_cfg = domain_cfg["domain1"]
    domain2_cfg = domain_cfg["domain2"]
    n_components = int(domain_cfg["n_components"])
    seed = int(domain_cfg["seed"])

    run_dir = None
    task_type = cfg.get("task_type", "domain-analysis")
    if "experiment_id" in cfg and "results_root" in cfg:
        run_dir = setup_run_dir(cfg, task_type=task_type, subdirs=None)
        write_run_info(
            run_dir,
            {
                "task_type": task_type,
                "domain1_npz": str(domain_cfg["domain1"].get("npz")),
                "domain2_npz": str(domain_cfg["domain2"].get("npz")),
                "pca_components": n_components,
                "seed": seed,
            },
        )

    def _normalize_sequence(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _load_domain(entry: Dict[str, Any]):
        if "npz" not in entry:
            raise ValueError("Each domain definition must include an 'npz' path.")
        path = Path(entry["npz"]).expanduser()
        dataset_ids = _normalize_sequence(entry.get("dataset_ids"))
        if dataset_ids is not None:
            dataset_ids = [int(v) for v in dataset_ids]
        dataset_names = _normalize_sequence(entry.get("dataset_names"))
        feats = load_features(path, dataset_ids=dataset_ids, dataset_names=dataset_names)
        return path, feats, dataset_ids, dataset_names

    domain1_path, feats_a, domain1_ids, domain1_names = _load_domain(domain1_cfg)
    domain2_path, feats_b, domain2_ids, domain2_names = _load_domain(domain2_cfg)
    output_dir, out_json, out_csv = _resolve_output_paths(domain_cfg, domain1_path)
    print(f"[domain_analysis] Output directory: {output_dir}")

    def _fmt_filter(ids, names):
        parts = []
        if ids:
            parts.append(f"dataset_ids={ids}")
        if names:
            parts.append(f"dataset_names={names}")
        return ", ".join(parts) if parts else "all samples"

    print(
        "[domain_analysis] Domain1: "
        f"{domain1_path} (N={feats_a.shape[0]}, D={feats_a.shape[1]}; {_fmt_filter(domain1_ids, domain1_names)})"
    )
    print(
        "[domain_analysis] Domain2: "
        f"{domain2_path} (N={feats_b.shape[0]}, D={feats_b.shape[1]}; {_fmt_filter(domain2_ids, domain2_names)})"
    )

    # Compute FDD (FID-style metric) on the raw DINO features before any PCA reduction.
    fdd_value = compute_fid_from_features(feats_a, feats_b)
    _, feats_a_pca, feats_b_pca = fit_pca_on_concat(
        feats_a,
        feats_b,
        n_components=n_components,
        random_state=seed,
    )
    lr_metrics = lr_domain_separability(
        feats_a_pca,
        feats_b_pca,
        random_state=seed,
    )

    print(
        "[domain_analysis] FDD (FID-style on raw DINO features) "
        f"between domain1 and domain2: {fdd_value:.4f}"
    )
    print(
        "[domain_analysis] Logistic regression domain separability: "
        f"accuracy={lr_metrics['accuracy'] * 100:.2f}%, "
        f"roc_auc={lr_metrics['roc_auc'] * 100:.2f}"
    )

    payload = {
        "metrics": {
            "frechet_dino_distance": fdd_value,
            "logreg_accuracy": lr_metrics["accuracy"],
            "logreg_roc_auc": lr_metrics["roc_auc"],
        },
        "config": {
            "domain1": {
                "npz": str(domain1_path),
                "dataset_ids": domain1_ids,
                "dataset_names": domain1_names,
                "num_samples": int(feats_a.shape[0]),
            },
            "domain2": {
                "npz": str(domain2_path),
                "dataset_ids": domain2_ids,
                "dataset_names": domain2_names,
                "num_samples": int(feats_b.shape[0]),
            },
            "n_components": n_components,
            "seed": seed,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with out_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        writer.writerow(["frechet_dino_distance", fdd_value])
        writer.writerow(["logreg_accuracy", lr_metrics["accuracy"]])
        writer.writerow(["logreg_roc_auc", lr_metrics["roc_auc"]])
    print(f"[domain_analysis] Saved metrics JSON to {out_json}")
    print(f"[domain_analysis] Saved metrics CSV to {out_csv}")

    if run_dir is not None and domain_cfg.get("use_run_dir_metrics", False):
        update_metrics(
            run_dir,
            "domain_analysis",
            {
                "frechet_dino_distance": fdd_value,
                "logreg_accuracy": lr_metrics["accuracy"],
                "logreg_roc_auc": lr_metrics["roc_auc"],
                "domain1_npz": str(domain1_path),
                "domain2_npz": str(domain2_path),
                "pca_components": n_components,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[domain_analysis] ERROR: {exc}", file=sys.stderr)
        raise
