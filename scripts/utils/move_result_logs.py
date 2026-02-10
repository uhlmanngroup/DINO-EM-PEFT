#!/usr/bin/env python3
"""
Utility to backfill Slurm log files into their corresponding run directories.

Usage:
    python scripts/utils/move_result_logs.py \
        --logs-dir /path/to/slurm/logs \
        --runs-dir /path/to/results/seg
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def extract_job_id(text: str) -> str | None:
    """Return the numeric job id embedded in a path-like string."""
    m = re.search(r"(?:pairedgrid|lucchigrid|drosogrid|droso)(?:-[^-_]+)?-(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"-(\d{5,})(?:_|$)", text.replace(".out", ""))
    if m:
        return m.group(1)
    return None


def find_run_dirs(root: Path) -> dict[str, Path]:
    """Map job ids to existing run directories (searching recursively)."""
    mapping: dict[str, Path] = {}
    for run_dir in root.rglob("*"):
        if not run_dir.is_dir():
            continue
        job_id = extract_job_id(run_dir.name)
        if job_id:
            mapping[job_id] = run_dir
    return mapping


def tag_from_run_name(name: str) -> str:
    """Derive the tag portion that follows `_seg_` in the run directory name."""
    parts = name.split("_seg_", 1)
    return parts[1] if len(parts) == 2 else name


def archive_logs(logs_dir: Path, runs_dir: Path, dry_run: bool = False) -> None:
    job_to_run = find_run_dirs(runs_dir)
    moved = 0
    skipped = 0

    for log_path in sorted(logs_dir.glob("*.out")):
        job_id = extract_job_id(log_path.stem)
        if not job_id:
            skipped += 1
            print(f"[skip] {log_path.name}: could not parse job id")
            continue

        run_dir = job_to_run.get(job_id)
        if not run_dir:
            skipped += 1
            print(f"[skip] {log_path.name}: no matching run directory for job {job_id}")
            continue

        tag = tag_from_run_name(run_dir.name)
        dest = run_dir / "logs" / f"{tag}_slurm.out"
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            print(f"[dry-run] would move {log_path} -> {dest}")
            continue

        if dest.exists():
            dest.unlink()

        shutil.move(str(log_path), str(dest))
        moved += 1
        print(f"[moved] {log_path.name} -> {dest}")

    print(f"Done. moved={moved} skipped={skipped}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", required=True, type=Path)
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="Do not move files, just print actions")
    args = ap.parse_args()

    archive_logs(args.logs_dir.expanduser(), args.runs_dir.expanduser(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
