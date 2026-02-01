#!/usr/bin/bash -l
#SBATCH --job-name=seg-summary
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "[error] Python not found at $PY" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ROOT=${ROOT:-../../data/DINO-LoRA/openclip/seg}
SUMMARY_DIR=${SUMMARY_DIR:-}
EXTRA_ARGS=()

if [ -n "$SUMMARY_DIR" ]; then
  EXTRA_ARGS+=(--summary-dir "$SUMMARY_DIR")
fi

echo "[summary] root=$ROOT"
if [ -n "$SUMMARY_DIR" ]; then
  echo "[summary] summary_dir=$SUMMARY_DIR"
fi

"$PY" scripts/analysis/summarize_seg_results.py --root "$ROOT" "${EXTRA_ARGS[@]}"
