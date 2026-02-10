#!/usr/bin/bash -l
#SBATCH --job-name=param-counts
#SBATCH --time=01:00:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CFG=${CFG:-configs/cluster/param_counts.yaml}
OUT=${OUT:-}
DEVICE=${DEVICE:-}
SKIP_FAILURES=${SKIP_FAILURES:-}

EXTRA_ARGS=()
if [ -n "$OUT" ]; then
  EXTRA_ARGS+=(--output "$OUT")
fi
if [ -n "$DEVICE" ]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi
if [ -n "$SKIP_FAILURES" ]; then
  EXTRA_ARGS+=(--skip-failures)
fi

echo "[params] cfg=$CFG"
if [ -n "$OUT" ]; then
  echo "[params] output=$OUT"
fi

"$PY" scripts/utils/param_count_table.py --cfg "$CFG" "${EXTRA_ARGS[@]}"
