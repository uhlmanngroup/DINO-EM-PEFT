#!/usr/bin/bash -l
#SBATCH --job-name=dino-domain
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Ensure CUDA libs are visible for torch/pytorch-fid even though we keep the job CPU-bound.
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "[error] Python not found at $PY" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CFG_PATH="${CFG_PATH:-configs/mac/domain_analysis.yaml}"
EXTRA_ARGS=()

if [ ! -f "$CFG_PATH" ]; then
  echo "[error] Config file not found: $CFG_PATH" >&2
  exit 2
fi

echo "[domain] cfg=$CFG_PATH"

"$PY" scripts/run_domain_analysis.py --cfg "$CFG_PATH" "${EXTRA_ARGS[@]}"
