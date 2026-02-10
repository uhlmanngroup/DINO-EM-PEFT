#!/usr/bin/bash -l
#SBATCH --job-name=ablation-paired-balance
#SBATCH --time=72:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-4

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
CFG="configs/cluster/paired_dinov2_paired_balance.yaml"

echo "[ablation] seed=${SEED} cfg=${CFG}"

"$PY" scripts/ablation_paired_balance_dinov2.py \
  --cfg "$CFG" \
  --seed "$SEED" \
  --variant both
