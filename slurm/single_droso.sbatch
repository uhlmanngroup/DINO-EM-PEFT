#!/usr/bin/bash -l
#SBATCH --job-name=dino-single-droso
#SBATCH --time=48:00:00
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

BASE_CFG=${BASE_CFG:-configs/cluster/droso_cluster.yaml}
DINO_SIZE=${DINO_SIZE:-base}
BACKBONE_NAME=${BACKBONE_NAME:-dinov2}
BACKBONE_VARIANT=${BACKBONE_VARIANT:-$DINO_SIZE}
EPOCHS=${EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}
AMP=${AMP:-}
RUN_TAG=${RUN_TAG:-droso${SLURM_JOB_ID:-manual}}

archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local array_suffix=""
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    array_suffix="_${SLURM_ARRAY_TASK_ID}"
  fi
  local src="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}${array_suffix}.out"
  if [ -f "$src" ]; then
    local dest="${dest_dir}/logs/${tag}_slurm.out"
    mkdir -p "$(dirname "$dest")"
    mv "$src" "$dest"
    echo "[log] moved slurm log → $dest"
  else
    echo "[log] slurm output not found at $src" >&2
  fi
}

prepare_cfg() {
  local base_cfg="$1" size="$2" epochs="$3" batch="$4" workers="$5" amp="$6" tag="$7" backbone_name="$8"
  "$PY" - "$base_cfg" "$size" "$epochs" "$batch" "$workers" "$amp" "$tag" "$backbone_name" <<'PY'
import sys, yaml, tempfile, pathlib
base, size, epochs, batch, workers, amp, tag, backbone_name = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["dino_size"] = size
backbone = cfg.get("backbone") or {}
backbone["name"] = backbone_name or backbone.get("name", "dinov2")
backbone["variant"] = size
cfg["backbone"] = backbone
if epochs:
    cfg["epochs"] = max(1, int(epochs))
if batch:
    cfg["batch_size"] = max(1, int(batch))
if workers:
    cfg["num_workers"] = max(0, int(workers))
if amp:
    cfg["amp"] = amp.lower() not in ("0","false","no","off")
tag = tag or "droso"
task_type = cfg.get("task_type", "seg")
default_out = pathlib.Path(cfg.get("out_dir", "runs/exp")).expanduser()
base_exp = cfg.get("experiment_id") or default_out.name
cfg["experiment_id"] = f"{base_exp}_{tag}"
root = pathlib.Path(cfg.get("results_root", default_out.parent)).expanduser()
cfg["results_root"] = str(root)
out_dir = root / task_type / cfg["experiment_id"]
cfg["out_dir"] = str(out_dir)
tmp = tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".yaml", delete=False)
with open(tmp.name, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(tmp.name)
print(cfg["out_dir"])
PY
}

mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$BACKBONE_VARIANT" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$AMP" "$RUN_TAG" "$BACKBONE_NAME")
RUNTIME_CFG="${prep[0]}"
RUN_DIR="${prep[1]}"

echo "[single] cfg=$RUNTIME_CFG  run_dir=$RUN_DIR"
trap 'rm -f "$RUNTIME_CFG"' EXIT

"$PY" scripts/train_em_seg.py --cfg "$RUNTIME_CFG"
"$PY" scripts/eval_em_seg.py --cfg "$RUNTIME_CFG" --out_csv "$RUN_DIR/${RUN_TAG}_metrics.csv"

archive_slurm_log "$RUN_DIR" "$RUN_TAG"
