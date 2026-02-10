#!/usr/bin/bash -l
#SBATCH --job-name=dino-feats-paired
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module --quiet load miniforge3 || module load miniforge3

PY="$HOME/data/conda/envs/dino-peft/bin/python"
[ -x "$PY" ] || { echo "[error] Python not found at $PY" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TQDM_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CFG="${BASE_CFG:-configs/cluster/paired_feat_analysis_cluster.yaml}"
DINO_SIZES=${DINO_SIZES:-"small base large giant"}
BACKBONE_NAME=${BACKBONE_NAME:-dinov2}
BATCH_DEFAULT=${BATCH_DEFAULT:-${BATCH_SIZE:-}}
NUM_WORKERS=${NUM_WORKERS:-}
DEVICE=${DEVICE:-cpu}
DATA_DIR_OVERRIDE=${DATA_DIR_OVERRIDE:-}
TASK_TYPE_OVERRIDE=${TASK_TYPE_OVERRIDE:-}
RUN_TAG=${RUN_TAG:-featgrid${SLURM_JOB_ID:-manual}}
CHECKPOINT_DEFAULT=${CHECKPOINT_DEFAULT:-}
REPEATS=${REPEATS:-1}

read -r -a SIZE_LIST <<< "$DINO_SIZES"
NUM_SIZES=${#SIZE_LIST[@]}
if (( NUM_SIZES == 0 )); then
  echo "[error] DINO_SIZES is empty" >&2
  exit 1
fi
if (( REPEATS < 1 )); then
  echo "[error] REPEATS must be >=1" >&2
  exit 1
fi
TOTAL_TASKS=$(( NUM_SIZES * REPEATS ))

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

get_checkpoint_for_size() {
  local size_key="${1^^}"
  local var="CHECKPOINT_${size_key}"
  local val="${!var:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    echo "$CHECKPOINT_DEFAULT"
  fi
}

get_batch_for_size() {
  local size_key="${1^^}"
  local var="BATCH_${size_key}"
  local val="${!var:-}"
  if [ -n "$val" ]; then
    echo "$val"
  else
    echo "$BATCH_DEFAULT"
  fi
}

prepare_cfg() {
  local base_cfg="$1" size="$2" tag="$3" batch="$4" workers="$5" device="$6" data_dir="$7" task_type="$8" checkpoint="$9" backbone_name="${10}"
  "$PY" - "$base_cfg" "$size" "$tag" "$batch" "$workers" "$device" "$data_dir" "$task_type" "$checkpoint" "$backbone_name" <<'PY'
import sys, yaml, tempfile, pathlib
base, size, tag, batch, workers, device, data_dir, task_type, checkpoint, backbone_name = sys.argv[1:]
cfg = yaml.safe_load(open(base))
model = cfg.setdefault("model", {})
model["dino_size"] = size
backbone = model.get("backbone") or {}
backbone["name"] = backbone_name or backbone.get("name", "dinov2")
backbone["variant"] = size
model["backbone"] = backbone
if checkpoint:
    model["checkpoint"] = checkpoint
runtime = cfg.setdefault("runtime", {})
if batch:
    runtime["batch_size"] = max(1, int(batch))
if workers:
    runtime["num_workers"] = max(0, int(workers))
if device:
    runtime["device"] = device
data_cfg = cfg.setdefault("data", {})
if data_dir:
    data_cfg["data_dir"] = data_dir
if task_type:
    cfg["task_type"] = task_type
tag = tag or size
task = cfg.get("task_type") or "feat-analysis"
default_root = pathlib.Path(cfg.get("results_root", "runs/feat")).expanduser()
cfg["results_root"] = str(default_root)
exp_base = cfg.get("experiment_id") or "feat"
cfg["experiment_id"] = f"{exp_base}_{tag}"
task_path = pathlib.Path(task)
run_dir = default_root / task_path / cfg["experiment_id"]
cfg["out_dir"] = str(run_dir)
data_cfg["input_path"] = str(run_dir / "features.npz")
tmp = tempfile.NamedTemporaryFile(prefix=f"{tag}_", suffix=".yaml", delete=False)
with open(tmp.name, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(tmp.name)
print(run_dir)
print(data_cfg["input_path"])
PY
}

run_config() {
  local size="$1"
  local size_idx="$2"
  local rep_id="$3"
  local size_clean="${size// /}"
  local tag="${RUN_TAG}_rep${rep_id}_s$((size_idx + 1))_${size_clean}"
  local batch_override
  batch_override="$(get_batch_for_size "$size_clean")"
  local checkpoint_path
  checkpoint_path="$(get_checkpoint_for_size "$size_clean")"
  if [ -n "$checkpoint_path" ] && [ ! -f "$checkpoint_path" ]; then
    echo "[error] checkpoint not found for size=$size_clean -> $checkpoint_path" >&2
    exit 2
  fi
  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$size_clean" "$tag" "$batch_override" "$NUM_WORKERS" "$DEVICE" "$DATA_DIR_OVERRIDE" "$TASK_TYPE_OVERRIDE" "$checkpoint_path" "$BACKBONE_NAME")
  local runtime_cfg="${prep[0]}"
  local run_dir="${prep[1]}"
  local features_path="${prep[2]}"
  echo "[feat] size=${size_clean} rep=${rep_id} cfg=$runtime_cfg run_dir=$run_dir"
  "$PY" scripts/extract_features.py --cfg "$runtime_cfg"
  if [ ! -f "$features_path" ]; then
    echo "[error] expected features at $features_path" >&2
    exit 3
  fi
  "$PY" scripts/run_pca.py "$runtime_cfg"
  archive_slurm_log "$run_dir" "$tag"
  rm -f "$runtime_cfg"
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  task_id=${SLURM_ARRAY_TASK_ID}
  if (( task_id < 0 || task_id >= TOTAL_TASKS )); then
    echo "[error] SLURM_ARRAY_TASK_ID=$task_id outside 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
  fi
  size_index=$(( task_id % NUM_SIZES ))
  rep_index=$(( task_id / NUM_SIZES + 1 ))
  run_config "${SIZE_LIST[$size_index]}" "$size_index" "$rep_index"
else
  echo "[info] SLURM_ARRAY_TASK_ID not set; running all $TOTAL_TASKS configs sequentially"
  for ((rep=1; rep<=REPEATS; rep++)); do
    for ((idx=0; idx<NUM_SIZES; idx++)); do
      run_config "${SIZE_LIST[$idx]}" "$idx" "$rep"
    done
  done
fi
