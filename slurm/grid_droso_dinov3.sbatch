#!/usr/bin/bash -l
#SBATCH --job-name=dino-grid-droso-dinov3
#SBATCH --time=48:00:00
#SBATCH --gpus=H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_CFG=${BASE_CFG:-configs/cluster/droso_dinov3_cluster.yaml}
DINO_SIZES=${DINO_SIZES:-"vits16 vitb16 vitl16"}
BACKBONE_NAME=${BACKBONE_NAME:-dinov3}
EPOCHS=${EPOCHS:-}
BATCH_DEFAULT=${BATCH_DEFAULT:-}
NUM_WORKERS=${NUM_WORKERS:-}
AMP=${AMP:-}
USE_LORA=${USE_LORA:-}
RUN_TAG=${RUN_TAG:-drosogrid-dinov3-${SLURM_JOB_ID:-manual}}
SEED=${SEED:-}
REPEATS=${REPEATS:-5}
WEIGHTS_VITS16=${WEIGHTS_VITS16:-}
WEIGHTS_VITB16=${WEIGHTS_VITB16:-}
WEIGHTS_VITL16=${WEIGHTS_VITL16:-}

read -r -a SIZE_LIST <<< "$DINO_SIZES"
NUM_SIZES=${#SIZE_LIST[@]}
if (( NUM_SIZES == 0 )); then
  echo "[error] DINO_SIZES is empty" >&2
  exit 1
fi
TOTAL_TASKS=$(( NUM_SIZES * REPEATS ))

archive_slurm_log() {
  local dest_dir="$1" tag="$2"
  local base="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  local src="$base"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    local with_suffix="${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    if [ -f "$with_suffix" ]; then
      src="$with_suffix"
    fi
  fi
  if [ -f "$src" ]; then
    local dest="${dest_dir}/logs/${tag}_slurm.out"
    mkdir -p "$(dirname "$dest")"
    mv "$src" "$dest"
    echo "[log] moved slurm log -> $dest"
  else
    echo "[log] slurm output not found at $src" >&2
  fi
}

prepare_cfg() {
  local base_cfg="$1" size="$2" epochs="$3" batch="$4" workers="$5" amp="$6" use_lora="$7" tag="$8" backbone_name="$9" weights_override="${10}" seed="${11}"
  "$PY" - "$base_cfg" "$size" "$epochs" "$batch" "$workers" "$amp" "$use_lora" "$tag" "$backbone_name" "$weights_override" "$seed" <<'PY'
import sys, yaml, tempfile, pathlib, re
base, size, epochs, batch, workers, amp, use_lora, tag, backbone_name, weights_override, seed = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["dino_size"] = size
backbone = cfg.get("backbone") or {}
backbone["name"] = backbone_name or backbone.get("name", "dinov3")
backbone["variant"] = size
if weights_override:
    backbone["weights"] = weights_override
else:
    base_weights = backbone.get("weights")
    if isinstance(base_weights, str) and size and "vits16" in base_weights.lower():
        backbone["weights"] = re.sub("vits16", size, base_weights, flags=re.IGNORECASE)
cfg["backbone"] = backbone
if seed:
    cfg["seed"] = int(seed)
if epochs:
    cfg["epochs"] = max(1, int(epochs))
if batch:
    cfg["batch_size"] = max(1, int(batch))
if workers:
    cfg["num_workers"] = max(0, int(workers))
if amp:
    cfg["amp"] = amp.lower() not in ("0","false","no","off")
if use_lora:
    use = use_lora.lower() not in ("0","false","no","off")
    cfg["use_lora"] = use
    lora_cfg = cfg.get("lora") or {}
    lora_cfg["enabled"] = use
    cfg["lora"] = lora_cfg
tag = tag or "grid"
task_type = cfg.get("task_type", "seg")
default_out = pathlib.Path(cfg.get("out_dir", "runs/exp")).expanduser()
exp_base = cfg.get("experiment_id") or default_out.name
if not cfg.get("use_lora", True):
    if "lora" in exp_base.lower():
        exp_base = re.sub("lora", "head", exp_base, flags=re.IGNORECASE)
    else:
        exp_base = f"{exp_base}_head"
cfg["experiment_id"] = f"{exp_base}_{tag}"
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

get_batch_for_size() {
  local size_key="${1^^}"
  local var="BATCH_${size_key}"
  local val="${!var:-}"
  if [ -n "$val" ]; then
    echo "$val"
  elif [ -n "$BATCH_DEFAULT" ]; then
    echo "$BATCH_DEFAULT"
  else
    echo ""
  fi
}

run_config() {
  local size="$1"
  local size_idx="$2"
  local rep_id="$3"
  local size_clean="${size// /}"
  local size_tag="${RUN_TAG}_rep${rep_id}_s$((size_idx + 1))_${size_clean}"
  local batch_val
  batch_val="$(get_batch_for_size "$size_clean")"
  local weights_override=""
  case "$size_clean" in
    vits16) weights_override="$WEIGHTS_VITS16" ;;
    vitb16) weights_override="$WEIGHTS_VITB16" ;;
    vitl16) weights_override="$WEIGHTS_VITL16" ;;
  esac
  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$size_clean" "$EPOCHS" "$batch_val" "$NUM_WORKERS" "$AMP" "$USE_LORA" "$size_tag" "$BACKBONE_NAME" "$weights_override" "$SEED")
  runtime_cfg="${prep[0]}"
  run_dir="${prep[1]}"
  echo "[grid size=${size_clean} rep=${rep_id}] batch=$batch_val lora=$USE_LORA cfg=$runtime_cfg"
  "$PY" scripts/train_em_seg.py --cfg "$runtime_cfg"
  "$PY" scripts/eval_em_seg.py --cfg "$runtime_cfg" --out_csv "$run_dir/${size_tag}_metrics.csv"
  archive_slurm_log "$run_dir" "$size_tag"
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
