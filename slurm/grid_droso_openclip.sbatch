#!/usr/bin/bash -l
#SBATCH --job-name=grid-droso-openclip
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

BASE_CFG="configs/cluster/droso_openclip_vitl14_cluster.yaml"
MODELS=${MODELS:-"ViT-L-14 ViT-H-14"}
PRETRAINED_DEFAULT=${PRETRAINED_DEFAULT:-"laion2b_s32b_b82k"}
PRETRAINED_VITL14=${PRETRAINED_VITL14:-"laion2b_s32b_b82k"}
PRETRAINED_VITH14=${PRETRAINED_VITH14:-"laion2b_s32b_b79k"}
WEIGHTS_VITL14=${WEIGHTS_VITL14:-}
WEIGHTS_VITH14=${WEIGHTS_VITH14:-}
EPOCHS=${EPOCHS:-}
BATCH_DEFAULT=${BATCH_DEFAULT:-}
NUM_WORKERS=${NUM_WORKERS:-}
AMP=${AMP:-}
RUN_TAG=${RUN_TAG:-drosogrid-openclip-${SLURM_JOB_ID:-manual}}
REPEATS=${REPEATS:-5}
SEED_BASE=${SEED_BASE:-0}

read -r -a MODEL_LIST <<< "$MODELS"
NUM_MODELS=${#MODEL_LIST[@]}
if (( NUM_MODELS == 0 )); then
  echo "[error] MODELS is empty" >&2
  exit 1
fi
TOTAL_TASKS=$(( NUM_MODELS * REPEATS ))

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
    echo "[log] moved slurm log -> $dest"
  else
    echo "[log] slurm output not found at $src" >&2
  fi
}

get_pretrained() {
  case "$1" in
    ViT-L-14) echo "${PRETRAINED_VITL14:-$PRETRAINED_DEFAULT}" ;;
    ViT-H-14) echo "${PRETRAINED_VITH14:-$PRETRAINED_DEFAULT}" ;;
    *) echo "$PRETRAINED_DEFAULT" ;;
  esac
}

get_weights() {
  case "$1" in
    ViT-L-14) echo "$WEIGHTS_VITL14" ;;
    ViT-H-14) echo "$WEIGHTS_VITH14" ;;
    *) echo "" ;;
  esac
}

get_batch_for_model() {
  local key="${1//[^a-zA-Z0-9]/}"
  local var="BATCH_${key^^}"
  local val="${!var:-}"
  if [ -n "$val" ]; then
    echo "$val"
  elif [ -n "$BATCH_DEFAULT" ]; then
    echo "$BATCH_DEFAULT"
  else
    echo ""
  fi
}

prepare_cfg() {
  local base_cfg="$1" model="$2" pretrained="$3" weights="$4" epochs="$5" batch="$6" workers="$7" amp="$8" tag="$9" seed="${10}"
  "$PY" - "$base_cfg" "$model" "$pretrained" "$weights" "$epochs" "$batch" "$workers" "$amp" "$tag" "$seed" <<'PY'
import sys, yaml, tempfile, pathlib
base, model, pretrained, weights, epochs, batch, workers, amp, tag, seed = sys.argv[1:]
cfg = yaml.safe_load(open(base))
backbone = cfg.get("backbone") or {}
backbone["name"] = "openclip"
if model:
    backbone["model"] = model
    backbone["variant"] = model
if pretrained:
    backbone["pretrained"] = pretrained
if weights:
    backbone["weights"] = weights
cfg["backbone"] = backbone
cfg["dino_size"] = backbone.get("model", model)
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

run_config() {
  local model="$1"
  local model_idx="$2"
  local rep_idx="$3"
  local model_tag="${model// /}"
  local seed=$(( SEED_BASE + rep_idx - 1 ))
  local pretrained
  pretrained="$(get_pretrained "$model")"
  local weights
  weights="$(get_weights "$model")"
  local batch_val
  batch_val="$(get_batch_for_model "$model")"
  local tag="${RUN_TAG}_rep${rep_idx}_m$((model_idx + 1))_${model_tag}_seed${seed}"
  mapfile -t prep < <(prepare_cfg "$BASE_CFG" "$model" "$pretrained" "$weights" "$EPOCHS" "$batch_val" "$NUM_WORKERS" "$AMP" "$tag" "$seed")
  runtime_cfg="${prep[0]}"
  run_dir="${prep[1]}"
  echo "[grid model=${model} rep=${rep_idx}] seed=$seed cfg=$runtime_cfg"
  "$PY" scripts/train_em_seg.py --cfg "$runtime_cfg"
  "$PY" scripts/eval_em_seg.py --cfg "$runtime_cfg" --out_csv "$run_dir/${tag}_metrics.csv"
  archive_slurm_log "$run_dir" "$tag"
  rm -f "$runtime_cfg"
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  task_id=${SLURM_ARRAY_TASK_ID}
  if (( task_id < 0 || task_id >= TOTAL_TASKS )); then
    echo "[error] SLURM_ARRAY_TASK_ID=$task_id outside 0..$((TOTAL_TASKS - 1))" >&2
    exit 1
  fi
  model_index=$(( task_id % NUM_MODELS ))
  rep_index=$(( task_id / NUM_MODELS + 1 ))
  run_config "${MODEL_LIST[$model_index]}" "$model_index" "$rep_index"
else
  echo "[info] SLURM_ARRAY_TASK_ID not set; running all $TOTAL_TASKS configs sequentially"
  for ((rep=1; rep<=REPEATS; rep++)); do
    for ((idx=0; idx<NUM_MODELS; idx++)); do
      run_config "${MODEL_LIST[$idx]}" "$idx" "$rep"
    done
  done
fi
