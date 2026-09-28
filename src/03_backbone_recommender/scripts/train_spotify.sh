#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WP_ROOT:${PYTHONPATH:-}"

REPO_ROOT="$(cd "$WP_ROOT/../.." && pwd)"
TRAIN_MODE="${GENPLAYLIST_TRAIN_MODE:-warmstart}"
DATA_CONFIG="${GENPLAYLIST_DATA_CONFIG:-spotify}"
DATA_ROOT="${GENPLAYLIST_DATA_ROOT:-$REPO_ROOT/data/dataset}"
ARTIFACT_ROOT="${GENPLAYLIST_ARTIFACT_ROOT:-$REPO_ROOT/data/dataset}"
CUE_ROOT="${GENPLAYLIST_CUE_ROOT:-$REPO_ROOT/src/02_creative_cues/outputs/production/latest}"
WARMSTART_CKPT="${GENPLAYLIST_WARMSTART_CKPT:-$REPO_ROOT/checkpoints/pretrained/ddbc/spotify30.ckpt}"
DEFAULT_PREPARED_ROOT="${DATA_ROOT%/dataset}/processed/genplaylist-v4-8cue-20item-joint-15to5"
PREPARED_DATA_ROOT="${GENPLAYLIST_PREPARED_DATA_ROOT:-$DEFAULT_PREPARED_ROOT}"
MODEL_SIZE="${GENPLAYLIST_MODEL_SIZE:-small}"
GLOBAL_BATCH_SIZE="${GENPLAYLIST_GLOBAL_BATCH_SIZE:-512}"
TRAIN_BATCH_SIZE="${GENPLAYLIST_TRAIN_BATCH_SIZE:-256}"
MAX_STEPS="${GENPLAYLIST_MAX_STEPS:-20000}"
LIMIT_TRAIN_BATCHES="${GENPLAYLIST_LIMIT_TRAIN_BATCHES:-1.0}"
LOSS_CURRICULUM="${GENPLAYLIST_LAYER_LOSS_CURRICULUM:-true}"
ACTIVE_CUES="${GENPLAYLIST_ACTIVE_CUES:-8}"
STRUCTURE_CONDITIONING="${GENPLAYLIST_STRUCTURE_CONDITIONING:-false}"
CUE_WEIGHT="${GENPLAYLIST_CUE_WEIGHT:-1.0}"
CUE_WARMUP_INITIAL_WEIGHT="${GENPLAYLIST_CUE_WARMUP_INITIAL_WEIGHT:-0.1}"
CUE_WARMUP_START_STEP="${GENPLAYLIST_CUE_WARMUP_START_STEP:-1000}"
CUE_WARMUP_END_STEP="${GENPLAYLIST_CUE_WARMUP_END_STEP:-5000}"
HISTORY_CONDITION="${GENPLAYLIST_HISTORY_CONDITION:-full}"
SEED="${GENPLAYLIST_SEED:-1}"

case "$HISTORY_CONDITION" in
  full|latent_only|cue_only|shuffled_cue) ;;
  *)
    echo "GENPLAYLIST_HISTORY_CONDITION must be full, latent_only, cue_only, or shuffled_cue" >&2
    exit 2
    ;;
esac
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "GENPLAYLIST_SEED must be a non-negative integer" >&2
  exit 2
fi

case "$ACTIVE_CUES" in
  0|4|8|16) ;;
  *)
    echo "GENPLAYLIST_ACTIVE_CUES must be one of 0, 4, 8, or 16" >&2
    exit 2
    ;;
esac
MODEL_LENGTH=$((2 + 20 * (5 + ACTIVE_CUES)))

case "$STRUCTURE_CONDITIONING" in
  true) STRUCTURE_VARIANT="with-structure" ;;
  false) STRUCTURE_VARIANT="no-structure" ;;
  *)
    echo "GENPLAYLIST_STRUCTURE_CONDITIONING must be true or false" >&2
    exit 2
    ;;
esac

case "$LOSS_CURRICULUM" in
  true)
    LOSS_VARIANT="rvq-cue-warmup-cw${CUE_WARMUP_INITIAL_WEIGHT}to${CUE_WEIGHT}-s${CUE_WARMUP_START_STEP}to${CUE_WARMUP_END_STEP}"
    ;;
  false)
    LOSS_VARIANT="uniform"
    ;;
  *)
    echo "GENPLAYLIST_LAYER_LOSS_CURRICULUM must be true or false" >&2
    exit 2
    ;;
esac

if [[ ! -f "$PREPARED_DATA_ROOT/prepared_manifest.json" ]]; then
  echo "Missing prepared WP-C data: $PREPARED_DATA_ROOT/prepared_manifest.json" >&2
  echo "Run scripts/prepare_wp_c_data.py before training." >&2
  exit 1
fi

case "$TRAIN_MODE" in
  warmstart)
    if [[ ! -f "$WARMSTART_CKPT" ]]; then
      echo "Missing DDBC warm-start checkpoint: $WARMSTART_CKPT" >&2
      exit 1
    fi
    CHECKPOINT_ARGS=(
      checkpointing.resume_from_ckpt=false
      checkpointing.warmstart_path="$WARMSTART_CKPT"
    )
    ;;
  resume)
    CHECKPOINT_ARGS=(
      checkpointing.resume_from_ckpt=true
      checkpointing.warmstart_path=null
    )
    ;;
  scratch)
    CHECKPOINT_ARGS=(
      checkpointing.resume_from_ckpt=false
      checkpointing.warmstart_path=null
    )
    ;;
  *)
    echo "GENPLAYLIST_TRAIN_MODE must be warmstart, resume, or scratch" >&2
    exit 2
    ;;
esac

# Generate unique run name with timestamp to avoid conflicts
RUN_NAME="genplaylist-v4-${DATA_CONFIG}-joint15to5-${ACTIVE_CUES}cue-history-${HISTORY_CONDITION}-${STRUCTURE_VARIANT}-${LOSS_VARIANT}-seed${SEED}-$(date +%Y%m%d-%H%M%S)"

cd "$WP_ROOT"
python main.py \
  loader.global_batch_size="$GLOBAL_BATCH_SIZE" \
  loader.batch_size="$TRAIN_BATCH_SIZE" \
  loader.eval_batch_size=128 \
  model="$MODEL_SIZE" \
  data="$DATA_CONFIG" \
  data_root="$DATA_ROOT" \
  catalog_embeddings_path="$ARTIFACT_ROOT/catalog_item_embeddings.npy" \
  item_id_to_row_path="$ARTIFACT_ROOT/item_id_to_row.json" \
  semantic_tokens_path="$ARTIFACT_ROOT/semantic_tokens.json" \
  codebook_weights_path="$ARTIFACT_ROOT/rvq_codebook_weights.npy" \
  item2cues_path="$CUE_ROOT/item2cues.json" \
  cue_vocab_path="$CUE_ROOT/cue_vocab.json" \
  cue_manifest_path="$CUE_ROOT/cue_manifest.json" \
  prepared_dataset_path="$PREPARED_DATA_ROOT" \
  active_cue_tokens="$ACTIVE_CUES" \
  history_condition="$HISTORY_CONDITION" \
  seed="$SEED" \
  model.length="$MODEL_LENGTH" \
  sampling.structure_conditioning="$STRUCTURE_CONDITIONING" \
  +run_name=${RUN_NAME} \
  trainer.max_steps="$MAX_STEPS" \
  trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES" \
  training.layer_loss_weights.enabled="$LOSS_CURRICULUM" \
  training.layer_loss_weights.cue_weight="$CUE_WEIGHT" \
  training.layer_loss_weights.warmup.initial_cue_weight="$CUE_WARMUP_INITIAL_WEIGHT" \
  training.layer_loss_weights.warmup.start_step="$CUE_WARMUP_START_STEP" \
  training.layer_loss_weights.warmup.end_step="$CUE_WARMUP_END_STEP" \
  parameterization=subs \
  eval.compute_generative_perplexity=False \
  sampling.steps=25 \
  "${CHECKPOINT_ARGS[@]}"
