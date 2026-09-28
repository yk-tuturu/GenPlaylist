#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WP_ROOT/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WP_ROOT:${PYTHONPATH:-}"

DATA_ROOT="${GENPLAYLIST_DATA_ROOT:-$REPO_ROOT/data/dataset}"
DATA_CONFIG="${GENPLAYLIST_DATA_CONFIG:-spotify}"
ARTIFACT_ROOT="${GENPLAYLIST_ARTIFACT_ROOT:-$REPO_ROOT/data/dataset}"
CUE_ROOT="${GENPLAYLIST_CUE_ROOT:-$REPO_ROOT/src/02_creative_cues/outputs/production/latest}"
DEFAULT_PREPARED_ROOT="${DATA_ROOT%/dataset}/processed/genplaylist-v4-8cue-20item-joint-15to5"
PREPARED_DATA_ROOT="${GENPLAYLIST_PREPARED_DATA_ROOT:-$DEFAULT_PREPARED_ROOT}"
EVAL_CKPT="${GENPLAYLIST_EVAL_CKPT:-}"
MODEL_SIZE="${GENPLAYLIST_MODEL_SIZE:-small}"
EVAL_BATCH_SIZE="${GENPLAYLIST_EVAL_BATCH_SIZE:-32}"
EVAL_SEED="${GENPLAYLIST_EVAL_SEED:-1}"
ALLOW_PROTOCOL_OVERRIDE="${GENPLAYLIST_EVAL_ALLOW_PROTOCOL_OVERRIDE:-false}"
ACTIVE_CUES="${GENPLAYLIST_ACTIVE_CUES:-8}"
STRUCTURE_CONDITIONING="${GENPLAYLIST_STRUCTURE_CONDITIONING:-false}"
HISTORY_CONDITION="${GENPLAYLIST_HISTORY_CONDITION:-full}"
# true only for diagnostics such as a Full checkpoint on shuffled_cue contexts.
ALLOW_HISTORY_MISMATCH="${GENPLAYLIST_EVAL_ALLOW_HISTORY_MISMATCH:-false}"
case "$HISTORY_CONDITION" in
  full|latent_only|cue_only|shuffled_cue) ;;
  *)
    echo "GENPLAYLIST_HISTORY_CONDITION must be full, latent_only, cue_only, or shuffled_cue" >&2
    exit 2
    ;;
esac
case "$ALLOW_HISTORY_MISMATCH" in
  true|false) ;;
  *)
    echo "GENPLAYLIST_EVAL_ALLOW_HISTORY_MISMATCH must be true or false" >&2
    exit 2
    ;;
esac
case "$ACTIVE_CUES" in
  0|4|8|16) ;;
  *)
    echo "GENPLAYLIST_ACTIVE_CUES must be one of 0, 4, 8, or 16" >&2
    exit 2
    ;;
esac
MODEL_LENGTH=$((2 + 20 * (5 + ACTIVE_CUES)))
case "$STRUCTURE_CONDITIONING" in
  true|false) ;;
  *)
    echo "GENPLAYLIST_STRUCTURE_CONDITIONING must be true or false" >&2
    exit 2
    ;;
esac

# The official protocol uses 256 reverse-diffusion steps. Shorter settings are
# smoke tests and must write to a separately named result file.
SAMPLING_STEPS="${GENPLAYLIST_EVAL_SAMPLING_STEPS:-256}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="${GENPLAYLIST_EVAL_RESULTS_ROOT:-$WP_ROOT/outputs/evaluation}"

if [[ -z "$EVAL_CKPT" ]]; then
  echo "GENPLAYLIST_EVAL_CKPT must point to the checkpoint to evaluate." >&2
  exit 2
fi
# Hydra changes into a fresh run directory, so relative paths would break.
EVAL_CKPT="$(realpath "$EVAL_CKPT")"

CKPT_LABEL="$(basename "$EVAL_CKPT" .ckpt)"
RESULTS_PATH="${GENPLAYLIST_EVAL_RESULTS_PATH:-$RESULTS_ROOT/wp-c-${CKPT_LABEL}-${ACTIVE_CUES}cue-history-${HISTORY_CONDITION}-structure${STRUCTURE_CONDITIONING}-steps${SAMPLING_STEPS}-seed${EVAL_SEED}-${STAMP}.json}"

for required in \
  "$EVAL_CKPT" \
  "$PREPARED_DATA_ROOT/prepared_manifest.json" \
  "$DATA_ROOT/catalog_metadata.json" \
  "$ARTIFACT_ROOT/catalog_item_embeddings.npy" \
  "$ARTIFACT_ROOT/item_id_to_row.json" \
  "$ARTIFACT_ROOT/semantic_tokens.json" \
  "$ARTIFACT_ROOT/rvq_codebook_weights.npy" \
  "$CUE_ROOT/item2cues.json" \
  "$CUE_ROOT/cue_vocab.json" \
  "$CUE_ROOT/cue_manifest.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing evaluation input: $required" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$RESULTS_PATH")"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

cd "$WP_ROOT"
python main.py \
  mode=rec_eval \
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
  model.length="$MODEL_LENGTH" \
  sampling.structure_conditioning="$STRUCTURE_CONDITIONING" \
  eval.checkpoint_path="$EVAL_CKPT" \
  eval.allow_history_condition_mismatch="$ALLOW_HISTORY_MISMATCH" \
  eval.results_path="$RESULTS_PATH" \
  eval.git_commit="$GIT_COMMIT" \
  eval.disable_ema=false \
  eval.allow_protocol_override="$ALLOW_PROTOCOL_OVERRIDE" \
  eval_batch_size="$EVAL_BATCH_SIZE" \
  seed="$EVAL_SEED" \
  sampling.steps="$SAMPLING_STEPS" \
  parameterization=subs \
  eval.compute_generative_perplexity=false \
  +run_name="genplaylist-v4-${DATA_CONFIG}-${ACTIVE_CUES}cue-structure${STRUCTURE_CONDITIONING}-joint15to5-official-eval-${STAMP}"

echo "Official WP-C result: $RESULTS_PATH"
