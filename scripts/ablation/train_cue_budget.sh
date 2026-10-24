#!/usr/bin/env bash
# Cue-budget ablation: train one dataset's cue-budget variants for given seeds.
#
# Usage:  bash scripts/ablation/train_cue_budget.sh <mpd|m4a> --gpu N --seeds "1 2 3"
#                                                   [--budgets "0 4 random8 16"] [--balanced]
#                                                   [--train-batch-size N]
#   e.g.  bash scripts/ablation/train_cue_budget.sh mpd --gpu 4 --seeds 1
#         bash scripts/ablation/train_cue_budget.sh m4a --gpu 7 --seeds "2 3" --budgets "16"
#         bash scripts/ablation/train_cue_budget.sh mpd --gpu 4 --seeds 1 --balanced
#
# Runs go to outputs/ablation-cue-budget/<mpd|m4a>-cues<N>-seed<S> (random-8:
# -cues8random-) and are logged in outputs/runs.csv. Budget 8 (ranked top 8)
# is the history-conditioning Full model and is never retrained here.
# --balanced (ablation 2b) keeps the cue share of the loss at its 8-cue value:
# it trains 4 and 16 cues with cue weight 8/budget into
# outputs/ablation-cue-budget-balanced/, plus budget 8noloss (8 cues as input
# only) when listed; default budgets "4 16". 0, 8, and random8 already have
# the 8-cue share and are reused, not retrained.
# --train-batch-size N sets the per-GPU micro-batch (train_spotify.sh default
# 256). The global batch stays 512 through gradient accumulation, so a smaller
# N only lowers GPU memory and speed. N must divide 512.
# Finished runs are refused by train_spotify.sh and skipped. Run
# prepare_cue_budget.sh first. See docs/ABLATIONS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
GPU=""; SEEDS=""; BUDGETS=""; TRAIN_BATCH_SIZE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --budgets) BUDGETS="${2:-}"; shift 2 ;;
    --balanced) CUE_LOSS=balanced; shift ;;
    --train-batch-size) TRAIN_BATCH_SIZE="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$TRAIN_BATCH_SIZE" ]] && ! { [[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
     && (( 512 % TRAIN_BATCH_SIZE == 0 )); }; then
  echo "--train-batch-size must be a positive divisor of 512, got: $TRAIN_BATCH_SIZE" >&2
  exit 2
fi
if [[ -z "$BUDGETS" ]]; then
  if [[ "$CUE_LOSS" == balanced ]]; then BUDGETS="4 16"; else BUDGETS="0 4 random8 16"; fi
fi
[[ -n "$GPU" && -n "$SEEDS" ]] || { echo "--gpu and --seeds are required" >&2; exit 2; }
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $seed" >&2; exit 2; }
done
ablation_env_is_clean || exit 2

# Check every requested variant's data before training anything.
missing=false
for budget in $BUDGETS; do
  cue_budget_variant "$budget" || exit 2
  [[ -n "$TRAINED_BY" ]] && continue
  for f in "$PREPARED/prepared_manifest.json" "$VARIANT_CUE_ROOT/item2cues.json"; do
    [[ -f "$f" ]] || { echo "Missing ($budget): $f" >&2; missing=true; }
  done
done
[[ "$missing" == false ]] || { echo "Run prepare_cue_budget.sh $DATASET first." >&2; exit 1; }

failed=()
for seed in $SEEDS; do
  for budget in $BUDGETS; do
    cue_budget_variant "$budget"
    if [[ -n "$TRAINED_BY" ]]; then
      echo "=== budget $budget is not trained here; it is the model from $TRAINED_BY"
      continue
    fi
    NAME=$RUN_PREFIX-seed$seed
    echo "=== $(date '+%F %T')  $NAME  ($ACTIVE_CUES cues, $TOKENS tokens, cue weight ${CUE_WEIGHT:-default}, GPU $GPU)"
    # Balanced budgets set their own cue loss weights; otherwise
    # train_spotify.sh keeps its defaults.
    extra_env=()
    if [[ -n "$CUE_WEIGHT" ]]; then
      extra_env=(GENPLAYLIST_CUE_WEIGHT=$CUE_WEIGHT
                  GENPLAYLIST_CUE_WARMUP_INITIAL_WEIGHT=$CUE_WARMUP_INITIAL)
    fi
    if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
      extra_env+=(GENPLAYLIST_TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE)
    fi
    # runs.csv only gets a row when training finishes, so a checkpoint without
    # a row is a crashed run: resume it in place.
    MODE=warmstart
    if [[ -f "$OUT/$NAME/checkpoints/last.ckpt" ]]; then
      if grep -q ",$NAME," "$OUT/runs.csv" 2>/dev/null; then
        echo "  already trained; skipping"
        continue
      fi
      echo "  unfinished run found; resuming"
      MODE=resume
    fi
    env ${extra_env[@]+"${extra_env[@]}"} \
    CUDA_VISIBLE_DEVICES=$GPU \
    GENPLAYLIST_TRAIN_MODE=$MODE \
    GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
    GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
    GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
    GENPLAYLIST_CUE_ROOT=$VARIANT_CUE_ROOT \
    GENPLAYLIST_ACTIVE_CUES=$ACTIVE_CUES \
    GENPLAYLIST_HISTORY_CONDITION=full \
    GENPLAYLIST_OUTPUT_NAME=$NAME \
    GENPLAYLIST_SEED=$seed \
    GENPLAYLIST_PREPARED_DATA_ROOT=$PREPARED \
      bash "$TRAIN_SCRIPT" \
      || { echo "!!! $NAME failed; continuing with the next run"; failed+=("$NAME"); }
  done
done

if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "=== all done $(date '+%F %T')"
