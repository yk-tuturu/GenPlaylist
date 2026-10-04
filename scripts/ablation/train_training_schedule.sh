#!/usr/bin/env bash
# Training-schedule ablation: train one dataset's loss-schedule variants
# (warmup, fixed, uniform) for given seeds, with gradient-norm logging.
#
# Usage:  bash scripts/ablation/train_training_schedule.sh <mpd|m4a> --gpu N --seeds "1 2 3"
#                                                          [--schedules "warmup fixed uniform"]
#   e.g.  bash scripts/ablation/train_training_schedule.sh mpd --gpu 4 --seeds 1
#         bash scripts/ablation/train_training_schedule.sh m4a --gpu 7 --seeds "1 2 3" --schedules fixed
#
# Runs go to outputs/ablation-training-schedule/<mpd|m4a>-<schedule>-seed<S>
# and are logged in outputs/runs.csv. They use the Full 8-cue prepared data of
# the history-conditioning ablation (<prefix>-8cue-full), so nothing new is
# prepared. Finished runs are skipped; unfinished ones are resumed.
# See docs/ABLATIONS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
GPU=""; SEEDS=""; SCHEDULES="$TRAINING_SCHEDULES_ALL"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --schedules) SCHEDULES="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$GPU" && -n "$SEEDS" ]] || { echo "--gpu and --seeds are required" >&2; exit 2; }
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $seed" >&2; exit 2; }
done
for schedule in $SCHEDULES; do training_schedule_variant "$schedule" || exit 2; done
ablation_env_is_clean || exit 2
training_schedule_variant warmup
for f in "$PREPARED/prepared_manifest.json" "$CUE_ROOT/item2cues.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f (prepare the history-conditioning Full data first)" >&2; exit 1; }
done

failed=()
for seed in $SEEDS; do
  for schedule in $SCHEDULES; do
    training_schedule_variant "$schedule"
    NAME=$RUN_PREFIX-seed$seed
    echo "=== $(date '+%F %T')  $NAME  (GPU $GPU)"
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
    CUDA_VISIBLE_DEVICES=$GPU \
    GENPLAYLIST_TRAIN_MODE=$MODE \
    GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
    GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
    GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
    GENPLAYLIST_CUE_ROOT=$VARIANT_CUE_ROOT \
    GENPLAYLIST_ACTIVE_CUES=$ACTIVE_CUES \
    GENPLAYLIST_HISTORY_CONDITION=full \
    GENPLAYLIST_LOSS_SCHEDULE=$SCHEDULE \
    GENPLAYLIST_LOG_GRAD_NORM=true \
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
