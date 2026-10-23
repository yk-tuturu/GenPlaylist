# Shared settings for the scripts in scripts/ablation/ (sourced, not run).
#
# Expects REPO. Provides:
#   ablation_dataset <mpd|m4a>     dataset paths (DATA_CONFIG, DATA_ROOT, ...)
#   cue_budget_variant <budget>    one cue-budget variant under CUE_LOSS (ACTIVE_CUES, ...)
#   training_schedule_variant <s>  one loss schedule (warmup, fixed, uniform)
#   ablation_env_is_clean          refuse leftover GENPLAYLIST_* variables

OUT=$REPO/src/03_backbone_recommender/outputs
TRAIN_SCRIPT=$REPO/src/03_backbone_recommender/scripts/train_spotify.sh
EVAL_SCRIPT=$REPO/src/03_backbone_recommender/scripts/eval_spotify.sh
RANDOM_CUE_SEED=42
CUE_BUDGETS_ALL="0 4 8 random8 16"

ablation_dataset() {
  DATASET=$1
  case "$DATASET" in
    mpd)
      DATA_CONFIG=spotify
      DATA_ROOT=$REPO/data/dataset
      ARTIFACT_ROOT=$REPO/data/dataset
      CUE_ROOT=$REPO/src/02_creative_cues/outputs/production/latest
      PREPARED_PREFIX=$REPO/data/processed/ablation-mpd
      MERT_DIR=$REPO/data/processed/mert-v1-95m-catalog-v1
      ;;
    m4a)
      DATA_CONFIG=music4all
      DATA_ROOT=$REPO/data/dataset-music4all-onion-v3-u500
      ARTIFACT_ROOT=$REPO/data/artifacts-music4all-onion-v3-u500
      CUE_ROOT=$REPO/data/processed/cues-music4all-onion-v3-u500/latest
      PREPARED_PREFIX=$REPO/data/processed/ablation-m4a-v3-u500
      MERT_DIR=$REPO/data/processed/mert-v1-95m-music4all-u500-audio-covered-v1
      ;;
    *)
      echo "Dataset must be mpd or m4a, got: ${DATASET:-<none>}" >&2
      return 2
      ;;
  esac
  RANDOM_CUE_ROOT=$REPO/data/processed/cues-$DATASET-random8-seed$RANDOM_CUE_SEED
}

# Cue-loss setting of the cue-budget ablation (set by --balanced):
#   standard  every cue token weighs 1.0, so the cue share of the loss grows
#             with the budget (44% at 4 cues, 62% at 8, 76% at 16)
#   balanced  the cues of a song weigh 8 in total (1.0 x 8 / budget each), so
#             every budget keeps the 8-cue share. 0, 8, and random8 already
#             have it and reuse their standard models; 8noloss (8 cues as input
#             only, cue weight 0) is an extra control.
CUE_LOSS=standard
CUE_BUDGETS_BALANCED="0 4 8 random8 16 8noloss"

# Sets, for one cue budget of the current dataset and CUE_LOSS:
#   ACTIVE_CUES  VARIANT_CUE_ROOT  PREPARED  TOKENS (per sequence)
#   RUN_PREFIX   model folder under $OUT without the seed suffix
#   CUE_WEIGHT  CUE_WARMUP_INITIAL  cue loss weights ("" = train_spotify.sh default)
#   TRAINED_BY   empty when this runner trains the model, else where it comes from
# Budget 8 (ranked top 8) is the history-conditioning Full model and its data.
cue_budget_variant() {
  local budget=$1
  VARIANT_CUE_ROOT=$CUE_ROOT
  CUE_WEIGHT=""
  CUE_WARMUP_INITIAL=""
  TRAINED_BY=""
  case "$CUE_LOSS:$budget" in
    standard:0|standard:4|standard:16|balanced:0)
      ACTIVE_CUES=$budget
      PREPARED=$PREPARED_PREFIX-cues$budget
      RUN_PREFIX=ablation-cue-budget/$DATASET-cues$budget
      ;;
    standard:8|balanced:8)
      ACTIVE_CUES=8
      PREPARED=$PREPARED_PREFIX-8cue-full
      RUN_PREFIX=ablation-history-cond/$DATASET-full
      TRAINED_BY="train_history_cond_$DATASET.sh"
      ;;
    standard:random8|balanced:random8)
      ACTIVE_CUES=8
      VARIANT_CUE_ROOT=$RANDOM_CUE_ROOT
      PREPARED=$PREPARED_PREFIX-cues8random
      RUN_PREFIX=ablation-cue-budget/$DATASET-cues8random
      ;;
    balanced:4)
      ACTIVE_CUES=4
      PREPARED=$PREPARED_PREFIX-cues4
      RUN_PREFIX=ablation-cue-budget-balanced/$DATASET-cues4
      CUE_WEIGHT=2.0
      CUE_WARMUP_INITIAL=0.2
      ;;
    balanced:16)
      ACTIVE_CUES=16
      PREPARED=$PREPARED_PREFIX-cues16
      RUN_PREFIX=ablation-cue-budget-balanced/$DATASET-cues16
      CUE_WEIGHT=0.5
      CUE_WARMUP_INITIAL=0.05
      ;;
    balanced:8noloss)
      ACTIVE_CUES=8
      PREPARED=$PREPARED_PREFIX-8cue-full
      RUN_PREFIX=ablation-cue-budget-balanced/$DATASET-cues8noloss
      CUE_WEIGHT=0.0
      CUE_WARMUP_INITIAL=0.0
      ;;
    *)
      local valid=$CUE_BUDGETS_ALL
      [[ "$CUE_LOSS" == balanced ]] && valid=$CUE_BUDGETS_BALANCED
      echo "Unknown $CUE_LOSS cue budget: $budget (use: $valid)" >&2
      return 2
      ;;
  esac
  if [[ "$CUE_LOSS" == balanced && -z "$CUE_WEIGHT" && -z "$TRAINED_BY" ]]; then
    TRAINED_BY="train_cue_budget.sh $DATASET (standard; same cue-loss share)"
  fi
  TOKENS=$((2 + 20 * (5 + ACTIVE_CUES)))
}

# Model folder (relative to $OUT) for a budget and seed. MPD history-condition
# seed-1 models predate the -seed suffix and live in the unsuffixed folder.
cue_budget_run_name() {
  local seed=$1
  local name=$RUN_PREFIX-seed$seed
  if [[ "$RUN_PREFIX" == ablation-history-cond/mpd-* && "$seed" == 1
        && ! -d "$OUT/$name" && -d "$OUT/$RUN_PREFIX" ]]; then
    name=$RUN_PREFIX
  fi
  echo "$name"
}

TRAINING_SCHEDULES_ALL="warmup fixed uniform"

# Sets, for one loss schedule of the current dataset (all use the Full 8-cue
# data from the history-conditioning ablation; all three arms are trained
# here, with gradient-norm logging):
#   SCHEDULE  ACTIVE_CUES  VARIANT_CUE_ROOT  PREPARED  TOKENS  RUN_PREFIX
training_schedule_variant() {
  case "$1" in
    warmup|fixed|uniform) SCHEDULE=$1 ;;
    *)
      echo "Unknown loss schedule: $1 (use: $TRAINING_SCHEDULES_ALL)" >&2
      return 2
      ;;
  esac
  ACTIVE_CUES=8
  VARIANT_CUE_ROOT=$CUE_ROOT
  PREPARED=$PREPARED_PREFIX-8cue-full
  TOKENS=262
  RUN_PREFIX=ablation-training-schedule/$DATASET-$SCHEDULE
}

# Evaluation must only read finished training runs. last.ckpt is not enough:
# training rewrites it every 500 steps. A run is finished when training wrote
# its runs.csv row (named runs) or its final step checkpoint (runs from before
# runs.csv, e.g. the moved MPD seed-1 runs).
FINAL_STEP=20000
run_is_finished() {
  local name=$1
  grep -qF ",$name," "$OUT/runs.csv" 2>/dev/null && return 0
  [[ -f "$OUT/$name/checkpoints/step-$FINAL_STEP.ckpt" ]]
}

# step-<K>.ckpt is written once; it is complete when the run has finished or
# training has already saved a later checkpoint.
checkpoint_is_settled() {
  local name=$1 step=$2 file saved
  run_is_finished "$name" && return 0
  for file in "$OUT/$name"/checkpoints/step-*.ckpt; do
    saved=${file##*/step-}
    saved=${saved%.ckpt}
    [[ "$saved" =~ ^[0-9]+$ ]] && (( saved > step )) && return 0
  done
  return 1
}

# A result older than its checkpoint was evaluated before training finished
# (or before the checkpoint was replaced) and must not be reused.
result_is_stale() {
  [[ -f "$1" && -f "$2" && "$1" -ot "$2" ]]
}

ablation_env_is_clean() {
  if env | grep -q '^GENPLAYLIST_'; then
    echo "Unset these first:" >&2
    env | grep '^GENPLAYLIST_' >&2
    return 1
  fi
}
