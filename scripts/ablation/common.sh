# Shared settings for the scripts in scripts/ablation/ (sourced, not run).
#
# Expects REPO. Provides:
#   ablation_dataset <mpd|m4a>     dataset paths (DATA_CONFIG, DATA_ROOT, ...)
#   cue_budget_variant <budget>    one cue-budget variant (ACTIVE_CUES, ...)
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

# Sets, for one cue budget of the current dataset:
#   ACTIVE_CUES  VARIANT_CUE_ROOT  PREPARED  TOKENS (per sequence)
#   RUN_PREFIX   model folder under $OUT without the seed suffix
# Budget 8 (ranked top 8) is the history-conditioning Full model and its data.
cue_budget_variant() {
  local budget=$1
  VARIANT_CUE_ROOT=$CUE_ROOT
  case "$budget" in
    0|4|16)
      ACTIVE_CUES=$budget
      PREPARED=$PREPARED_PREFIX-cues$budget
      RUN_PREFIX=ablation-cue-budget/$DATASET-cues$budget
      ;;
    8)
      ACTIVE_CUES=8
      PREPARED=$PREPARED_PREFIX-8cue-full
      RUN_PREFIX=ablation-history-cond/$DATASET-full
      ;;
    random8)
      ACTIVE_CUES=8
      VARIANT_CUE_ROOT=$RANDOM_CUE_ROOT
      PREPARED=$PREPARED_PREFIX-cues8random
      RUN_PREFIX=ablation-cue-budget/$DATASET-cues8random
      ;;
    *)
      echo "Unknown cue budget: $budget (use: $CUE_BUDGETS_ALL)" >&2
      return 2
      ;;
  esac
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

ablation_env_is_clean() {
  if env | grep -q '^GENPLAYLIST_'; then
    echo "Unset these first:" >&2
    env | grep '^GENPLAYLIST_' >&2
    return 1
  fi
}
