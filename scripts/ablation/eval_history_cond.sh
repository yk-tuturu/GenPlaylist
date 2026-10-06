#!/usr/bin/env bash
# History-conditioning ablation: evaluate the trained variants (WP-C proxy eval
# + MERT metrics).
#
# Usage:
#   bash scripts/ablation/eval_history_cond.sh <mpd|m4a> --gpu N [--seeds "1 2 3"] [--conds "full cue_only"]
#                         [--no-mert] [--dry-run]
#
# Examples:
#   bash scripts/ablation/eval_history_cond.sh mpd --gpu 4                          # seed 1, all 4 conditions
#   bash scripts/ablation/eval_history_cond.sh mpd --gpu 4 --seeds "2 3"
#   bash scripts/ablation/eval_history_cond.sh m4a --gpu 7 --seeds 1 --conds "full shuffled_cue"
#   bash scripts/ablation/eval_history_cond.sh m4a --gpu 7 --seeds "1 2 3" --dry-run   # only print the plan
#
# Models are read from outputs/ablation-history-cond/<mpd|m4a>-<cond>-seed<N>.
# For MPD seed 1, the older unsuffixed folder mpd-<cond> is used if present.
# Finished evaluations and MERT outputs are skipped, so re-running is safe.
# See docs/ABLATIONS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Only for run_is_finished / result_is_stale; the settings below are this
# script's own.
source "$REPO/scripts/ablation/common.sh"
OUT=$REPO/src/03_backbone_recommender/outputs
EVAL_SCRIPT=$REPO/src/03_backbone_recommender/scripts/eval_spotify.sh
RESULT_FILE=last-steps256-evalseed1.json
ALL_CONDS="full latent_only cue_only shuffled_cue"

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

[[ "${1:-}" == -h || "${1:-}" == --help ]] && usage
DATASET="${1:-}"; shift || true
GPU=""; SEEDS="1"; CONDS="$ALL_CONDS"; RUN_MERT=true; DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --conds) CONDS="${2:-}"; shift 2 ;;
    --no-mert) RUN_MERT=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

case "$DATASET" in
  mpd)
    DATA_CONFIG=spotify
    DATA_ROOT=$REPO/data/dataset
    ARTIFACT_ROOT=$REPO/data/dataset
    CUE_ROOT=$REPO/src/02_creative_cues/outputs/production/latest
    PREPARED_PREFIX=$REPO/data/processed/ablation-mpd-8cue
    MERT_DIR=$REPO/data/processed/mert-v1-95m-catalog-v1
    ;;
  m4a)
    DATA_CONFIG=music4all
    DATA_ROOT=$REPO/data/dataset-music4all-onion-v3-u500
    ARTIFACT_ROOT=$REPO/data/artifacts-music4all-onion-v3-u500
    CUE_ROOT=$REPO/data/processed/cues-music4all-onion-v3-u500/latest
    PREPARED_PREFIX=$REPO/data/processed/ablation-m4a-v3-u500-8cue
    MERT_DIR=$REPO/data/processed/mert-v1-95m-music4all-u500-audio-covered-v1
    ;;
  *) echo "First argument must be mpd or m4a" >&2; usage ;;
esac

[[ -n "$GPU" ]] || { echo "--gpu is required" >&2; usage; }
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $seed" >&2; exit 2; }
done
for cond in $CONDS; do
  [[ " $ALL_CONDS " == *" $cond "* ]] || { echo "Unknown condition: $cond (use: $ALL_CONDS)" >&2; exit 2; }
done
if env | grep -q '^GENPLAYLIST_'; then
  echo "Unset these first:" >&2; env | grep '^GENPLAYLIST_' >&2; exit 2
fi
for f in "$EVAL_SCRIPT" "$DATA_ROOT/catalog_metadata.json" "$ARTIFACT_ROOT/item_id_to_row.json" "$CUE_ROOT/item2cues.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
if [[ "$RUN_MERT" == true && ! -f "$MERT_DIR/mert_manifest.json" ]]; then
  echo "Missing MERT folder: $MERT_DIR (or pass --no-mert)" >&2; exit 1
fi

# Folder name of a trained model, relative to $OUT.
run_name() {
  local cond=$1 seed=$2
  local name="ablation-history-cond/$DATASET-$cond-seed$seed"
  if [[ "$DATASET" == mpd && "$seed" == 1 && ! -d "$OUT/$name" \
        && -d "$OUT/ablation-history-cond/mpd-$cond" ]]; then
    name="ablation-history-cond/mpd-$cond"
  fi
  echo "$name"
}

ran=() skipped=() failed=() pending=()
for seed in $SEEDS; do
  for cond in $CONDS; do
    name=$(run_name "$cond" "$seed")
    ckpt=$OUT/$name/checkpoints/last.ckpt
    result=$OUT/$name/results/$RESULT_FILE
    mert=${result%.json}-mert.json
    prepared=$PREPARED_PREFIX-$cond
    echo "=== $(date '+%F %T')  $DATASET  $cond  seed $seed  ($name)"

    if ! run_is_finished "$name"; then
      if [[ -f "$result" ]]; then
        echo "  STALE: $result was evaluated before training finished; delete it and re-run"
        failed+=("$name (stale result)")
      else
        echo "  not finished training yet (no runs.csv row or step-$FINAL_STEP.ckpt); skipping"
        pending+=("$name")
      fi
      continue
    fi
    if result_is_stale "$result" "$ckpt"; then
      echo "  STALE: $result is older than $ckpt; delete it and re-run"
      failed+=("$name (stale result)"); continue
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "  no checkpoint at $ckpt; skipping"; failed+=("$name (no checkpoint)"); continue
    fi
    if [[ ! -f "$prepared/prepared_manifest.json" ]]; then
      echo "  no prepared data at $prepared; skipping"; failed+=("$name (no prepared data)"); continue
    fi

    if [[ -f "$result" ]]; then
      echo "  eval: already done"; skipped+=("$name eval")
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  eval: would run on GPU $GPU -> $result"
    else
      if CUDA_VISIBLE_DEVICES=$GPU \
         GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
         GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
         GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
         GENPLAYLIST_CUE_ROOT=$CUE_ROOT \
         GENPLAYLIST_OUTPUT_NAME=$name \
         GENPLAYLIST_HISTORY_CONDITION=$cond \
         GENPLAYLIST_PREPARED_DATA_ROOT=$prepared \
           bash "$EVAL_SCRIPT"; then
        ran+=("$name eval")
      else
        echo "  !!! eval failed"; failed+=("$name eval"); continue
      fi
    fi

    [[ "$RUN_MERT" == true ]] || continue
    if [[ -f "$mert" ]]; then
      echo "  mert: already done"; skipped+=("$name mert")
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  mert: would run -> $mert"
    elif python "$REPO/scripts/evaluate_mert_proxy.py" --prediction-result "$result" \
           --mert-dir "$MERT_DIR" --output "$mert"; then
      ran+=("$name mert")
    else
      echo "  !!! mert failed"; failed+=("$name mert")
    fi
  done
done

echo
echo "=== Summary ($DATASET)"
printf '%-14s %-5s %9s %9s %9s %9s\n' condition seed N1-MERT Recall@5 M2M-MERT Cov@5
for seed in $SEEDS; do
  for cond in $CONDS; do
    mert=$OUT/$(run_name "$cond" "$seed")/results/${RESULT_FILE%.json}-mert.json
    if [[ -f "$mert" ]]; then
      python -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; print('%-14s %-5s %9.4f %9.4f %9.4f %9.4f' % (sys.argv[2], sys.argv[3], m['n1_mert'], m['recall_at_5'], m['m2m_mert'], m['coverage_at_5']))" "$mert" "$cond" "$seed"
    else
      printf '%-14s %-5s %9s\n' "$cond" "$seed" "(no MERT result)"
    fi
  done
done
echo "ran: ${#ran[@]}  already done: ${#skipped[@]}  not finished: ${#pending[@]}  failed: ${#failed[@]}"
if [[ ${#pending[@]} -gt 0 ]]; then
  printf '  not finished training (re-run later): %s\n' "${pending[@]}"
fi
if [[ ${#failed[@]} -gt 0 ]]; then
  printf '  failed: %s\n' "${failed[@]}"
  exit 1
fi
