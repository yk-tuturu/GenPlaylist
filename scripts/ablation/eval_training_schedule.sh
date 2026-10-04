#!/usr/bin/env bash
# Training-schedule ablation: evaluate one dataset's loss-schedule models.
#
# Usage:  bash scripts/ablation/eval_training_schedule.sh <mpd|m4a> --gpu N [--seeds "1 2 3"]
#             [--schedules "warmup fixed uniform"] [--mode final|curve|both]
#             [--curve-checkpoints "1000 ... 20000"] [--curve-sampling-steps 64]
#             [--curve-examples N|all] [--no-mert] [--dry-run]
#   e.g.  bash scripts/ablation/eval_training_schedule.sh mpd --gpu 4 --seeds "1 2 3" --mode both
#
# final  official evaluation of last.ckpt (EMA, 256 steps, all test
#        histories) plus MERT metrics: the numbers for the results table.
# curve  unofficial training-curve points: each step-<K>.ckpt with raw (non-EMA)
#        weights, --curve-sampling-steps denoising steps, and a fixed random
#        subset of --curve-examples test histories (default: all for MPD,
#        2000 for Music4All). The last point is also evaluated at 256 steps
#        on the same subset, to check that the reduced setting keeps the
#        ranking of the schedules.
# Finished evaluations are skipped, so re-running is safe. See docs/ABLATIONS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"
RESULT_FILE=last-steps256-evalseed1.json
SUBSET_SEED=0

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
GPU=""; SEEDS="1"; SCHEDULES="$TRAINING_SCHEDULES_ALL"; MODE=final
CURVE_CHECKPOINTS="1000 2000 3000 4000 5000 7500 10000 15000 20000"
CURVE_STEPS=64; CURVE_EXAMPLES=""; RUN_MERT=true; DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --schedules) SCHEDULES="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --curve-checkpoints) CURVE_CHECKPOINTS="${2:-}"; shift 2 ;;
    --curve-sampling-steps) CURVE_STEPS="${2:-}"; shift 2 ;;
    --curve-examples) CURVE_EXAMPLES="${2:-}"; shift 2 ;;
    --no-mert) RUN_MERT=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$GPU" ]] || { echo "--gpu is required" >&2; exit 2; }
case "$MODE" in final|curve|both) ;; *) echo "--mode must be final, curve, or both" >&2; exit 2 ;; esac
for value in $SEEDS $CURVE_CHECKPOINTS $CURVE_STEPS; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Not a number: $value" >&2; exit 2; }
done
if [[ -z "$CURVE_EXAMPLES" ]]; then
  [[ "$DATASET" == m4a ]] && CURVE_EXAMPLES=2000 || CURVE_EXAMPLES=all
fi
[[ "$CURVE_EXAMPLES" == all || "$CURVE_EXAMPLES" =~ ^[1-9][0-9]*$ ]] \
  || { echo "--curve-examples must be a positive number or all" >&2; exit 2; }
for schedule in $SCHEDULES; do training_schedule_variant "$schedule" || exit 2; done
ablation_env_is_clean || exit 2
for f in "$EVAL_SCRIPT" "$DATA_ROOT/catalog_metadata.json" "$ARTIFACT_ROOT/item_id_to_row.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
if [[ "$MODE" != curve && "$RUN_MERT" == true && ! -f "$MERT_DIR/mert_manifest.json" ]]; then
  echo "Missing MERT folder: $MERT_DIR (or pass --no-mert)" >&2; exit 1
fi
LAST_CURVE_POINT=${CURVE_CHECKPOINTS##* }
CURVE_SUFFIX="-raw"
[[ "$CURVE_EXAMPLES" != all ]] && CURVE_SUFFIX+="-n${CURVE_EXAMPLES}s${SUBSET_SEED}"

ran=() skipped=() failed=()

# eval_one <name> <checkpoint file> <sampling steps> <official:true|false>
eval_one() {
  local name=$1 ckpt_file=$2 steps=$3 official=$4
  local label=${ckpt_file%.ckpt} suffix=""
  [[ "$official" == false ]] && suffix=$CURVE_SUFFIX
  local result=$OUT/$name/results/$label-steps$steps-evalseed1$suffix.json
  if [[ -f "$result" ]]; then
    echo "  $label @$steps: already done"; skipped+=("$name $label@$steps"); return 0
  fi
  if [[ ! -f "$OUT/$name/checkpoints/$ckpt_file" ]]; then
    echo "  $label: no checkpoint $ckpt_file; skipping"; failed+=("$name $ckpt_file (missing)"); return 1
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "  $label @$steps: would run -> $result"; return 0
  fi
  local extra=()
  if [[ "$official" == false ]]; then
    extra=(GENPLAYLIST_EVAL_ALLOW_PROTOCOL_OVERRIDE=true GENPLAYLIST_EVAL_DISABLE_EMA=true)
    [[ "$CURVE_EXAMPLES" != all ]] && extra+=(GENPLAYLIST_EVAL_MAX_EXAMPLES=$CURVE_EXAMPLES
                                             GENPLAYLIST_EVAL_SUBSET_SEED=$SUBSET_SEED)
  fi
  if env ${extra[@]+"${extra[@]}"} \
       CUDA_VISIBLE_DEVICES=$GPU \
       GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
       GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
       GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
       GENPLAYLIST_CUE_ROOT=$VARIANT_CUE_ROOT \
       GENPLAYLIST_ACTIVE_CUES=$ACTIVE_CUES \
       GENPLAYLIST_HISTORY_CONDITION=full \
       GENPLAYLIST_OUTPUT_NAME=$name \
       GENPLAYLIST_EVAL_CKPT_FILE=$ckpt_file \
       GENPLAYLIST_EVAL_SAMPLING_STEPS=$steps \
       GENPLAYLIST_PREPARED_DATA_ROOT=$PREPARED \
       bash "$EVAL_SCRIPT"; then
    ran+=("$name $label@$steps")
  else
    echo "  !!! $label @$steps failed"; failed+=("$name $label@$steps"); return 1
  fi
}

for seed in $SEEDS; do
  for schedule in $SCHEDULES; do
    training_schedule_variant "$schedule"
    name=$RUN_PREFIX-seed$seed
    echo "=== $(date '+%F %T')  $DATASET  $schedule  seed $seed  ($name)"
    if [[ "$MODE" != curve ]]; then
      if eval_one "$name" last.ckpt 256 true && [[ "$RUN_MERT" == true ]]; then
        result=$OUT/$name/results/$RESULT_FILE
        mert=${result%.json}-mert.json
        if [[ -f "$mert" ]]; then
          echo "  mert: already done"
        elif [[ "$DRY_RUN" == true ]]; then
          echo "  mert: would run -> $mert"
        elif python "$REPO/scripts/evaluate_mert_proxy.py" --prediction-result "$result" \
               --mert-dir "$MERT_DIR" --output "$mert"; then
          ran+=("$name mert")
        else
          echo "  !!! mert failed"; failed+=("$name mert")
        fi
      fi
    fi
    if [[ "$MODE" != final ]]; then
      for step in $CURVE_CHECKPOINTS; do
        eval_one "$name" "step-$step.ckpt" "$CURVE_STEPS" false
      done
      # Sampling-steps check at the last curve point, same subset and weights.
      if [[ "$CURVE_STEPS" != 256 ]]; then
        eval_one "$name" "step-$LAST_CURVE_POINT.ckpt" 256 false
      fi
    fi
  done
done

echo
if [[ "$MODE" != curve ]]; then
  echo "=== Final ($DATASET training schedule; official)"
  printf '%-8s %-5s %9s %9s %9s %9s %9s\n' schedule seed N1-MERT Recall@5 M2M-MERT Cov@5 CueF1
  for seed in $SEEDS; do
    for schedule in $SCHEDULES; do
      training_schedule_variant "$schedule"
      result=$OUT/$RUN_PREFIX-seed$seed/results/$RESULT_FILE
      python - "$result" "${result%.json}-mert.json" "$schedule" "$seed" <<'PY'
import json, sys
from pathlib import Path
result, mert, schedule, seed = sys.argv[1:]
row = f"{schedule:<8} {seed:<5}"
if not Path(mert).is_file() or not Path(result).is_file():
    print(f"{row}  (no result)")
    raise SystemExit
m = json.loads(Path(mert).read_text())["metrics"]
cue = json.loads(Path(result).read_text())["metrics"].get("m2m_cue_f1", float("nan"))
print(f"{row} {m['n1_mert']:9.4f} {m['recall_at_5']:9.4f} {m['m2m_mert']:9.4f} "
      f"{m['coverage_at_5']:9.4f} {cue:9.4f}")
PY
    done
  done
fi
if [[ "$MODE" != final ]]; then
  echo "=== Curve ($DATASET; raw weights, $CURVE_STEPS steps, examples: $CURVE_EXAMPLES; unofficial)"
  printf '%-8s %-5s %7s %6s %9s %9s %9s\n' schedule seed ckpt steps Recall@5 CueF1 CLHEcos
  for seed in $SEEDS; do
    for schedule in $SCHEDULES; do
      training_schedule_variant "$schedule"
      for step in $CURVE_CHECKPOINTS check; do
        steps=$CURVE_STEPS; point=$step
        if [[ "$step" == check ]]; then
          [[ "$CURVE_STEPS" == 256 ]] && continue
          steps=256; point=$LAST_CURVE_POINT
        fi
        result=$OUT/$RUN_PREFIX-seed$seed/results/step-$point-steps$steps-evalseed1$CURVE_SUFFIX.json
        python - "$result" "$schedule" "$seed" "$point" "$steps" <<'PY'
import json, sys
from pathlib import Path
result, schedule, seed, point, steps = sys.argv[1:]
row = f"{schedule:<8} {seed:<5} {point:>7} {steps:>6}"
if not Path(result).is_file():
    print(f"{row}  (no result)")
    raise SystemExit
m = json.loads(Path(result).read_text())["metrics"]
print(f"{row} {m.get('m2m_recall', float('nan')):9.4f} "
      f"{m.get('m2m_cue_f1', float('nan')):9.4f} {m.get('m2m_cosine', float('nan')):9.4f}")
PY
      done
    done
  done
fi
echo "ran: ${#ran[@]}  already done: ${#skipped[@]}  failed: ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
  printf '  failed: %s\n' "${failed[@]}"
  exit 1
fi
