#!/usr/bin/env bash
# Model study A (available history): evaluate the Full models with only the
# most recent k of the 15 references, the older ones blanked with null tokens.
#
# Usage:  bash scripts/model_study/eval_history_length.sh <mpd|m4a> --gpu N [--seeds "1 2 3"]
#                    [--lengths "1 5 10 15"] [--no-mert] [--dry-run]
#   e.g.  bash scripts/model_study/eval_history_length.sh mpd --gpu 4 --seeds "1 2 3"
#
# Models are the history-conditioning Full models
# (outputs/ablation-history-cond/<ds>-full-seed<N>; MPD seed 1: mpd-full).
# k = 15 is the official result (last-steps256-evalseed1.json), reused when it
# exists; other k write last-steps256-evalseed1-hist<k>.json. Every k uses the
# official sampling settings (EMA, 256 steps, seed 1, all test histories).
# Finished evaluations are skipped. See docs/model_study.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
GPU=""; SEEDS="1"; LENGTHS="1 5 10 15"; RUN_MERT=true; DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --lengths) LENGTHS="${2:-}"; shift 2 ;;
    --no-mert) RUN_MERT=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$GPU" ]] || { echo "--gpu is required" >&2; exit 2; }
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $seed" >&2; exit 2; }
done
for k in $LENGTHS; do
  [[ "$k" =~ ^[0-9]+$ ]] && (( k >= 1 && k <= 15 )) \
    || { echo "History length must be 1-15: $k" >&2; exit 2; }
done
ablation_env_is_clean || exit 2
cue_budget_variant 8   # the Full model: 8 ranked cues, history condition full
for f in "$EVAL_SCRIPT" "$PREPARED/prepared_manifest.json" "$DATA_ROOT/catalog_metadata.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
if [[ "$RUN_MERT" == true && ! -f "$MERT_DIR/mert_manifest.json" ]]; then
  echo "Missing MERT folder: $MERT_DIR (or pass --no-mert)" >&2; exit 1
fi

result_name() {  # official file for k = 15, -hist<k> otherwise
  if [[ "$1" == 15 ]]; then echo last-steps256-evalseed1.json
  else echo "last-steps256-evalseed1-hist$1.json"; fi
}

ran=() skipped=() failed=()
for seed in $SEEDS; do
  name=$(cue_budget_run_name "$seed")
  for k in $LENGTHS; do
    result=$OUT/$name/results/$(result_name "$k")
    mert=${result%.json}-mert.json
    echo "=== $(date '+%F %T')  $DATASET  seed $seed  k=$k  ($name)"
    if [[ -f "$result" ]]; then
      echo "  eval: already done"; skipped+=("$name k=$k")
    elif [[ ! -f "$OUT/$name/checkpoints/last.ckpt" ]]; then
      echo "  no checkpoint in $OUT/$name; skipping"; failed+=("$name (no checkpoint)"); continue
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  eval: would run on GPU $GPU -> $result"
    else
      history_env=()
      [[ "$k" != 15 ]] && history_env=(GENPLAYLIST_EVAL_HISTORY_LENGTH=$k)
      if env ${history_env[@]+"${history_env[@]}"} \
           CUDA_VISIBLE_DEVICES=$GPU \
           GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
           GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
           GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
           GENPLAYLIST_CUE_ROOT=$VARIANT_CUE_ROOT \
           GENPLAYLIST_ACTIVE_CUES=$ACTIVE_CUES \
           GENPLAYLIST_HISTORY_CONDITION=full \
           GENPLAYLIST_OUTPUT_NAME=$name \
           GENPLAYLIST_PREPARED_DATA_ROOT=$PREPARED \
           bash "$EVAL_SCRIPT"; then
        ran+=("$name k=$k")
      else
        echo "  !!! eval failed"; failed+=("$name k=$k"); continue
      fi
    fi

    [[ "$RUN_MERT" == true ]] || continue
    if [[ -f "$mert" ]]; then
      echo "  mert: already done"
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  mert: would run -> $mert"
    elif python "$REPO/scripts/evaluate_mert_proxy.py" --prediction-result "$result" \
           --mert-dir "$MERT_DIR" --output "$mert"; then
      ran+=("$name k=$k mert")
    else
      echo "  !!! mert failed"; failed+=("$name k=$k mert")
    fi
  done
done

echo
echo "=== Summary ($DATASET available history; k = references kept)"
printf '%-4s %-5s %9s %9s %9s %9s %9s\n' k seed N1-MERT Recall@5 M2M-MERT Cov@5 CueF1
for seed in $SEEDS; do
  name=$(cue_budget_run_name "$seed")
  for k in $LENGTHS; do
    result=$OUT/$name/results/$(result_name "$k")
    python - "$result" "${result%.json}-mert.json" "$k" "$seed" <<'PY'
import json, sys
from pathlib import Path
result, mert, k, seed = sys.argv[1:]
row = f"{k:<4} {seed:<5}"
if not Path(result).is_file() or not Path(mert).is_file():
    print(f"{row}  (no result)")
    raise SystemExit
m = json.loads(Path(mert).read_text())["metrics"]
cue = json.loads(Path(result).read_text())["metrics"].get("m2m_cue_f1", float("nan"))
print(f"{row} {m['n1_mert']:9.4f} {m['recall_at_5']:9.4f} {m['m2m_mert']:9.4f} "
      f"{m['coverage_at_5']:9.4f} {cue:9.4f}")
PY
  done
done
echo "ran: ${#ran[@]}  already done: ${#skipped[@]}  failed: ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
  printf '  failed: %s\n' "${failed[@]}"
  exit 1
fi
