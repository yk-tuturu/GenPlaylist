#!/usr/bin/env bash
# Cue-budget ablation: evaluate one dataset's cue-budget models (WP-C proxy
# eval + MERT metrics) and print a summary with the cost of each budget.
#
# Usage:  bash scripts/ablation/eval_cue_budget.sh <mpd|m4a> --gpu N [--seeds "1 2 3"]
#                    [--budgets "0 4 8 random8 16"] [--no-mert] [--dry-run]
#   e.g.  bash scripts/ablation/eval_cue_budget.sh mpd --gpu 4 --seeds "1 2 3"
#         bash scripts/ablation/eval_cue_budget.sh m4a --gpu 7 --budgets "8 random8" --dry-run
#
# Budget 8 (ranked top 8) is the history-conditioning Full model
# (ablation-history-cond/<ds>-full-seed<N>, MPD seed 1: mpd-full); its result
# is reused when it already exists. Finished evaluations and MERT outputs are
# skipped, so re-running is safe. See docs/ABLATIONS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"
RESULT_FILE=last-steps256-evalseed1.json

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
GPU=""; SEEDS="1"; BUDGETS="$CUE_BUDGETS_ALL"; RUN_MERT=true; DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --budgets) BUDGETS="${2:-}"; shift 2 ;;
    --no-mert) RUN_MERT=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$GPU" ]] || { echo "--gpu is required" >&2; exit 2; }
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $seed" >&2; exit 2; }
done
for budget in $BUDGETS; do cue_budget_variant "$budget" || exit 2; done
ablation_env_is_clean || exit 2
for f in "$EVAL_SCRIPT" "$DATA_ROOT/catalog_metadata.json" "$ARTIFACT_ROOT/item_id_to_row.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
if [[ "$RUN_MERT" == true && ! -f "$MERT_DIR/mert_manifest.json" ]]; then
  echo "Missing MERT folder: $MERT_DIR (or pass --no-mert)" >&2; exit 1
fi

ran=() skipped=() failed=() pending=()
for seed in $SEEDS; do
  for budget in $BUDGETS; do
    cue_budget_variant "$budget"
    name=$(cue_budget_run_name "$seed")
    ckpt=$OUT/$name/checkpoints/last.ckpt
    result=$OUT/$name/results/$RESULT_FILE
    mert=${result%.json}-mert.json
    echo "=== $(date '+%F %T')  $DATASET  budget $budget  seed $seed  ($name)"

    if ! run_is_finished "$name"; then
      if [[ -f "$result" ]]; then
        echo "  STALE: $result was evaluated before training finished; delete it and re-run"
        failed+=("$name (stale result)")
      else
        echo "  not finished training yet (no runs.csv row or step-$FINAL_STEP.ckpt); skipping"
        pending+=("$name")
      fi
      continue
    elif result_is_stale "$result" "$ckpt"; then
      echo "  STALE: $result is older than $ckpt; delete it and re-run"
      failed+=("$name (stale result)"); continue
    elif [[ -f "$result" ]]; then
      echo "  eval: already done"; skipped+=("$name eval")
    elif [[ ! -f "$ckpt" ]]; then
      echo "  no checkpoint at $ckpt; skipping"; failed+=("$name (no checkpoint)"); continue
    elif [[ ! -f "$PREPARED/prepared_manifest.json" ]]; then
      echo "  no prepared data at $PREPARED; skipping"; failed+=("$name (no prepared data)"); continue
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  eval: would run on GPU $GPU -> $result"
    elif CUDA_VISIBLE_DEVICES=$GPU \
         GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
         GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
         GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
         GENPLAYLIST_CUE_ROOT=$VARIANT_CUE_ROOT \
         GENPLAYLIST_ACTIVE_CUES=$ACTIVE_CUES \
         GENPLAYLIST_HISTORY_CONDITION=full \
         GENPLAYLIST_OUTPUT_NAME=$name \
         GENPLAYLIST_PREPARED_DATA_ROOT=$PREPARED \
           bash "$EVAL_SCRIPT"; then
      ran+=("$name eval")
    else
      echo "  !!! eval failed"; failed+=("$name eval"); continue
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
echo "=== Summary ($DATASET cue budget)"
printf '%-8s %-5s %6s %9s %9s %9s %9s %9s\n' budget seed tokens N1-MERT Recall@5 M2M-MERT Cov@5 s/history
for seed in $SEEDS; do
  for budget in $BUDGETS; do
    cue_budget_variant "$budget"
    result=$OUT/$(cue_budget_run_name "$seed")/results/$RESULT_FILE
    python - "$result" "${result%.json}-mert.json" "$budget" "$seed" "$TOKENS" <<'PY'
import json, sys
from pathlib import Path
result, mert, budget, seed, tokens = sys.argv[1:]
row = f"{budget:<8} {seed:<5} {tokens:>6}"
if not Path(mert).is_file():
    print(f"{row}  (no MERT result)")
    raise SystemExit
m = json.loads(Path(mert).read_text())["metrics"]
timing = json.loads(Path(result).read_text()).get("evaluation", {}).get("timing", {})
per = timing.get("generation_seconds_per_history")
per = f"{per:9.4f}" if per is not None else f"{'-':>9}"
print(f"{row} {m['n1_mert']:9.4f} {m['recall_at_5']:9.4f} {m['m2m_mert']:9.4f} "
      f"{m['coverage_at_5']:9.4f} {per}")
PY
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
