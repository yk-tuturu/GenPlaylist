#!/usr/bin/env bash
# Model study B1 (quality vs efficiency): evaluate the Full models with
# different numbers of reverse-diffusion (denoising) steps.
#
# Usage:  bash scripts/model_study/eval_sampling_steps.sh <mpd|m4a> --gpu N [--seeds "1 2 3"]
#                    [--steps "16 32 64 128 256"] [--no-mert] [--dry-run]
#   e.g.  bash scripts/model_study/eval_sampling_steps.sh mpd --gpu 4 --seeds "1 2 3"
#
# Models are the history-conditioning Full models (as in eval_history_length.sh).
# Only the step count changes: EMA, seed 1, all test histories. 256 steps is
# the official result (last-steps256-evalseed1.json); it is reused when it
# already records evaluation.timing, otherwise it is re-run into
# last-steps256-evalseed1-timing.json so the official file is never touched.
# Use an idle GPU: time per history is only comparable without other jobs.
# Finished evaluations are skipped. See docs/model_study.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
GPU=""; SEEDS="1"; STEPS_LIST="16 32 64 128 256"; RUN_MERT=true; DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --steps) STEPS_LIST="${2:-}"; shift 2 ;;
    --no-mert) RUN_MERT=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$GPU" ]] || { echo "--gpu is required" >&2; exit 2; }
for value in $SEEDS $STEPS_LIST; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Not a positive number: $value" >&2; exit 2; }
done
ablation_env_is_clean || exit 2
cue_budget_variant 8   # the Full model: 8 ranked cues, history condition full
for f in "$EVAL_SCRIPT" "$PREPARED/prepared_manifest.json" "$DATA_ROOT/catalog_metadata.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
if [[ "$RUN_MERT" == true && ! -f "$MERT_DIR/mert_manifest.json" ]]; then
  echo "Missing MERT folder: $MERT_DIR (or pass --no-mert)" >&2; exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
  [[ -n "$util" && "$util" -gt 10 ]] && \
    echo "WARNING: GPU $GPU is at ${util}% utilization; timing will be inflated." >&2
fi

has_timing() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
ok = path.is_file() and json.loads(path.read_text()).get("evaluation", {}).get("timing")
raise SystemExit(0 if ok else 1)
PY
}

# Result file for one step count; prints the path to use.
result_path() {
  local name=$1 steps=$2
  local official=$OUT/$name/results/last-steps256-evalseed1.json
  if [[ "$steps" == 256 ]]; then
    # No official result yet: create it (it records timing). An official
    # result without timing (evaluated before timing existed) is kept and
    # timed again in a separate file.
    if [[ ! -f "$official" ]] || has_timing "$official"; then echo "$official"
    else echo "$OUT/$name/results/last-steps256-evalseed1-timing.json"; fi
  else
    echo "$OUT/$name/results/last-steps$steps-evalseed1.json"
  fi
}

ran=() skipped=() failed=() pending=()
for seed in $SEEDS; do
  name=$(cue_budget_run_name "$seed")
  for steps in $STEPS_LIST; do
    result=$(result_path "$name" "$steps")
    echo "=== $(date '+%F %T')  $DATASET  seed $seed  $steps steps  ($name)"
    if ! run_is_finished "$name"; then
      if [[ -f "$result" ]]; then
        echo "  STALE: $result was evaluated before training finished; delete it and re-run"
        failed+=("$name $steps (stale result)")
      else
        echo "  not finished training yet (no runs.csv row or step-$FINAL_STEP.ckpt); skipping"
        pending+=("$name $steps")
      fi
      continue
    elif result_is_stale "$result" "$OUT/$name/checkpoints/last.ckpt"; then
      echo "  STALE: $result is older than last.ckpt; delete it and re-run"
      failed+=("$name $steps (stale result)"); continue
    elif [[ -f "$result" ]]; then
      echo "  eval: already done ($(basename "$result"))"; skipped+=("$name $steps")
    elif [[ ! -f "$OUT/$name/checkpoints/last.ckpt" ]]; then
      echo "  no checkpoint in $OUT/$name; skipping"; failed+=("$name (no checkpoint)"); continue
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  eval: would run on GPU $GPU -> $result"
    else
      step_env=()
      [[ "$steps" != 256 ]] && step_env=(GENPLAYLIST_EVAL_ALLOW_PROTOCOL_OVERRIDE=true)
      [[ "$result" == *-timing.json ]] && step_env=(GENPLAYLIST_EVAL_RESULTS_PATH=$result)
      if env ${step_env[@]+"${step_env[@]}"} \
           CUDA_VISIBLE_DEVICES=$GPU \
           GENPLAYLIST_DATA_CONFIG=$DATA_CONFIG \
           GENPLAYLIST_DATA_ROOT=$DATA_ROOT \
           GENPLAYLIST_ARTIFACT_ROOT=$ARTIFACT_ROOT \
           GENPLAYLIST_CUE_ROOT=$VARIANT_CUE_ROOT \
           GENPLAYLIST_ACTIVE_CUES=$ACTIVE_CUES \
           GENPLAYLIST_HISTORY_CONDITION=full \
           GENPLAYLIST_OUTPUT_NAME=$name \
           GENPLAYLIST_EVAL_SAMPLING_STEPS=$steps \
           GENPLAYLIST_PREPARED_DATA_ROOT=$PREPARED \
           bash "$EVAL_SCRIPT"; then
        ran+=("$name $steps")
      else
        echo "  !!! eval failed"; failed+=("$name $steps"); continue
      fi
    fi

    # MERT metrics depend only on the predictions; the -timing re-run of the
    # official setting reuses the official file's MERT result.
    [[ "$RUN_MERT" == true ]] || continue
    mert_source=$result
    [[ "$result" == *-timing.json ]] && mert_source=$OUT/$name/results/last-steps256-evalseed1.json
    mert=${mert_source%.json}-mert.json
    if [[ -f "$mert" ]]; then
      echo "  mert: already done"
    elif [[ "$DRY_RUN" == true ]]; then
      echo "  mert: would run -> $mert"
    elif [[ ! -f "$mert_source" ]]; then
      echo "  !!! no result for MERT: $mert_source"; failed+=("$name $steps mert")
    elif python "$REPO/scripts/evaluate_mert_proxy.py" --prediction-result "$mert_source" \
           --mert-dir "$MERT_DIR" --output "$mert"; then
      ran+=("$name $steps mert")
    else
      echo "  !!! mert failed"; failed+=("$name $steps mert")
    fi
  done
done

echo
echo "=== Summary ($DATASET denoising steps)"
printf '%-6s %-5s %9s %9s %9s %9s %9s %10s  %s\n' steps seed N1-MERT Recall@5 M2M-MERT Cov@5 Unique s/history device
for seed in $SEEDS; do
  name=$(cue_budget_run_name "$seed")
  for steps in $STEPS_LIST; do
    result=$(result_path "$name" "$steps")
    mert_source=$result
    [[ "$result" == *-timing.json ]] && mert_source=$OUT/$name/results/last-steps256-evalseed1.json
    python - "$result" "${mert_source%.json}-mert.json" "$steps" "$seed" <<'PY'
import json, sys
from pathlib import Path
result, mert, steps, seed = sys.argv[1:]
row = f"{steps:<6} {seed:<5}"
if not Path(result).is_file() or not Path(mert).is_file():
    print(f"{row}  (no result)")
    raise SystemExit
m = json.loads(Path(mert).read_text())["metrics"]
payload = json.loads(Path(result).read_text())
timing = payload.get("evaluation", {}).get("timing", {})
per = timing.get("generation_seconds_per_history")
per = f"{per:10.4f}" if per is not None else f"{'-':>10}"
unique = payload["metrics"].get("m2m_unique_ratio", float("nan"))
print(f"{row} {m['n1_mert']:9.4f} {m['recall_at_5']:9.4f} {m['m2m_mert']:9.4f} "
      f"{m['coverage_at_5']:9.4f} {unique:9.4f} {per}  {timing.get('device', '-')}")
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
