#!/usr/bin/env bash
# Cue-budget ablation: build the random-8 cue table and prepare + validate the
# WP-C data for each cue budget of one dataset.
#
# Usage:  bash scripts/ablation/prepare_cue_budget.sh <mpd|m4a> [--budgets "0 4 random8 16"]
#   e.g.  bash scripts/ablation/prepare_cue_budget.sh mpd
#         bash scripts/ablation/prepare_cue_budget.sh m4a --budgets "random8"
#
# Budget 8 (ranked top 8) reuses the history-conditioning Full data
# (<prefix>-8cue-full) and is not prepared here. Folders that already have a
# prepared_manifest.json are skipped. The checkout must be committed and must
# not change while this runs. See docs/ABLATIONS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/ablation/common.sh"

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ablation_dataset "${1:-}" || exit 2
shift
BUDGETS="0 4 random8 16"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budgets) BUDGETS="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for budget in $BUDGETS; do cue_budget_variant "$budget" || exit 2; done
ablation_env_is_clean || exit 2
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "The checkout has uncommitted changes; prepare_wp_c_data.py would refuse:" >&2
  git -C "$REPO" status --short >&2
  exit 1
fi
for f in "$DATA_ROOT/catalog_metadata.json" "$ARTIFACT_ROOT/item_id_to_row.json" \
         "$CUE_ROOT/item2cues.json" "$CUE_ROOT/cue_manifest.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

failed=()
for budget in $BUDGETS; do
  cue_budget_variant "$budget"
  echo "=== $(date '+%F %T')  $DATASET  budget $budget  ($ACTIVE_CUES cues, $TOKENS tokens)"
  if [[ "$budget" == 8 ]]; then
    echo "  ranked top 8 reuses $PREPARED; nothing to prepare"
    continue
  fi
  if [[ "$budget" == random8 && ! -f "$RANDOM_CUE_ROOT/item2cues.json" ]]; then
    echo "  building random-8 cue table -> $RANDOM_CUE_ROOT"
    python "$REPO/scripts/ablation/build_random_cue_table.py" --cue-dir "$CUE_ROOT" \
      --output-dir "$RANDOM_CUE_ROOT" --select 8 --seed "$RANDOM_CUE_SEED" \
      || { failed+=("$budget (random cue table)"); continue; }
  fi
  common_args=(--data-dir "$DATA_ROOT" --artifact-dir "$ARTIFACT_ROOT"
               --cue-dir "$VARIANT_CUE_ROOT" --active-cues "$ACTIVE_CUES")
  if [[ -f "$PREPARED/prepared_manifest.json" ]]; then
    echo "  already prepared: $PREPARED (validating only)"
  elif ! python "$REPO/scripts/prepare_wp_c_data.py" "${common_args[@]}" \
         --output-dir "$PREPARED"; then
    failed+=("$budget (prepare)"); continue
  fi
  if python "$REPO/scripts/validate_wp_c_prepared_data.py" "${common_args[@]}" \
       --prepared-dir "$PREPARED"; then
    echo "  prepared and validated: $PREPARED"
  else
    failed+=("$budget (validate)")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "=== all done $(date '+%F %T')"
