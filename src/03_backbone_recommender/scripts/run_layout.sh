# Shared run-folder layout for train_spotify.sh and eval_spotify.sh (sourced).
#
# GENPLAYLIST_OUTPUT_NAME names a fixed run folder under $OUTPUT_ROOT, for
# example "ablation-cond/mpd/cue_only-seed2". Without it, Hydra's dated
# outputs/<data>/<date>/<time>/ folders are used exactly as before.
#
# Expects WP_ROOT.

OUTPUT_ROOT="${GENPLAYLIST_OUTPUT_ROOT:-$WP_ROOT/outputs}"
OUTPUT_NAME="${GENPLAYLIST_OUTPUT_NAME:-}"
RUN_DIR=""
if [[ -n "$OUTPUT_NAME" ]]; then
  if [[ ! "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ \
        || "/$OUTPUT_NAME/" == */../* || "/$OUTPUT_NAME/" == */./* ]]; then
    echo "GENPLAYLIST_OUTPUT_NAME may only contain letters, digits, '.', '_', '-' and '/' between names: $OUTPUT_NAME" >&2
    exit 2
  fi
  RUN_DIR="$OUTPUT_ROOT/$OUTPUT_NAME"
fi
