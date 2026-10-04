#!/usr/bin/env bash
# History-conditioning ablation: train the 4 MPD variants (full, latent_only,
# cue_only, shuffled_cue) for the given seeds, one after another.
#
# Usage:  bash scripts/ablation/train_history_cond_mpd.sh <gpu> <seed> [<seed> ...]
#   e.g.  bash scripts/ablation/train_history_cond_mpd.sh 4 2 3
#
# Runs go to outputs/ablation-history-cond/mpd-<cond>-seed<N> and are logged in
# outputs/runs.csv. Finished runs are refused by train_spotify.sh and skipped.
# See docs/ABLATION_CONDITIONING.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT=$REPO/src/03_backbone_recommender/scripts/train_spotify.sh
CONDS="full latent_only cue_only shuffled_cue"

GPU="${1:-}"; shift || true
SEEDS=("$@")
if [[ -z "$GPU" || ${#SEEDS[@]} -eq 0 ]]; then
  echo "Usage: $0 <gpu> <seed> [<seed> ...]" >&2; exit 2
fi
for SEED in "${SEEDS[@]}"; do
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $SEED" >&2; exit 2; }
done

# MPD uses the train_spotify.sh defaults; refuse leftovers such as Music4All paths.
if env | grep -q '^GENPLAYLIST_'; then
  echo "Unset these first:" >&2; env | grep '^GENPLAYLIST_' >&2; exit 2
fi

missing=false
for V in $CONDS; do
  f=$REPO/data/processed/ablation-mpd-8cue-$V/prepared_manifest.json
  [[ -f "$f" ]] || { echo "Missing prepared data: $f" >&2; missing=true; }
done
[[ "$missing" == false ]] || exit 1

for SEED in "${SEEDS[@]}"; do
  for V in $CONDS; do
    NAME=ablation-history-cond/mpd-$V-seed$SEED
    echo "=== $(date '+%F %T')  $NAME  (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU \
    GENPLAYLIST_OUTPUT_NAME=$NAME \
    GENPLAYLIST_HISTORY_CONDITION=$V \
    GENPLAYLIST_SEED=$SEED \
    GENPLAYLIST_PREPARED_DATA_ROOT=$REPO/data/processed/ablation-mpd-8cue-$V \
      bash "$TRAIN_SCRIPT" \
      || echo "!!! $NAME failed; continuing with the next run"
  done
done
echo "=== all done $(date '+%F %T')"
