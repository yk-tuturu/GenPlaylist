#!/usr/bin/env bash
# History-conditioning ablation: train the 4 Music4All (v3-u500) variants (full,
# latent_only, cue_only, shuffled_cue) for the given seeds.
#
# Usage:  bash scripts/ablation/train_history_cond_m4a.sh <gpu> <seed> [<seed> ...]
#   e.g.  bash scripts/ablation/train_history_cond_m4a.sh 4 1
#         bash scripts/ablation/train_history_cond_m4a.sh 4 2 3
#
# Runs go to outputs/ablation-history-cond/m4a-<cond>-seed<N> and are logged in
# outputs/runs.csv. Finished runs are refused by train_spotify.sh and skipped.
# See docs/ABLATION_CONDITIONING.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT=$REPO/src/03_backbone_recommender/scripts/train_spotify.sh
CONDS="full latent_only cue_only shuffled_cue"
D=$REPO/data/dataset-music4all-onion-v3-u500
A=$REPO/data/artifacts-music4all-onion-v3-u500
C=$REPO/data/processed/cues-music4all-onion-v3-u500/latest

GPU="${1:-}"; shift || true
SEEDS=("$@")
if [[ -z "$GPU" || ${#SEEDS[@]} -eq 0 ]]; then
  echo "Usage: $0 <gpu> <seed> [<seed> ...]" >&2; exit 2
fi
for SEED in "${SEEDS[@]}"; do
  [[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be a number: $SEED" >&2; exit 2; }
done

# This script sets every data path itself; refuse leftovers from other runs.
if env | grep -q '^GENPLAYLIST_'; then
  echo "Unset these first:" >&2; env | grep '^GENPLAYLIST_' >&2; exit 2
fi

# Check the Music4All inputs and all 4 prepared folders before training anything.
missing=false
for f in "$D/catalog_metadata.json" "$D/dataset_card.json" "$A/item_id_to_row.json" "$C/item2cues.json"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; missing=true; }
done
for V in $CONDS; do
  f=$REPO/data/processed/ablation-m4a-v3-u500-8cue-$V/prepared_manifest.json
  [[ -f "$f" ]] || { echo "Missing prepared data: $f" >&2; missing=true; }
done
[[ "$missing" == false ]] || exit 1

for SEED in "${SEEDS[@]}"; do
  for V in $CONDS; do
    NAME=ablation-history-cond/m4a-$V-seed$SEED
    echo "=== $(date '+%F %T')  $NAME  (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU \
    GENPLAYLIST_DATA_CONFIG=music4all \
    GENPLAYLIST_DATA_ROOT=$D \
    GENPLAYLIST_ARTIFACT_ROOT=$A \
    GENPLAYLIST_CUE_ROOT=$C \
    GENPLAYLIST_OUTPUT_NAME=$NAME \
    GENPLAYLIST_HISTORY_CONDITION=$V \
    GENPLAYLIST_SEED=$SEED \
    GENPLAYLIST_PREPARED_DATA_ROOT=$REPO/data/processed/ablation-m4a-v3-u500-8cue-$V \
      bash "$TRAIN_SCRIPT" \
      || echo "!!! $NAME failed; continuing with the next run"
  done
done
echo "=== all done $(date '+%F %T')"
