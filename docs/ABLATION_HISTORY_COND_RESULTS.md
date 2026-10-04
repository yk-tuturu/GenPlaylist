# History-conditioning ablation: results (2026-10-04)

Results of the ablation described in
[`ABLATIONS.md`](ABLATIONS.md#ablation-1-conditioning-channels), from the bundle
`src/03_backbone_recommender/outputs/ablation-history-cond-results-20261004`.

## Summary

- **Music4All:** each channel adds signal and both are needed for the best
  result. Recall@5 ranks Full > shuffled-cue > latent-only > cue-only. Every
  gap to Full is significant (paired bootstrap, p < 1e-4) and holds for all
  three seeds.
- **Latent references carry more signal than cues.** Removing the latent
  channel (cue-only) costs 40% of Full's Recall@5 on Music4All. Removing the
  cues (latent-only) costs 18%.
- **Cue semantics matter on Music4All.** Shuffled-cue keeps the cue tokens but
  takes them from another user, and it loses 12% Recall@5. So the model reads
  what the cues say, not only that cue tokens are present.
- **MPD gives the same direction, but the evidence is weak.** There is one
  seed and 941 test histories, and no Recall@5 gap is significant. On MPD,
  shuffled-cue matches Full on retrieval (Recall@5 0.0191 vs 0.0185). Full
  still leads on cue F1, N1-MERT and CLHE cosine.
- **MERT metrics hardly separate the variants.** M2M-MERT spans only
  0.8791–0.8803 on Music4All and 0.8907–0.8913 on MPD. Exact-item and CLHE
  metrics are much more sensitive.

## Data checked

| | MPD | Music4All (v3-u500) |
|---|---|---|
| Seeds evaluated | 1 (folders `mpd-<cond>`) | 1, 2, 3 |
| Test histories | 941 | 19,771 |
| Catalog items | 5,119 | 10,950 |
| Chance Recall@5 (5 / catalog) | 0.00098 | 0.00046 |

All 16 results passed these checks:

- `official_protocol: true`.
- The checkpoint condition equals the test-context condition, so no mismatch
  diagnostics are mixed in.
- 256 sampling steps, eval seed 1, EMA.
- Target item IDs are identical across all variants of a dataset, so the
  history-level comparisons are paired.
- Recall@5 recomputed from the saved predictions matches the MERT JSON.

## Results

Music4All values are the mean ± SD over three seeds. MPD values are from a
single seed.

### Music4All

| Variant | Recall@5 | Hit@5 | N1-MERT | M2M-MERT | CLHE M2M cos | Cue F1 | Coverage@5 |
|---|---|---|---|---|---|---|---|
| **Full** | **0.0096** ± 0.0003 | **0.0431** | **0.8649** | **0.8803** | **0.3673** | **0.4449** | 0.7147 |
| Latent-only | 0.0079 ± 0.0003 | 0.0358 | 0.8644 | 0.8800 | 0.3618 | 0.4386 | 0.7181 |
| Cue-only | 0.0058 ± 0.0003 | 0.0266 | 0.8635 | 0.8791 | 0.3328 | 0.4411 | 0.7209 |
| Shuffled-cue | 0.0084 ± 0.0003 | 0.0380 | 0.8645 | 0.8799 | 0.3655 | 0.4401 | 0.7176 |

The seed SDs are small: about 0.0003 for Recall@5, 0.001 for cue F1 and
0.0004 for N1-MERT. Every gap to Full in Recall@5, Hit@5, CLHE cosine and
cue F1 is several SDs wide.

### MPD (seed 1)

| Variant | Recall@5 [95% CI] | Hit@5 | N1-MERT | M2M-MERT | CLHE M2M cos | Cue F1 | Coverage@5 |
|---|---|---|---|---|---|---|---|
| **Full** | 0.0185 [0.0149, 0.0223] | **0.0925** | **0.8747** | 0.8912 | **0.5831** | **0.2842** | 0.3686 |
| Latent-only | 0.0164 [0.0128, 0.0200] | 0.0786 | 0.8720 | 0.8908 | 0.5692 | 0.2743 | 0.3798 |
| Cue-only | 0.0157 [0.0123, 0.0193] | 0.0776 | 0.8698 | 0.8907 | 0.5577 | 0.2724 | 0.3739 |
| Shuffled-cue | **0.0191** [0.0153, 0.0232] | 0.0914 | 0.8715 | **0.8913** | 0.5821 | 0.2777 | 0.3723 |

### Paired Recall@5 differences

Each variant is compared with Full history by history: 10,000 bootstrap
samples, seed 42. For Music4All, each history's Recall@5 is averaged over the
three seeds first.

| Comparison | MPD Δ Recall@5 [95% CI] | Music4All Δ Recall@5 [95% CI] | Music4All per seed (1 / 2 / 3) |
|---|---|---|---|
| Latent-only − Full | −0.0021 [−0.0066, +0.0026] | **−0.0017** [−0.0021, −0.0013] | −0.0015 / −0.0023 / −0.0013 |
| Cue-only − Full | −0.0028 [−0.0070, +0.0015] | **−0.0038** [−0.0042, −0.0034] | −0.0035 / −0.0045 / −0.0034 |
| Shuffled-cue − Full | +0.0006 [−0.0038, +0.0049] | **−0.0012** [−0.0016, −0.0008] | −0.0012 / −0.0013 / −0.0010 |
| Latent-only − Cue-only | +0.0006 [−0.0038, +0.0051] | **+0.0021** [+0.0017, +0.0025] | +0.0020 / +0.0022 / +0.0021 |

Bold values have CIs that exclude zero.

Relative to Full on Music4All:

| Variant | Recall@5 | Hit@5 |
|---|---|---|
| Latent-only | −18% | −17% |
| Cue-only | −40% | −38% |
| Shuffled-cue | −12% | −12% |

## Reading the comparisons

| Comparison | Question | MPD | Music4All |
|---|---|---|---|
| Full vs shuffled-cue | Do cue *semantics* matter? | Not for retrieval. Full is better on cue F1 (+0.007) and N1-MERT (+0.003). | **Yes.** Full is better on every metric. |
| Full vs latent-only | Do cues add anything over latent references? | Same direction, not significant | **Yes**, +0.0017 Recall@5 |
| Full vs cue-only | Do latent references add anything over cues? | Same direction, not significant | **Yes**, +0.0038 Recall@5 |
| Latent-only vs cue-only | Which channel carries more signal? | Too close to call | **Latent**, on Recall@5, Hit@5, CLHE cosine and N1-MERT |

### Channel-specific effects

- **Cue F1 follows the cue channel.** On Music4All, cue-only (0.4411) beats
  latent-only (0.4386) and shuffled-cue (0.4401) on cue F1, even though it is
  worst on item retrieval. Real history cues help the model generate the right
  target cues. Wrong cues (shuffled) help about as little as none
  (latent-only).
- **CLHE similarity follows the latent channel.** Without latent references,
  CLHE M2M cosine falls from 0.367 to 0.333 on Music4All and from 0.583 to
  0.558 on MPD. These are the largest relative drops of any variant on that
  metric. Dropping the cues costs only about 0.005 on Music4All and 0.014 on
  MPD.
- **Coverage rises a little when conditioning gets weaker.** Every ablated
  variant has slightly higher Coverage@5 than Full: 0.718–0.721 vs 0.715 on
  Music4All. With less history to condition on, predictions spread more
  evenly over the catalog. The prediction-concentration statistics show no
  popularity collapse: the most-predicted item accounts for ≤0.6% of all
  predictions in every run.

### Prediction overlap

Prediction overlap is the mean per-history Jaccard similarity of the predicted
item sets. Each variant was compared with Full trained with the same seed.

| | Latent-only | Cue-only | Shuffled-cue | Full, seed 1 vs seed 2 |
|---|---|---|---|---|
| MPD | 0.066 | 0.064 | 0.084 | – |
| Music4All | 0.084 | 0.069 | 0.116 | 0.080 |

Predictions vary a lot even within one variant: two Full seeds agree on only
about 8% of items. Shuffled-cue is the variant closest to Full and cue-only is
the furthest. Shuffled-cue even overlaps Full more than a second Full seed
does, but that comparison is not like for like: models trained with the same
seed share initialization and data order. This agrees with the retrieval
ranking: keeping the latent channel and the cue token layout preserves most of
Full's behavior.

## Caveats

- **MPD has only seed 1.** `runs.csv` shows that seeds 2 and 3 were trained for
  all four MPD variants, but their results are not in this bundle. Evaluating
  them would show whether the MPD shuffled-cue ≈ Full result is real or noise:

  ```bash
  bash scripts/ablation/eval_history_cond.sh mpd --gpu N --seeds "2 3"
  ```

- **No paired bootstrap for MERT metrics.** The MERT result files store only
  aggregates, and the MERT catalog embeddings are not available locally.
  Computing paired MERT and cue-F1 differences needs the per-history values.
  The simplest option is for `evaluate_mert_proxy.py` to save `per_history`.
- **One sampling seed.** All variants use eval seed 1, so the seed SDs reflect
  training variance only.
- **The MPD seed-1 runs were moved by hand.** They are not in `runs.csv`.
  Their checkpoint hashes appear only in each result JSON (`checkpoint.sha256`).
- **The no-cues baseline (DDBC-SFT) and the "Full on shuffled test cues"
  diagnostic are not in this bundle.** The diagnostic would test directly
  whether Full reads its cues at inference.

## Provenance

- All runs: git commit `5fd6457`, clean worktree, 20,000 steps, warm start,
  loss `rvq-cue-warmup-cw0.1to1.0-s1000to5000`, 8 active cues.
- Prepared-manifest SHA-256 values per variant:

  | Variant | MPD | Music4All |
  |---|---|---|
  | Full | `58048ab4…` | `dc0305ed…` |
  | Latent-only | `57adb73f…` | `255eaf55…` |
  | Cue-only | `a65618c2…` | `69fac6f1…` |
  | Shuffled-cue | `3ea94989…` | `613d0c95…` |

- Checkpoint hashes are listed in `runs.csv`, and in each result JSON for the
  MPD seed-1 runs.
- The analysis script, which recomputes Recall@5 from the predictions and
  runs the paired bootstrap, was a one-off and is not in the repository.
