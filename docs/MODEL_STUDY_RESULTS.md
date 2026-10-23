# GenPlaylist model study results (studies A and B1)

Results bundle: [`genplaylist-results-20261006-1159`](../src/03_backbone_recommender/outputs/genplaylist-results-20261006-1159)
(collected 2026-10-06 11:59). Spec: [`model_study.md`](model_study.md). Studies
B2 and C are planned and have no results yet.

**Headline.** More listening history gives better predictions on both
datasets. MPD is still improving at 15 references, while Music4All is nearly
saturated by 10. On the one model evaluated so far, 16 denoising steps match
256 steps, and none of the step counts differs significantly from 256. Cost
scales linearly with steps, so 16 steps take about a fifteenth of the time.

> **Status.** Study A is complete (18 of 18 evaluations). Study B1 has 4 of 24
> evaluations, all on Music4All Full seed 1. Since the 11:12 bundle, the only
> new model-study result is the Music4All seed-1 128-step run. Study A is
> unchanged.

## Contents

- [Method](#method)
- [Study A: available history](#study-a-available-history)
- [Study B1: denoising steps](#study-b1-denoising-steps) ⚠️
- [Completeness and flags](#completeness-and-flags)
- [Next steps](#next-steps)

## Method

- **Models.** Both studies evaluate the Full models of the history-conditioning
  ablation (8 ranked cues, `history_condition=full`, warm-up schedule), seeds
  1-3 per dataset. Evaluation uses EMA weights, sampling seed 1 and all test
  histories (MPD 941, Music4All 19,771).
- **Metrics.** N1-MERT, Recall@5, M2M-MERT and Coverage@5 come from each
  `*-mert.json`. Cue F1, Hit and the prediction unique ratio come from the
  evaluation JSON. Seconds per history come from `evaluation.timing`. Recall@5
  is shown in %.
- **Aggregation.** Values are the mean over the available seeds, with the seed
  range in brackets.
- **Significance.** Δ Recall@5 is a paired, history-level bootstrap (10,000
  samples, seed 42) against the reference setting (k = 15 for study A, 256
  steps for study B1). It was recomputed from the stored predictions:
  per-history Recall@5 is the multiset overlap of predicted and target item
  IDs divided by 5, which reproduces the stored `m2m_recall` for every file.
  Histories from all seeds are pooled into one bootstrap. The relative change
  is the mean difference divided by the reference mean. **Bold** marks a 95%
  CI that excludes zero. MERT metrics could not be bootstrapped because the
  catalog embeddings are not in the bundle.
- **Validity checks.** Every result was created after its model's `runs.csv`
  finish time, and its checkpoint SHA-256 matches `runs.csv`, so **no result
  is STALE**. The exception is MPD seed 1 (`mpd-full`), which predates
  `runs.csv` and cannot be checked from this bundle.

## Study A: available history

The model sees only the most recent k of its 15 references; the older ones
are blanked with the null tokens (see the spec). Complete: 3 k values × 3
seeds × 2 datasets = 18 of 18, plus the reused official k = 15 results.

| Dataset | Kept refs k | Recall@5 % | Δ R@5 vs k = 15 (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | Hit | Cue F1 |
|---|---|---|---|---|---|---|---|---|
| MPD | 1 | 0.213 [0.085–0.319] | **−88.7%** (−1.906 to −1.431 pp) | 0.8684 | 0.8848 | 0.195 | 0.010 | 0.166 |
| MPD | 5 | 0.878 [0.404–1.126] | **−53.2%** (−1.247 to −0.751 pp) | 0.8708 | 0.8880 | 0.314 | 0.042 | 0.219 |
| MPD | 10 | 1.332 [0.935–1.552] | **−29.1%** (−0.786 to −0.298 pp) | 0.8720 | 0.8904 | 0.360 | 0.064 | 0.255 |
| MPD | 15 | 1.877 [1.849–1.934] | — | 0.8730 | 0.8911 | 0.374 | 0.090 | 0.281 |
| Music4All | 1 | 0.244 [0.113–0.376] | **−74.5%** (−0.754 to −0.672 pp) | 0.8604 | 0.8751 | 0.594 | 0.012 | 0.364 |
| Music4All | 5 | 0.673 [0.410–0.829] | **−29.7%** (−0.322 to −0.245 pp) | 0.8631 | 0.8785 | 0.696 | 0.030 | 0.416 |
| Music4All | 10 | 0.884 [0.765–0.950] | **−7.6%** (−0.106 to −0.039 pp) | 0.8647 | 0.8799 | 0.716 | 0.040 | 0.435 |
| Music4All | 15 | 0.957 [0.928–0.991] | — | 0.8649 | 0.8803 | 0.715 | 0.043 | 0.445 |

- **Every metric rises with k on both datasets.** The one exception is
  Music4All Coverage@5, which is flat from k = 10. Every drop against k = 15
  is significant.
- **MPD needs more history than Music4All.** MPD still gains 29% from k = 10
  to 15; Music4All gains 8%. The curated MPD playlists seem to carry
  preference information spread across the whole list, while recent listening
  dominates in Music4All.
- **Diversity falls with short history on MPD.** MPD Coverage@5 halves at
  k = 1 (0.195 vs 0.374), so with little context the model falls back on a
  narrower set of items. Music4All drops less (0.594 vs 0.715).
- **Caveat: low k is out of distribution.** The Full models were trained only
  on 15 real references. At low k the seed range is far wider than at k = 15
  (Music4All k = 1: 0.113–0.376% vs 0.928–0.991%), and seed 3 is the weakest
  at every k < 15 on both datasets. The three models react very differently
  to blanked references they never saw in training. The k = 1 and k = 5
  drops therefore mix "less history" with an unseen input pattern, and are
  upper bounds on the pure history effect. A model trained with random
  history dropout would separate the two.
- **Excluded.** `mpd-full/results/last-steps16-evalseed1-hist5-n100s0.json`
  (16 steps, 100 histories, no MERT) is a smoke test.

## Study B1: denoising steps

⚠️ **Partial: 4 of 24 evaluations.** Only Music4All Full seed 1 at 16, 32, 64
and 128 steps exists.

**On that one model, quality is flat from 16 to 256 steps, and generation
time scales linearly with steps.** None of the Recall@5 differences from
256 steps is significant, and the MERT metrics agree to within 0.0002.

| Steps | Recall@5 % | Δ R@5 vs 256 (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | Unique ratio | Cue F1 | s / history |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.952 | +2.6% (−0.055 to +0.104 pp) | 0.8648 | 0.8802 | 0.724 | 0.989 | 0.443 | 0.0088 |
| 32 | 0.893 | −3.7% (−0.114 to +0.045 pp) | 0.8647 | 0.8803 | 0.718 | 0.988 | 0.443 | 0.0176 |
| 64 | 0.939 | +1.2% (−0.069 to +0.091 pp) | 0.8649 | 0.8803 | 0.722 | 0.989 | 0.444 | 0.0335 |
| 128 | 0.956 | +3.1% (−0.051 to +0.107 pp) | 0.8648 | 0.8802 | 0.722 | 0.989 | 0.444 | 0.0677 |
| 256 (official) | 0.928 | — | 0.8647 | 0.8802 | 0.715 | 0.988 | 0.444 | *missing* ¹ |

¹ The official result predates timing, and the planned
`last-steps256-evalseed1-timing.json` re-run is not in the bundle. The four
measured step counts cost 0.52–0.55 ms per step per history, which
extrapolates to about 0.135 s at 256 steps. The study A runs of the same
Music4All models at 256 steps (masking does not change compute) measured
0.116–0.131 s on the same GPU type, which fits that estimate.

- **Cost.** 16 steps would cut backbone generation time by about 13–15× (0.0088
  s vs 0.12–0.135 s per history) with no measurable loss.
- **Diversity holds.** Coverage@5 and the unique ratio do not drop at low step
  counts; Coverage@5 is in fact slightly higher at every reduced step count
  (0.718–0.724 vs 0.715).
- **No trend in Recall@5.** The values go up and down with step count (32
  steps is lowest, 128 highest), which looks like sampling noise around a
  flat line rather than a trade-off.
- **Timing is indicative.** The four runs ran back to back on an A40
  (2026-10-06 10:45–11:29); the bundle does not record whether the GPU was
  otherwise idle.
- This rests on a single model and dataset; treat it as a lead, not a result.
  MPD, with its 941 test histories, will have much wider intervals.

## Completeness and flags

| Study | Expected | Present | Missing |
|---|---|---|---|
| A | 18 | 18 | — |
| B1, step sweep | 24 | 4 | Music4All seeds 2-3 (all step counts); MPD seeds 1-3 (all step counts) |
| B1, 256-step timing re-runs | 6 | 0 | All six (none of the official 256-step results records timing) |

All six Full models have finished training, so the missing evaluations need
no training.

- **MPD seed 1 is unverifiable here.** `mpd-full` predates `runs.csv`. The
  runners treat it as finished when it contains `step-20000.ckpt`, but
  checkpoints are not in the bundle. Its official result is dated 2026-09-30,
  and it is the k = 15 and 256-step baseline for MPD seed 1.
- **The bootstrap pools seeds.** A per-seed bootstrap with a seed-level summary
  would give somewhat wider intervals, mainly for MPD.

## Next steps

1. Run study B1 for the remaining five models, including the 256-step timing
   re-runs, on an idle GPU:
   `eval_sampling_steps.sh m4a --seeds "2 3"` and
   `eval_sampling_steps.sh mpd --seeds "1 2 3"`.
2. If 16 steps holds across seeds and on MPD, consider making it the default
   for backbone generation in the listener study and the end-to-end pipeline,
   keeping 256 steps as the official reported setting.
3. For study A, a Full model trained with random history dropout would
   separate the history effect from the out-of-distribution effect at low k.
