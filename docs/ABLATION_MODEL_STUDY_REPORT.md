# GenPlaylist ablation and model study report

Results bundle: [`genplaylist-results-20261006-1112`](../src/03_backbone_recommender/outputs/genplaylist-results-20261006-1112)
(collected 2026-10-06 11:12). Specs: [`ABLATIONS.md`](ABLATIONS.md) and
[`model_study.md`](model_study.md).

**Headline.** Cues in the *history* help item retrieval (removing them costs
17-18% Recall@5 on both datasets), but training the model to *predict* cues
costs more than that: the 0-cue model has the best Recall@5 and Coverage@5 on
both datasets, and Recall@5 falls as the cue budget grows. More listening
history helps on both datasets, and MPD is still improving at 15 references.
The warm-up schedule shows no significant final-score benefit over fixed
weights. On the one model evaluated so far, 16 denoising steps match 256 steps
at roughly a tenth of the generation time.

> **Incomplete results.** Ablation 1 and study A are complete. Ablation 2 is
> missing Music4All seeds 2-3, ablation 3 is missing 10 of 18 final results
> (all Music4All `uniform`), and study B1 has 3 of 24 evaluations. Items marked
> ⚠️ below rest on partial data. The full gap list is in
> [Completeness](#completeness-and-flags).

## Contents

- [Method](#method)
- [Ablation 1: conditioning channels](#ablation-1-conditioning-channels)
- [Ablation 2: cue budget](#ablation-2-cue-budget)
- [Ablation 3: training schedule](#ablation-3-training-schedule) ⚠️
- [Study A: available history](#study-a-available-history)
- [Study B1: denoising steps](#study-b1-denoising-steps) ⚠️
- [Completeness and flags](#completeness-and-flags)
- [Next steps](#next-steps)

## Method

- **Metrics.** N1-MERT, Recall@5, M2M-MERT and Coverage@5 come from each
  `*-mert.json`. Cue F1 and the prediction unique ratio come from the
  evaluation JSON (`m2m_cue_f1`, `m2m_unique_ratio`). Seconds per history come
  from `evaluation.timing`. Recall@5 is shown in %.
- **Aggregation.** Values are the mean over the available training seeds, with
  the seed range in brackets.
- **Significance.** Δ Recall@5 is a paired, history-level bootstrap (10,000
  samples, seed 42) against the reference variant. It was recomputed from the
  stored predictions: per-history Recall@5 is the multiset overlap of predicted
  and target item IDs divided by 5, which reproduces the stored `m2m_recall`
  exactly (MPD Full seed 1: 0.01849, Music4All Full seed 1: 0.00928). Histories
  from all paired seeds are pooled into one bootstrap. **Bold** marks a 95% CI
  that excludes zero. MERT metrics could not be bootstrapped because the
  catalog embeddings are not in the bundle.
- **Test sets.** MPD has 941 test histories, so one hit moves Recall@5 by about
  0.02 points and seed spreads are wide. Music4All has 19,771 test histories.
- **Validity checks.** Every final result was created after its run's
  `runs.csv` finish time, and its checkpoint SHA-256 matches `runs.csv`, so
  **no result is STALE**. Exception: the four MPD seed-1 runs of ablation 1
  (`mpd-<condition>`, unsuffixed) predate `runs.csv` and cannot be checked from
  this bundle (see [flags](#completeness-and-flags)).
- **Unchanged metrics.** N1-MERT and M2M-MERT move by at most about 0.002
  across every variant of every ablation, so Recall@5 and Coverage@5 carry the
  comparisons below.

## Ablation 1: conditioning channels

**Both channels carry signal, and their roles differ by dataset.** Removing
history cues (`latent_only`) significantly lowers Recall@5 on both datasets.
Removing latent references (`cue_only`) costs 40% on Music4All but is not
significant on MPD. Shuffling cues to another history costs 12% on Music4All,
so cue *semantics* matter there. On MPD, shuffled cues do as well as real ones.

Complete: 4 conditions × 3 seeds × 2 datasets = 24 of 24.

| Dataset | Condition | Recall@5 % | Δ R@5 vs Full (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | Cue F1 |
|---|---|---|---|---|---|---|---|
| MPD | Full | 1.877 [1.849–1.934] | — | 0.8730 | 0.8911 | 0.374 | 0.281 |
| MPD | latent_only | 1.566 [1.424–1.637] | **−16.6%** (−0.581 to −0.043 pp) | 0.8724 | 0.8909 | 0.374 | 0.278 |
| MPD | cue_only | 1.700 [1.573–1.785] | −9.4% (−0.432 to +0.085 pp) | 0.8715 | 0.8908 | 0.372 | 0.275 |
| MPD | shuffled_cue | 1.863 [1.785–1.913] | −0.8% (−0.269 to +0.241 pp) | 0.8716 | 0.8912 | 0.376 | 0.279 |
| Music4All | Full | 0.957 [0.928–0.991] | — | 0.8649 | 0.8803 | 0.715 | 0.445 |
| Music4All | latent_only | 0.787 [0.760–0.824] | **−17.8%** (−0.208 to −0.131 pp) | 0.8644 | 0.8800 | 0.718 | 0.439 |
| Music4All | cue_only | 0.578 [0.544–0.610] | **−39.6%** (−0.418 to −0.340 pp) | 0.8635 | 0.8791 | 0.721 | 0.441 |
| Music4All | shuffled_cue | 0.841 [0.807–0.861] | **−12.1%** (−0.154 to −0.078 pp) | 0.8645 | 0.8799 | 0.718 | 0.440 |

| Comparison | MPD | Music4All |
|---|---|---|
| Do cue semantics matter? (Full vs shuffled) | No detectable effect | Yes, −12% |
| Do cues add over latents? (Full vs latent_only) | Yes, −17% without them | Yes, −18% without them |
| Do latents add over cues? (Full vs cue_only) | Not significant (−9%) | Yes, −40% without them |
| Which channel carries more? (cue_only vs latent_only) | No significant difference (+8.6%, CI −0.120 to +0.404 pp) | Latents (**−26.5%** for cue_only) |

On MPD, `latent_only` loses Recall@5 but `shuffled_cue` does not. A plausible
reading is that MPD benefits from cue tokens being present (extra conditioning
positions), not from what they say. Music4All, with real listening sequences,
uses both the latent references and the specific cues.

The unofficial Full-on-shuffled-cues diagnostic
(`on-shuffled_cue-last-steps256-evalseed1.json`) is not in the bundle.

## Ablation 2: cue budget

**Fewer cues give better item retrieval on both datasets.** The 0-cue model has
the highest Recall@5 and Coverage@5. Recall@5 falls steadily from 0 to 16 cues,
and N1-MERT and M2M-MERT stay flat. Cue ranking (ranked-8 vs random-8) makes no
significant difference to retrieval. On these metrics, 8 cues are not a good
balance point. Because of the loss-balance confound (below), this is not yet
proof that cue *information* hurts.

### MPD (complete: 3 seeds per budget)

| Budget | Tokens / seq | Recall@5 % | Δ R@5 vs ranked-8 (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | s / history (median) |
|---|---|---|---|---|---|---|---|
| 0 | 102 | 2.338 [2.232–2.444] | **+24.5%** (+0.135 to +0.779 pp) | 0.8735 | 0.8925 | 0.406 | 0.088 |
| 4 | 182 | 2.246 [2.019–2.465] | **+19.6%** (+0.057 to +0.673 pp) | 0.8737 | 0.8913 | 0.383 | 0.096 |
| 8 ranked (Full, ablation 1) | 262 | 1.877 [1.849–1.934] | — | 0.8730 | 0.8911 | 0.374 | — ¹ |
| 8 random | 262 | 1.800 [1.679–1.913] | −4.2% (−0.361 to +0.205 pp) | 0.8727 | 0.8916 | 0.370 | 0.107 |
| 16 | 422 | 1.382 [1.169–1.552] | **−26.4%** (−0.772 to −0.220 pp) | 0.8712 | 0.8909 | 0.370 | 0.119 |

### Music4All ⚠️ (seed 1 only; seeds 2-3 missing)

| Budget | Tokens / seq | Recall@5 % | Δ R@5 vs ranked-8 seed 1 (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | s / history |
|---|---|---|---|---|---|---|---|
| 0 | 102 | 1.360 | **+46.6%** (+0.342 to +0.522 pp) | 0.8654 | 0.8806 | 0.761 | 0.086 |
| 4 | 182 | 0.944 | +1.7% (−0.066 to +0.097 pp) | 0.8648 | 0.8799 | 0.739 | 0.089 |
| 8 ranked (Full s1) | 262 | 0.928 | — | 0.8647 | 0.8802 | 0.715 | — ¹ |
| 8 random | 262 | 0.970 | +4.6% (−0.032 to +0.117 pp) | 0.8651 | 0.8803 | 0.718 | 0.099 |
| 16 | 422 | 0.808 | **−12.9%** (−0.198 to −0.040 pp) | 0.8649 | 0.8800 | 0.702 | 0.115 |

¹ The ranked-8 results were evaluated before timing existed. The identically
configured warm-up models of ablation 3 took 0.098–0.123 s (MPD) and 0.103 s
(Music4All) per history.

Cue F1 is not comparable across budgets (see the spec). Within ranked-8 vs
random-8 it is 0.281 vs 0.215 (MPD) and 0.444 vs 0.322 (Music4All). Ranked cues
are much easier to predict, but that does not translate into better item
retrieval.

**Cost.** Going from 102 to 422 tokens per sequence raises generation time by
only about 35% (MPD 0.088 → 0.119 s). At batch 32 on an A40, sequence length
is not the main cost. The MPD seed-1 timings for 0 and 4 cues (0.148 and
0.165 s) are outliers, probably from GPU contention, so medians are shown. All
timings are indicative only.

**Interpretation and caveat.** Taken with ablation 1, cues help as
*conditioning* (`latent_only` < Full), but predicting them competes with
predicting items. The spec's loss-balance caveat applies directly: the cue
share of the per-song loss weight rises from 0% (0 cues) to about 44%, 62% and
76% (4, 8 and 16 cues), which tracks the Recall@5 decline. A control with the
cue loss weight scaled to keep the cue share constant, or a model conditioned
on cues without predicting them, would separate the two explanations.

## Ablation 3: training schedule

⚠️ **Partial: 8 of 18 final results.** MPD has seeds 1-2 for each schedule;
Music4All has warm-up and fixed for seed 1 only, and no `uniform` results.

**No evidence that the warm-up helps the final score.** Neither fixed nor
uniform differs significantly from warm-up on either dataset. The retrained
warm-up arm reproduces the ablation 1 Full models within noise, which passes
the reproducibility check.

| Dataset | Schedule | Seeds | Recall@5 % | Δ R@5 vs warm-up (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | Cue F1 |
|---|---|---|---|---|---|---|---|---|
| MPD | warm-up | 1, 2 | 2.094 [1.807–2.380] | — | 0.8723 | 0.8912 | 0.372 | 0.282 |
| MPD | fixed | 1, 2 | 1.955 [1.955–1.955] | −6.6% (−0.468 to +0.191 pp) | 0.8729 | 0.8917 | 0.377 | 0.282 |
| MPD | uniform | 1, 2 | 1.881 [1.807–1.955] | −10.2% (−0.553 to +0.138 pp) | 0.8722 | 0.8908 | 0.370 | 0.279 |
| Music4All | warm-up | 1 | 0.925 | — | 0.8648 | 0.8803 | 0.721 | 0.445 |
| Music4All | fixed | 1 | 0.919 | −0.7% (−0.075 to +0.064 pp) | 0.8655 | 0.8805 | 0.718 | 0.447 |
| Music4All | uniform | — | *missing* | — | — | — | — | — |

Reproducibility (retrained warm-up vs ablation 1 Full, same seeds): MPD +10.7%
(CI −0.128 to +0.521 pp, 2 seeds); Music4All −0.3% (CI −0.063 to +0.058 pp,
seed 1).

### Training curves (unofficial: raw weights, 64 steps)

MPD shows the mean of seeds 1-2 over all 941 histories. Music4All shows seed 1
on the 2,000-history subset.

| Step | MPD warm-up R@5 % / cue F1 | MPD fixed | MPD uniform | M4A warm-up | M4A fixed |
|---|---|---|---|---|---|
| 1,000 | 1.78 / 0.270 | 1.52 / 0.275 | 1.56 / 0.271 | 0.56 / **0.407** | 0.68 / **0.435** |
| 2,000 | 1.86 / 0.267 | 1.88 / 0.271 | 1.63 / 0.268 | 0.76 / 0.426 | 0.78 / 0.445 |
| 3,000 | 1.75 / 0.270 | 1.64 / 0.268 | 1.67 / 0.275 | 0.71 / 0.430 | 0.67 / 0.445 |
| 4,000 | 1.55 / 0.272 | 1.46 / 0.273 | 1.39 / 0.272 | 0.71 / 0.433 | 0.74 / 0.430 |
| 5,000 | 1.45 / 0.269 | 1.81 / 0.276 | 1.78 / 0.272 | 0.70 / 0.434 | 0.90 / 0.438 |
| 7,500 | 1.77 / 0.273 | 1.52 / 0.274 | 1.56 / 0.276 | 0.75 / 0.442 | 0.80 / 0.441 |
| 10,000 | 1.66 / 0.273 | 1.57 / 0.275 | 1.67 / 0.269 | 0.81 / 0.436 | 0.94 / 0.442 |
| 15,000 | 1.82 / 0.275 | 1.64 / 0.268 | 1.52 / 0.276 | 0.72 / 0.451 | 0.96 / 0.458 |
| 20,000 | 1.94 / 0.283 | 1.91 / 0.286 | 1.56 / 0.277 | 0.90 / 0.435 | 0.75 / 0.435 |
| 20,000 @ 256 steps | 1.86 / 0.287 | 1.78 / 0.286 | 1.82 / 0.277 | 0.75 / 0.432 | 0.74 / 0.432 |

- **Recall@5 is flat from step 1,000**, with step-to-step noise of about ±0.3
  pp (MPD) and ±0.1 pp (Music4All). The warm start already gives most of the
  final retrieval quality, so "steps to 90% of final Recall@5" is reached at
  the first curve point for every arm and does not separate the schedules.
- **Cue F1 shows the training-time effect.** On Music4All, warm-up starts
  lower (0.407 vs 0.435 at step 1,000, while its cue weight is still 0.1) and
  catches up by about step 5,000-7,500, when the ramp ends. That matches the
  spec's "warm-up only delays cue learning" pattern, but it rests on one seed.
- **64 vs 256 steps check.** At step 20,000, the schedule ranking is not
  stable between 64 and 256 steps (MPD: warm-up > fixed > uniform at 64 steps,
  warm-up > uniform > fixed at 256). All of these gaps are within noise.
- **Not available:** gradient-norm and clipping statistics, and the unweighted
  `layer_nll/*` losses. The TensorBoard logs and the
  `export_training_curves.py` CSVs are not in the bundle, so the stability
  question (spikier gradients or an early RVQ-loss rise under `fixed`) cannot
  be answered yet.

## Study A: available history

**More history gives better predictions on both datasets.** MPD is still
improving at 15 references; Music4All is nearly saturated by 10. Complete: 3
k values × 3 seeds × 2 datasets = 18 of 18, plus the reused k = 15 results.

| Dataset | Kept refs k | Recall@5 % | Δ R@5 vs k = 15 (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | Cue F1 |
|---|---|---|---|---|---|---|---|
| MPD | 1 | 0.213 [0.085–0.319] | **−88.7%** (−1.906 to −1.431 pp) | 0.8684 | 0.8848 | 0.195 | 0.166 |
| MPD | 5 | 0.878 [0.404–1.126] | **−53.2%** (−1.247 to −0.751 pp) | 0.8707 | 0.8880 | 0.314 | 0.219 |
| MPD | 10 | 1.332 [0.935–1.552] | **−29.1%** (−0.786 to −0.298 pp) | 0.8720 | 0.8904 | 0.360 | 0.255 |
| MPD | 15 | 1.877 [1.849–1.934] | — | 0.8730 | 0.8911 | 0.374 | 0.281 |
| Music4All | 1 | 0.244 [0.113–0.376] | **−74.5%** (−0.754 to −0.672 pp) | 0.8604 | 0.8751 | 0.594 | 0.364 |
| Music4All | 5 | 0.673 [0.410–0.829] | **−29.7%** (−0.322 to −0.245 pp) | 0.8631 | 0.8786 | 0.696 | 0.416 |
| Music4All | 10 | 0.884 [0.765–0.950] | **−7.6%** (−0.106 to −0.039 pp) | 0.8647 | 0.8799 | 0.716 | 0.435 |
| Music4All | 15 | 0.957 [0.928–0.991] | — | 0.8649 | 0.8803 | 0.715 | 0.445 |

- Every metric in the table rises monotonically with k on both datasets. The one exception
  is Music4All Coverage@5, which is flat from k = 10.
- The curated MPD playlists need more history than the Music4All listening
  sequences. MPD gains 29% from k = 10 to 15; Music4All gains 8%.
- **Caveat (from the spec), reinforced by the data.** The seed range at low k
  is far wider than at k = 15. For example, Music4All k = 1 ranges from 0.113%
  to 0.376%, against 0.928-0.991% at k = 15. The three Full models react very
  differently to blanked references they never saw in training. The low-k
  numbers therefore mix "less history" with an out-of-distribution input, and
  the k = 1 and k = 5 drops are upper bounds on the pure history effect.
- An extra file,
  `ablation-history-cond/mpd-full/results/last-steps16-evalseed1-hist5-n100s0.json`
  (16 steps, 100 histories, no MERT), is a smoke test. It is excluded.

## Study B1: denoising steps

⚠️ **Partial: 3 of 24 evaluations.** Only Music4All Full seed 1 at 16, 32 and
64 steps exists.

**On that one model, quality is flat from 16 to 256 steps, and generation time
scales linearly with steps.** None of the differences from 256 steps is
significant.

| Steps | Recall@5 % | Δ R@5 vs 256 (95% CI) | N1-MERT | M2M-MERT | Coverage@5 | Unique ratio | s / history |
|---|---|---|---|---|---|---|---|
| 16 | 0.952 | +2.6% (−0.055 to +0.104 pp) | 0.8648 | 0.8802 | 0.724 | 0.989 | 0.0088 |
| 32 | 0.893 | −3.7% (−0.114 to +0.045 pp) | 0.8647 | 0.8803 | 0.718 | 0.988 | 0.0176 |
| 64 | 0.939 | +1.2% (−0.069 to +0.091 pp) | 0.8649 | 0.8803 | 0.722 | 0.989 | 0.0335 |
| 128 | *missing* | — | — | — | — | — | — |
| 256 (official) | 0.928 | — | 0.8647 | 0.8802 | 0.715 | 0.988 | *missing* ² |

² The official result predates timing, and the planned
`last-steps256-evalseed1-timing.json` re-run is not in the bundle. Extrapolated
linearly, 256 steps take about 0.14 s. Other 256-step evaluations of the same
model and configuration measured 0.10-0.13 s (study A and ablation 3), which
fits that estimate.

If this holds across seeds and on MPD, 16 steps would cut backbone generation
cost by about 12× with no measurable loss. Diversity holds too: Coverage@5 and
the unique ratio do not drop at low step counts. This rests on a single model;
treat it as a lead, not a result.

## Completeness and flags

| Experiment | Expected | Present | Missing | Training status of missing runs (from `runs.csv`) |
|---|---|---|---|---|
| Ablation 1 | 24 | 24 | — | — |
| Ablation 2, MPD | 12 | 12 | — | — |
| Ablation 2, Music4All | 12 | 4 | cues0, cues4, cues8random, cues16 × seeds 2-3 | `m4a-cues0-seed2` **finished** (2026-10-06 08:46) but not evaluated; the other 7 have no `runs.csv` row (still training or not started) |
| Ablation 3, final | 18 | 8 | MPD warm-up/fixed/uniform seed 3; Music4All warm-up s2-3, fixed s2-3, uniform s1-3 | **All 10 finished training** (`runs.csv`) but not evaluated |
| Ablation 3, curves | 18 runs | 8 runs | Same 10 runs | Same |
| Ablation 3, grad-norm / TensorBoard export | 18 | 0 | All | Not in the bundle |
| Study A | 18 | 18 | — | — |
| Study B1 | 24 + timing re-runs | 3 | Music4All s1 @128; Music4All s2-3 (all steps); MPD s1-3 (all steps); all six 256-step timing re-runs | Models are finished; evaluations not run |

Other flags:

- **MPD seed-1 runs of ablation 1 are unverifiable here.** The four unsuffixed
  folders (`mpd-full`, `mpd-latent_only`, `mpd-cue_only`, `mpd-shuffled_cue`)
  predate `runs.csv`. The runners treat them as finished when they contain
  `step-20000.ckpt`, but checkpoints are not in the bundle. Their results are
  dated 2026-09-30, and they also serve as ranked-8 seed 1 in ablation 2 and
  as the k = 15 / 256-step baselines of the model studies.
- **Timing is indicative only.** Several MPD evaluations ran close together on
  2026-10-06 06:43-07:00, and two seed-1 timings are clear outliers.
- **The bootstrap pools seeds.** The paired bootstrap pools histories across
  seeds. A per-seed bootstrap with a seed-level summary, as the spec suggests,
  would give somewhat wider intervals for the MPD comparisons.
- **Smoke-test file.** `mpd-full/.../last-steps16-evalseed1-hist5-n100s0.json`
  is excluded (see study A).

## Next steps

1. Evaluate the 11 runs that have finished training but have no results: 10
   in ablation 3 and `m4a-cues0-seed2`:
   `eval_training_schedule.sh mpd|m4a --mode both` and
   `eval_cue_budget.sh m4a`.
2. Run study B1 for the remaining models, including 128 steps and the 256-step
   timing re-runs, on an idle GPU.
3. Export the TensorBoard scalars (`export_training_curves.py`) for the
   gradient-norm and per-layer-loss analysis of ablation 3.
4. Given the ablation 2 result, consider a loss-balance control for the cue
   budget, such as a constant cue share of the loss weight. A Full model with
   history cues but no cue-prediction loss would show whether cues should be
   conditioning-only.
