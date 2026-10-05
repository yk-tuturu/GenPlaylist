# GenPlaylist model studies

The model studies probe the trained GenPlaylist backbone beyond the ablations
in [`ABLATIONS.md`](ABLATIONS.md). They share its protocol, datasets, run
folders, and `scripts/ablation/common.sh`.

| Study | Question | Needs training | Status |
|---|---|---|---|
| [A. Available history](#study-a-available-history) | How does performance change with 1, 5, 10, or 15 reference tracks? | no | Implemented |
| [B1. Denoising steps](#study-b1-denoising-steps) | How do quality, diversity, and inference time trade off over the number of denoising steps? | no | Implemented |
| [B2. Audio generator settings](#planned-studies) | The same trade-off for ACE-Step inference steps and guidance scale, on generated audio | no | Planned |
| [C. Initialization](#planned-studies) | Pretrained backbone vs training from scratch vs a frozen backbone, at a matched budget | yes | Planned |

## Models

Studies A and B1 evaluate the **Full models of the history-conditioning
ablation** (8 ranked cues, `history_condition=full`, warm-up loss schedule),
3 seeds per dataset:

| Dataset | Model folders (under `src/03_backbone_recommender/outputs/`) | Prepared data |
|---|---|---|
| MPD | `ablation-history-cond/mpd-full` (seed 1), `mpd-full-seed2`, `mpd-full-seed3` | `data/processed/ablation-mpd-8cue-full` |
| Music4All v3-u500 | `ablation-history-cond/m4a-full-seed1`, `-seed2`, `-seed3` | `data/processed/ablation-m4a-v3-u500-8cue-full` |

The runners find these through `cue_budget_variant 8` in `common.sh`, which
also handles the unsuffixed MPD seed-1 folder.

## Study A: available history

Does more listening history give a better estimate of the user's preference?
The same trained model is evaluated with only the most recent k of the 15
reference tracks, for k = 1, 5, 10, 15.

### Masking

- The **most recent k** references (the ones just before the predicted
  songs) are kept; the 15 - k oldest are blanked.
- A blanked reference keeps its BOI token; its RVQ and conflict tokens become
  the RVQ-null token (0) and its cue tokens become the cue-null token (`<unk>`,
  token 845). These are the null tokens of the history-conditioning ablation.
- The 15-slot layout is unchanged, so the five predicted songs stay at the
  positions the model was trained on. Shortening the context instead would move
  them, and k = 1 would be rejected (joint completion needs at least two
  references).
- Masking happens at evaluation time in `main.py`, on each test batch before
  generation. The prepared data is not modified, and no fingerprinted
  preparation code changed.
- `mu_c` and `context_emb` still describe all 15 references, but they are not
  read while CFG and structure conditioning are off (the default); history
  masking refuses to run if either is on.

### Settings

Every k uses the official sampling settings: EMA weights, 256 denoising
steps, sampling seed 1, all test histories. k = 15 is the official result,
`last-steps256-evalseed1.json`, reused when it exists. Other k write
`last-steps256-evalseed1-hist<k>.json`, with
`evaluation.history_length = {kept_references, masked_references, kept:
"most_recent"}` and `official_protocol: false`. Each also gets MERT metrics.

### Caveat

The Full model was always trained with 15 real references and never saw
blanked ones. Low-k results therefore combine "less history" with an input
pattern the model has not seen. A model trained with random history dropout
would separate the two; it is not part of this implementation and would need
extra training runs.

### Running it

```bash
bash scripts/model_study/eval_history_length.sh mpd --gpu 4 --seeds "1 2 3"
```

```bash
bash scripts/model_study/eval_history_length.sh m4a --gpu 4 --seeds "1 2 3"
```

`--lengths "1 5 10 15"` sets the k values, `--no-mert` skips MERT, and
`--dry-run` prints the plan. The summary lists k, seed, N1-MERT, Recall@5,
M2M-MERT, Coverage@5, and cue F1. That is 3 new k values x 3 seeds x 2
datasets = 18 evaluations, no training.

### Reading the results

Plot each metric against k (mean and seed range). A rising curve supports the
claim that multiple listening events give a better preference estimate; a
curve that flattens early shows how much history is enough. Compare
datasets: Music4All histories are real listening sequences, MPD histories are
curated playlists.

## Study B1: denoising steps

How do personalization, diversity, and inference time trade off over the
number of reverse-diffusion (denoising) steps?

### Settings

- Step counts 16, 32, 64, 128, and 256 (the official setting).
- Everything else is official: EMA weights, sampling seed 1, all test
  histories. Step counts other than 256 need the protocol override and are
  marked `official_protocol: false`; their files are
  `last-steps<S>-evalseed1.json`.
- Reported per step count: N1-MERT, Recall@5, M2M-MERT (personalization),
  Coverage@5 and the prediction unique ratio (diversity), and generation
  seconds per history from `evaluation.timing` (inference cost).
- **256 steps:** the official result is reused when it records
  `evaluation.timing`. Results evaluated before timing existed (the MPD
  seed-1 runs) are timed again into `last-steps256-evalseed1-timing.json`, and
  the official file is never overwritten. MERT metrics depend only on the
  predictions, so the timing re-run reuses the official MERT result.

### Timing

Time per history is comparable only between runs on the same GPU type (all
server GPUs are A40s) with the same batch size (32) and **no competing jobs**.
The runner warns when the chosen GPU is above 10% utilization; pick an idle
GPU, or report timing as indicative. Generation time covers only the sampling
calls, synchronized on the GPU.

### Classifier-free guidance

The backbone's CFG option (`sampling.cfg_enabled`) only drops a mean-pooled
CLHE vector of the references, not the reference tokens the model is
conditioned on, and the models were not trained with CFG. A CFG-scale sweep
on the backbone would therefore not test guidance in any meaningful sense, so
this study varies denoising steps only. Guidance is studied on the audio
generator in study B2.

### Running it

Use an idle GPU:

```bash
bash scripts/model_study/eval_sampling_steps.sh mpd --gpu 4 --seeds "1 2 3"
```

```bash
bash scripts/model_study/eval_sampling_steps.sh m4a --gpu 4 --seeds "1 2 3"
```

`--steps "16 32 64 128 256"` sets the step counts. The summary lists steps,
seed, the four MERT metrics, unique ratio, seconds per history, and device.
That is 4 new step counts x 3 seeds x 2 datasets = 24 evaluations, plus a
256-step timing run for models evaluated before timing existed.

### Reading the results

Plot personalization and diversity against seconds per history, one point per
step count (mean over seeds). The knee of the curve is the cheapest setting
that keeps most of the 256-step quality.

## Planned studies

- **B2. Audio generator settings.** Render the official GenPlaylist plans with
  different ACE-Step inference steps (for example 15, 30, 60, 100; 60 today)
  and guidance scales (for example 3, 7.5, 15; 15 today) on a fixed subset of
  histories, and report FAD, MERT History Fit, CLAP-A, cross-history
  diversity, and render time. Needs `--guidance-scale` (now hard-coded to
  15.0) and per-song timing in `scripts/run_end_to_end_synthesis.py`.
- **C. Initialization.** Pretrained (the Full models) vs scratch
  (`GENPLAYLIST_TRAIN_MODE=scratch`) vs a frozen backbone (transformer blocks
  frozen; embeddings and output layer trained), with the same steps, batch,
  data, and seeds. Needs a `training.freeze_backbone` option. Note that
  "scratch" still initializes the RVQ token embeddings from the codebook
  (`models/dit.py`); only the transformer and the other embeddings start
  random.

## Code changes

| File | Change |
|---|---|
| `src/03_backbone_recommender/evaluation_protocol.py` | New `history_mask_positions(reference_items, tokens_per_item, semantic_tokens, keep)`: the semantic and cue positions of the oldest `reference_items - keep` reference blocks in a `[BOS, references..., EOS]` context. BOI positions are never returned. |
| `src/03_backbone_recommender/main.py` | `rec_eval` reads `eval.history_length`. For k < 15 it blanks those positions in every test batch with `SEMANTIC_NULL_TOKEN` and `CUE_NULL_TOKEN`, checks the context length, refuses CFG and structure conditioning, records `evaluation.history_length`, and sets `official_protocol: false`. k = 15 or null changes nothing. |
| `src/03_backbone_recommender/configs/config.yaml` | `eval.history_length: null`. |
| `src/03_backbone_recommender/scripts/eval_spotify.sh` | New `GENPLAYLIST_EVAL_HISTORY_LENGTH` (1-15). For k < 15 it passes `eval.history_length` and adds `-hist<k>` to the result file name; k = 15 adds nothing. It needs no protocol override and combines with the curve options. The test-subset arguments are now appended to the curve arguments instead of replacing them. |
| `src/03_backbone_recommender/test_evaluation_protocol.py` | +2 tests: only the oldest blocks' RVQ, conflict, and cue positions are blanked, never BOI or the kept blocks; k = 15 blanks nothing, k = 1 blanks 14 blocks, a 0-cue layout has no cue positions, and k outside 1-15 is rejected. |
| `scripts/model_study/eval_history_length.sh` | Runner for study A. |
| `scripts/model_study/eval_sampling_steps.sh` | Runner for study B1. |

No fingerprinted preparation code changed, so every prepared folder stays
valid. Training is untouched.

### Compatibility with the ablations

The committed and the new `eval_spotify.sh` were run side by side with a
stand-in `python` that records its arguments, for every earlier usage:
official named evaluations, the mismatch diagnostic, smoke overrides, the
dated fallback, 0-cue evaluations, ablation 3 curve points with and without a
test subset, and the raw-weights-without-override error. All produced
byte-identical arguments and output, as did `train_spotify.sh` (unchanged).
With `eval.history_length` unset, `rec_eval` takes no new code path, so all
ablation results are produced exactly as before.

Both runners were also run end to end against the real `eval_spotify.sh`
(with only `python` replaced), covering the unsuffixed MPD seed-1 folder, an
existing official result, a model with no results, an official result
without timing, re-runs, and invalid input.
