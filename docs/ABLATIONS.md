# GenPlaylist ablations

This document covers the required ablations of the GenPlaylist backbone: what
each one tests, how it is implemented, and how to run it on MPD and
Music4All-Onion. All of them share one frozen protocol, one run-folder layout,
and one set of runner scripts, described first.

| # | Ablation | Question | Status |
|---|---|---|---|
| 1 | [Conditioning channels](#ablation-1-conditioning-channels) | Which part of the history drives predictions: latent references, cues, or just extra cue tokens? | Implemented; results in [`ABLATION_HISTORY_COND_RESULTS.md`](ABLATION_HISTORY_COND_RESULTS.md) |
| 2 | [Cue budget](#ablation-2-cue-budget) | Does cue ranking matter, and is 8 cues a good balance of information and sequence length? | Implemented |
| 3 | [Training schedule](#ablation-3-training-schedule-planned) | Does warming up the cue loss help over fixed loss weights from the start? | Planned |

Contents:

- [Shared setup](#shared-setup)
- [Ablation 1: conditioning channels](#ablation-1-conditioning-channels)
- [Ablation 2: cue budget](#ablation-2-cue-budget)
- [Ablation 3: training schedule (planned)](#ablation-3-training-schedule-planned)
- [Repository changes](#repository-changes)

## Shared setup

### Protocol

Every model in every ablation follows the frozen 15->5 protocol in
[`WP_C_TRAIN_EVAL_PROTOCOL.md`](WP_C_TRAIN_EVAL_PROTOCOL.md):

- the same splits, catalog, and test histories per dataset
- warm start from `checkpoints/pretrained/ddbc/spotify30.ckpt`
- 20,000 optimization steps at global batch 512, and the final EMA checkpoint
  (there is no validation split, so no checkpoint is selected on test data)
- evaluation with 256 reverse-diffusion steps, sampling seed 1, EMA weights,
  full-catalog retrieval, and 5x5 Hungarian matching
- three training seeds (1, 2, 3) per variant

Only the property under study changes between variants of one ablation.

### Datasets

| | MPD | Music4All-Onion v3-u500 |
|---|---|---|
| Catalog | 5,119 songs | 10,950 songs |
| Training windows | 57,331 | 1,227,307 |
| Test histories | 941 | 19,771 |
| `GENPLAYLIST_DATA_CONFIG` | `spotify` | `music4all` |
| Data (`--data-dir`) | `data/dataset` | `data/dataset-music4all-onion-v3-u500` |
| Song tokens (`--artifact-dir`) | `data/dataset` | `data/artifacts-music4all-onion-v3-u500` |
| Cues (`--cue-dir`) | `src/02_creative_cues/outputs/production/latest` | `data/processed/cues-music4all-onion-v3-u500/latest` |
| MERT catalog embeddings | `data/processed/mert-v1-95m-catalog-v1` | `data/processed/mert-v1-95m-music4all-u500-audio-covered-v1` |
| Prepared-folder prefix | `data/processed/ablation-mpd` | `data/processed/ablation-m4a-v3-u500` |

Music4All v3-u500 is tuteng's expanded catalog, built at commit `613bdf5`. Its
source folders are on the server under `/home/wjzhang/tt_workspace/model/GenPlaylist`
and are copied byte-for-byte into the repository's gitignored `data/`:

| Folder | Key file SHA-256 |
|---|---|
| `data/dataset-music4all-onion-v3-u500` | `catalog_metadata.json` `8f8abf7f…` |
| `data/artifacts-music4all-onion-v3-u500` | `item_id_to_row.json` `cf86c136…` |
| `data/processed/cues-music4all-onion-v3-u500/latest` | `item2cues.json` `7e8c5642…` |

Its cue vocabulary is byte-identical to MPD's (`cue_vocab.json` `cd229492…`),
so cue IDs mean the same on both datasets. Both MERT folders use
`m-a-p/MERT-v1-95M` at revision `12af15fe…`; each covers its whole catalog.

Always pass `--cue-dir` when preparing Music4All data; without it, preparation
silently uses the MPD cues. The runner scripts set every path themselves (see
[`scripts/ablation/common.sh`](../scripts/ablation/common.sh)).

### Run folders and `runs.csv`

`GENPLAYLIST_OUTPUT_NAME` gives a run a fixed folder instead of Hydra's dated
`outputs/<data>/<date>/<time>/`:

```
src/03_backbone_recommender/outputs/
├── runs.csv                                one row per finished named training run
├── ablation-history-cond/m4a-cue_only-seed2/
│   ├── .hydra/
│   ├── checkpoints/{last,step-500,...}.ckpt
│   ├── results/last-steps256-evalseed1.json
│   ├── results/last-steps256-evalseed1-mert.json
│   └── eval-runs/...                       Hydra folders of evaluation runs
└── ablation-cue-budget/mpd-cues16-seed1/...
```

- The name is used exactly as given (letters, digits, `.`, `_`, `-`, and `/`
  between names). Put the variant and seed into it; the runner scripts do.
- Training refuses to start in a folder that already has `last.ckpt`.
  `GENPLAYLIST_TRAIN_MODE=resume` continues a crashed run in the same folder,
  with the same variables as the original run.
- `runs.csv` records finish time, output name, dataset, history condition,
  active cues, loss variant, seed, steps, train mode, git commit, dirty flag,
  prepared-manifest SHA-256, checkpoint path, and checkpoint SHA-256. Only
  finished runs get a row; a resumed run adds another row.
- Evaluation with the same `GENPLAYLIST_OUTPUT_NAME` evaluates that folder's
  `last.ckpt` (`GENPLAYLIST_EVAL_CKPT_FILE=step-10000.ckpt` picks an
  intermediate one) and writes to its `results/` folder. It refuses to
  overwrite an existing result.
- `GENPLAYLIST_OUTPUT_ROOT` moves the whole tree.
- Without `GENPLAYLIST_OUTPUT_NAME`, training and evaluation behave as before.

### Runner scripts (`scripts/ablation/`)

The runners wrap `train_spotify.sh` and `eval_spotify.sh` with the per-dataset
paths. They find the repository from their own location, refuse to start when
`GENPLAYLIST_*` variables are already set or required data is missing, skip
finished work, and continue with the next run when one fails. Run them inside
tmux; each prints its usage with `--help`. Commands in this document run from
the repository root.

| Script | Ablation | Purpose |
|---|---|---|
| `train_history_cond_mpd.sh <gpu> <seed>...` | 1 | Train the four MPD conditioning variants |
| `train_history_cond_m4a.sh <gpu> <seed>...` | 1 | Same for Music4All |
| `eval_history_cond.sh <mpd\|m4a> --gpu N [--seeds] [--conds] [--no-mert] [--dry-run]` | 1 | Evaluation + MERT, with a summary table |
| `prepare_cue_budget.sh <mpd\|m4a> [--budgets]` | 2 | Build the random-8 cue table, prepare and validate each budget |
| `train_cue_budget.sh <mpd\|m4a> --gpu N --seeds "..." [--budgets]` | 2 | Train the cue-budget variants; resumes unfinished runs |
| `eval_cue_budget.sh <mpd\|m4a> --gpu N [--seeds] [--budgets] [--no-mert] [--dry-run]` | 2 | Evaluation + MERT, with tokens and seconds per history |
| `build_random_cue_table.py` | 2 | Derive the random-subset cue folder (called by `prepare_cue_budget.sh`) |
| `common.sh` | all | Dataset paths and cue-budget variants (sourced) |

### Evaluation outputs

Each evaluation writes `results/last-steps256-evalseed1.json` with the
metrics, predictions, checkpoint and prepared-data hashes, git commit, and:

- `evaluation.official_protocol`: false for any override or mismatch
- `evaluation.history_condition`: `{test_contexts, checkpoint}`
- `evaluation.active_cue_tokens` and `evaluation.tokens_per_sequence`
- `evaluation.timing`: `generation_seconds`, `generation_seconds_per_history`,
  `loop_seconds`, `eval_batch_size`, `device`. Generation time covers only the
  sampling calls (synchronized on GPU), not data loading or metrics. Compare
  timings only between runs on the same GPU type with no competing jobs.

`scripts/evaluate_mert_proxy.py` turns it into `...-mert.json` with the paper
metrics: N1-MERT, Recall@5, M2M-MERT, Coverage@5, and 95% bootstrap intervals.
The runner scripts do both steps.

To download results, the one-off `collect_results.sh [experiment]` (kept
outside the repository) bundles every model's `results/*.json` and
`runs.csv` into one archive.

## Ablation 1: conditioning channels

Which part of the reference history drives GenPlaylist's predictions: the
latent musical references (RVQ/conflict tokens), the creative cues, or merely
the presence of extra cue tokens?

### Variants

Every variant uses the main 8-cue layout (13 tokens per item, 262-token
sequences). Only the **15 reference items** change. The five target items
always carry their real RVQ, conflict, and cue tokens, so every variant is
trained and scored on the same objective.

| `history_condition` | Reference RVQ + conflict tokens | Reference cue tokens |
|---|---|---|
| `full` | real | real |
| `latent_only` | real | cue-null token |
| `cue_only` | RVQ-null token | real |
| `shuffled_cue` | real | real cues of a donor history |

The existing 0-cue model (DDBC-SFT) has no cues anywhere and is a different
layout; it can be reported as an extra "no cues" row. It is not
`latent_only`.

#### Null tokens

The runtime vocabulary (IDs 0-2893) has no spare IDs, and adding one would
change the embedding size and break the `spotify30.ckpt` warm start.

- **Cue-null** is cue ID 0 (`<unk>`, token 845). The frozen cue table never
  assigns `<unk>` to a real item ([`CUE_VOCAB_FREEZE.md`](CUE_VOCAB_FREEZE.md)),
  so it never occurs in real histories. The validator enforces this.
- **RVQ-null** is token 0, the padding/BOS token. MASK is not used: reference
  contexts containing MASK are rejected, and MASK means "to be denoised".

Training loss is computed only on target payloads, and the `subs`
parameterization copies every unmasked position, so null tokens in the history
never enter the loss or the sampler. For `cue_only`, the per-row CLHE
statistics (`context_emb`, `mu_c`, `sigma_c2`) are also zeroed; they are unused
while CFG and structure conditioning are off, and zeroing them guarantees that
no latent information leaks in.

#### Shuffled-cue donors

`shuffled_cue` keeps the number, position, and format of cue tokens but takes
them from another history. Reference position *p* receives the cues of
reference item *p* of the donor row.

- Donors always come from a **different group**: a different playlist for MPD
  (rows are `<playlist>:joint5:<start>`) and a different user for Music4All
  (rows are `m4a-<split>-<user pseudonym>-...`). MPD training windows overlap
  with stride one, so a naive shuffle would often pick a near-copy of the same
  history.
- The assignment is a seeded permutation, so every row donates exactly once.
  Test donors use `--donor-seed`, train donors `--donor-seed + 1`.
- The mapping is computed once over the full split and stored as
  `history_donors.json` in the prepared folder, with each donor's row ID. Its
  hash is recorded in the manifest.

### Running it

Prepared folders are `<prefix>-8cue-<condition>`. Prepare and validate them
once per dataset (the checkout must be committed and unchanged while this
runs; MPD shown, add the Music4All paths from [Datasets](#datasets) for
Music4All):

```bash
for V in full latent_only cue_only shuffled_cue; do python scripts/prepare_wp_c_data.py --data-dir data/dataset --artifact-dir data/dataset --output-dir $PWD/data/processed/ablation-mpd-8cue-$V --active-cues 8 --history-condition $V --donor-seed 42 && python scripts/validate_wp_c_prepared_data.py --data-dir data/dataset --artifact-dir data/dataset --prepared-dir $PWD/data/processed/ablation-mpd-8cue-$V --active-cues 8 --history-condition $V; done
```

Each Music4All prepared folder is about 2.5 GB.

Train (runs go to `ablation-history-cond/<mpd|m4a>-<condition>-seed<N>`):

```bash
bash scripts/ablation/train_history_cond_mpd.sh 4 1 2 3
```

```bash
bash scripts/ablation/train_history_cond_m4a.sh 4 1 2 3
```

Evaluate, including MERT:

```bash
bash scripts/ablation/eval_history_cond.sh mpd --gpu 4 --seeds "1 2 3"
```

The MPD seed-1 models live in the unsuffixed folders `mpd-<condition>`; the
evaluation script falls back to them.

**Diagnostic (no training):** the Full checkpoint evaluated on shuffled test
cues tests whether Full reads its cues at inference. The result is marked
unofficial and saved as `on-shuffled_cue-last-steps256-evalseed1.json` in the
Full run's folder:

```bash
GENPLAYLIST_OUTPUT_NAME=ablation-history-cond/mpd-full GENPLAYLIST_HISTORY_CONDITION=shuffled_cue GENPLAYLIST_EVAL_ALLOW_HISTORY_MISMATCH=true GENPLAYLIST_PREPARED_DATA_ROOT=$PWD/data/processed/ablation-mpd-8cue-shuffled_cue bash src/03_backbone_recommender/scripts/eval_spotify.sh
```

### Reading the results

| Comparison | Question answered |
|---|---|
| Full vs shuffled-cue | Do cue *semantics* matter, beyond extra conditioning tokens? |
| Full vs latent-only | Do cues add anything over latent references? |
| Full vs cue-only | Do latent references add anything over cues? |
| Latent-only vs cue-only | Which channel carries more preference signal? |

Report the mean and spread over seeds, with paired history-level bootstrap
differences against Full (10,000 samples, seed 42). Results so far are in
[`ABLATION_HISTORY_COND_RESULTS.md`](ABLATION_HISTORY_COND_RESULTS.md).

## Ablation 2: cue budget

Does relevance ranking of cues matter, and do eight cues balance information
against conditioning length? Every song stores 16 ranked cues; WP-C uses the
first `active_cues` of them.

### Variants

| Budget | Cues per song | Tokens per sequence | Model folder | Prepared folder |
|---|---|---|---|---|
| `0` | none | 102 | `ablation-cue-budget/<ds>-cues0-seed<N>` | `<prefix>-cues0` |
| `4` | ranks 1-4 | 182 | `ablation-cue-budget/<ds>-cues4-seed<N>` | `<prefix>-cues4` |
| `8` | ranks 1-8 | 262 | **reused**: `ablation-history-cond/<ds>-full-seed<N>` | `<prefix>-8cue-full` |
| `random8` | 8 random of the 16 stored | 262 | `ablation-cue-budget/<ds>-cues8random-seed<N>` | `<prefix>-cues8random` |
| `16` | all 16 | 422 | `ablation-cue-budget/<ds>-cues16-seed<N>` | `<prefix>-cues16` |

Every variant uses `history_condition=full` and otherwise the shared
protocol. The ranked-8 variant *is* the Full model of ablation 1 (same data,
settings, and seeds), so it is not retrained. New training is
4 variants x 3 seeds x 2 datasets = 24 runs.

### Design decisions

Agreed with the supervisor:

1. **A separate model per budget.** Each model is trained and evaluated with
   its own cue count. Evaluating one model with cues removed would test
   robustness to missing cues, not which budget is best.
2. **Random-8 definition.** Each song gets one random subset of 8 of its 16
   stored cues, drawn once from a seed and the song ID, so it is identical in
   every history, split, and rerun. The 8 keep their original rank order, so
   only *which* cues are used differs from ranked-8. The random table replaces
   the ranked one everywhere: in the history and for the predicted target
   songs.
3. **No loss-balance control.** The standard loss settings are kept for every
   budget; the resulting confound is stated as a caveat (below).

### Random-8 cue table

WP-C always uses the first `active_cues` stored cues, so random-8 needs no
code change. `scripts/ablation/build_random_cue_table.py` writes a new cue
folder in which each song's 16 cues are reordered: the random 8 first, then
the remaining 8, both in original rank order. Preparing it with
`--active-cues 8 --cue-dir <random folder>` gives "8 random cues from the same
stored set".

- The subset comes from `sha256(seed, song ID)` (seed 42), so it does not
  depend on which other songs are in the table.
- `cue_vocab.json` is copied byte-for-byte. `cue_manifest.json` keeps the
  fields the tokenizer checks, points `item2cues_sha256` at the new table, and
  records the derivation under `derived_cue_table` (method, select, seed,
  source hashes).
- The folders are `data/processed/cues-mpd-random8-seed42` and
  `data/processed/cues-m4a-random8-seed42`.
- Because the tokenizer and preparation code are untouched, every prepared
  folder of ablation 1 stays valid.

### Caveats

- **Cue F1 is not comparable across budgets.** Predicting 4 cues is a
  different task from predicting 16, random-8 predicts different labels from
  ranked-8, and 0 cues has no cue F1. Compare budgets on N1-MERT, Recall@5,
  M2M-MERT, and the exact-match metrics; compare cue F1 only within a budget.
- **Loss balance changes with the budget.** Each cue token carries a loss
  weight of 1.0 after warm-up, against 2.0 + 1.5 + 1.0 + 0.5 = 5.0 for the
  RVQ and conflict tokens. The cue share of the per-song loss weight is
  therefore about 44% at 4 cues, 62% at 8, and 76% at 16. A budget effect may
  partly be a loss-balance effect. No control run was made.
- **Cost depends on the GPU.** Sequence length is exact; generation time per
  history (`evaluation.timing`) and training speed (`it/s` in the training
  log) are comparable only between runs on the same GPU type without
  competing jobs.
- **Music4All 16 cues is the heaviest variant:** about 4 GB per prepared
  folder and the slowest training and evaluation.

### Running it

The checkout must be committed and must not change while data is prepared.

1. **Prepare** (builds the random-8 table first; existing folders are only
   re-validated):

   ```bash
   bash scripts/ablation/prepare_cue_budget.sh mpd
   ```

   ```bash
   bash scripts/ablation/prepare_cue_budget.sh m4a
   ```

2. **Smoke test** one variant (500 steps):

   ```bash
   CUDA_VISIBLE_DEVICES=4 GENPLAYLIST_OUTPUT_NAME=smoke/mpd-cues16 GENPLAYLIST_ACTIVE_CUES=16 GENPLAYLIST_PREPARED_DATA_ROOT=$PWD/data/processed/ablation-mpd-cues16 GENPLAYLIST_MAX_STEPS=500 bash src/03_backbone_recommender/scripts/train_spotify.sh
   ```

3. **Train** (budget 8 is skipped; unfinished runs are resumed):

   ```bash
   bash scripts/ablation/train_cue_budget.sh mpd --gpu 4 --seeds "1 2 3"
   ```

   ```bash
   bash scripts/ablation/train_cue_budget.sh m4a --gpu 4 --seeds "1 2 3"
   ```

   `--budgets "16"` trains only some variants, for example to spread them over
   several GPUs.

4. **Evaluate** all five budgets, including ranked-8 from ablation 1:

   ```bash
   bash scripts/ablation/eval_cue_budget.sh mpd --gpu 4 --seeds "1 2 3"
   ```

   The summary lists budget, seed, tokens per sequence, N1-MERT, Recall@5,
   M2M-MERT, Coverage@5, and generation seconds per history. Ranked-8 results
   that were evaluated before timing was added show `-` for seconds per
   history.

### Reading the results

| Comparison | Question answered |
|---|---|
| Ranked-8 vs random-8 | Does relevance ranking matter? (paired bootstrap) |
| 0 vs 4 vs 8 vs 16 | How much does each additional block of cues add? |
| Metric gain vs tokens and seconds per history | Is 8 a reasonable balance of information and conditioning length? |

## Ablation 3: training schedule (planned)

Compare fixed loss weights from the start of training against the cue-weight
warm-up, in which the cue objective is introduced gradually while the
musical-token objective stays active, with the same total steps and
checkpoint rule. Report final performance and training curves, to separate
better optimization stability from a difference in effective training time.

Existing support to build on:

- `GENPLAYLIST_LAYER_LOSS_CURRICULUM=false` trains with uniform weights; the
  default `true` is the warm-up (cue weight 0.1 until step 1,000, then linear
  to 1.0 by step 5,000), and `GENPLAYLIST_CUE_WARMUP_*` set its shape. The run
  name and `runs.csv` record the loss variant.
- Checkpoints are saved every 500 steps, and
  `GENPLAYLIST_EVAL_CKPT_FILE=step-<N>.ckpt` evaluates any of them, for
  training curves. Use curves only to describe training, never to choose a
  checkpoint.
- Training logs per-component unweighted NLL and the effective weights to
  TensorBoard.

## Repository changes

All changes are on branch `waikei-ablation`, on top of `aa65657` (merge of
`waikei-test`, which contains tuteng's last Music4All commit `613bdf5`).

| Commit | Date | Summary |
|---|---|---|
| `3d98434` | 2026-09-29 | Ablation 1: history conditions in the tokenizer, data preparation, validation, evaluation guard, runner flags, tests |
| `5fd6457` | 2026-10-01 | Named run folders (`GENPLAYLIST_OUTPUT_NAME`), resume, and `runs.csv` |
| `f42158a` | 2026-10-04 | Ablation 1 runner scripts in `scripts/ablation/` |
| *(cue-budget commit)* | 2026-10-04 | Ablation 2: random cue table, cue-budget runners, evaluation timing, this document renamed to `ABLATIONS.md` |

### Commit `3d98434`: history-condition ablation

**`src/03_backbone_recommender/genplaylist_tokenizer.py`**

- New constants: `HISTORY_CONDITIONS` (`full`, `latent_only`, `cue_only`,
  `shuffled_cue`), `CUE_NULL_TOKEN` (cue ID 0, `<unk>`, token 845),
  `SEMANTIC_NULL_TOKEN` (0, the padding/BOS token), `HISTORY_DONOR_FILE`
  (`history_donors.json`), and `HISTORY_DONOR_SCHEMA`.
- New functions:
  - `history_group(row_id)`: the playlist (MPD `<playlist>:joint5:<start>`) or
    user pseudonym (Music4All `m4a-<split>-<user>-...`) that produced a row.
  - `build_history_donors(rows, reference_items, seed)`: a seeded permutation
    in which every row receives a donor from a different group and every row
    donates exactly once. It refuses when one group owns more than half the
    rows, because no such permutation exists then.
  - `load_history_donors(path)`: reads `history_donors.json`.
- New tokenizer methods:
  - `set_history_condition(condition, donors=None)` validates and stores the
    variant. Donors are only accepted for `shuffled_cue`.
  - `encode_references(reference_ids, split=, row_id=)` encodes history items
    under the variant. It is the single place where history items are
    encoded: in `encode_playlist`, in the test branch of `tokenize`, and in the
    prepared test vectors.
- `encode_playlist` takes optional `split` and `row_id` and builds the target
  mask independently of the history encoding. For `cue_only`, it zeroes
  `context_emb`, `mu_c`, and `sigma_c2`.
- `tokenize` passes each row's `bundle` ID through as `row_id`.
- `from_dataset_config` reads `history_condition` from the config and, for
  `shuffled_cue`, loads `history_donors.json` from `prepared_dataset_path`.
- Target items, `encode_item`, decoding, the type mask, and the completion
  builder are unchanged.

**`scripts/prepare_wp_c_data.py`**

- New flags: `--history-condition` (default `full`) and `--donor-seed`
  (default 42; test donors use the seed, train donors the seed + 1).
- For `shuffled_cue`, writes `history_donors.json` into the prepared folder,
  covered by the folder's output hashes.
- The manifest gains `history_condition: {condition, donor_seed, donor_file,
  donor_sha256}`; the donor fields only appear for `shuffled_cue`.
- Refuses a non-`full` condition with `--active-cues 0`.

**`src/03_backbone_recommender/prepared_data.py`**

- `validate_prepared_manifest` raises `Prepared history condition mismatch`
  when the cache's condition differs from the configured `history_condition`.
- New `prepared_history_condition(manifest)`; manifests without the field
  count as `full`.

**`scripts/validate_wp_c_prepared_data.py`**

- New flag `--history-condition`; it also passes the prepared folder to the
  tokenizer so donors load.
- New `_validate_history_condition` re-derives every train and test row
  from catalog tokens, independently of the tokenizer, and checks:
  - BOI at every item start
  - nulls where the variant blanks a channel, real tokens elsewhere
  - real RVQ and cue tokens on all five train targets
  - `shuffled_cue`: donor map covers exactly the split, donors come from
    another group, donor items equal the donor row's references, and the
    assignment is a permutation
  - no donor file for other conditions
  - `cue_only`: zeroed `mu_c` and `context_emb`
  - Arrow test contexts identical to `vectors/eval_context_input_ids.npy`
  - `latent_only`: no real catalog cue equals the cue-null token

**`src/03_backbone_recommender/configs/config.yaml`**

- `history_condition: full`
- `eval.allow_history_condition_mismatch: false`

**`src/03_backbone_recommender/main.py`**

- New `_checkpoint_history_condition(path)` reads the condition a checkpoint
  was *trained* with from its saved hyperparameters. Evaluation overrides the
  in-memory hyperparameters, so the file is read directly. Older checkpoints
  count as `full`.
- `rec_eval` refuses test contexts of a different condition unless
  `eval.allow_history_condition_mismatch=true`.
- The result JSON gains `evaluation.history_condition: {test_contexts,
  checkpoint}`. `evaluation.official_protocol` is false when the protocol
  override or a condition mismatch is used.

**`src/03_backbone_recommender/scripts/train_spotify.sh`**

- `GENPLAYLIST_HISTORY_CONDITION` (default `full`) and `GENPLAYLIST_SEED`
  (default `1`, passed as Hydra `seed=`). Both are validated and appear in the
  run name.

**`src/03_backbone_recommender/scripts/eval_spotify.sh`**

- `GENPLAYLIST_HISTORY_CONDITION` and `GENPLAYLIST_EVAL_ALLOW_HISTORY_MISMATCH`;
  the condition appears in the default result filename.
- `GENPLAYLIST_EVAL_CKPT` is resolved to an absolute path. Hydra changes into
  a new run folder, so relative checkpoint paths used to fail.

**Tests**

- New `test_history_condition.py` (11 tests): `full` is byte-identical to plain
  item encoding; each variant blanks or swaps only the history channel;
  targets are never changed; `cue_only` contexts are accepted by joint
  completion; `shuffled_cue` without donors is rejected; unknown conditions
  are rejected; row-ID parsing for MPD and Music4All; donors are a
  cross-group permutation, deterministic per seed, and impossible
  assignments are rejected.
- `test_prepared_data.py` (+2 tests): condition mismatch is rejected;
  pre-ablation manifests count as `full`.

### Commit `5fd6457`: named run folders

**New `src/03_backbone_recommender/scripts/run_layout.sh`** (sourced by both
runners): resolves `GENPLAYLIST_OUTPUT_ROOT` (default
`src/03_backbone_recommender/outputs`) and `GENPLAYLIST_OUTPUT_NAME`, and
rejects names with characters outside `A-Z a-z 0-9 . _ -`, `.`/`..`
components, or empty components.

**`src/03_backbone_recommender/scripts/train_spotify.sh`**

- With `GENPLAYLIST_OUTPUT_NAME`, passes `hydra.run.dir=<root>/<name>`.
- Refuses to start when that folder already has `checkpoints/last.ckpt`.
- `GENPLAYLIST_TRAIN_MODE=resume` requires an existing `last.ckpt` and continues
  in the same folder.
- After a successful run, appends a row to `<root>/runs.csv` (fields listed in
  [Run folders](#run-folders-and-runscsv)).

**`src/03_backbone_recommender/scripts/eval_spotify.sh`**

- With `GENPLAYLIST_OUTPUT_NAME` and no `GENPLAYLIST_EVAL_CKPT`, evaluates
  `<name>/checkpoints/$GENPLAYLIST_EVAL_CKPT_FILE` (default `last.ckpt`).
- Writes `<name>/results/<checkpoint>-steps<S>-evalseed<E>.json`, prefixed
  with `on-<condition>-` for mismatch diagnostics. Refuses to overwrite an
  existing result unless `GENPLAYLIST_EVAL_RESULTS_PATH` is given.
- Hydra scratch folders go to `<name>/eval-runs/`.
- A missing checkpoint gives `Checkpoint not found: ...`.

### Commit `f42158a`: ablation 1 runner scripts

`scripts/ablation/train_history_cond_mpd.sh`,
`scripts/ablation/train_history_cond_m4a.sh`, and
`scripts/ablation/eval_history_cond.sh` (see
[Runner scripts](#runner-scripts-scriptsablation)).

### Cue-budget commit

- **New `scripts/ablation/build_random_cue_table.py`** and
  **`scripts/ablation/test_build_random_cue_table.py`** (5 tests): the
  selected cues are a rank-ordered subset followed by the rest; selection is
  deterministic per seed and independent of other songs; it is not just the
  top ranks; invalid requests are rejected; the command writes a valid cue
  folder (byte-identical vocabulary, updated manifest hash, derivation
  record) and refuses to overwrite.
- **New `scripts/ablation/common.sh`**, **`prepare_cue_budget.sh`**,
  **`train_cue_budget.sh`**, and **`eval_cue_budget.sh`**.
- **`src/03_backbone_recommender/main.py`**: `rec_eval` times the sampling
  calls and adds `evaluation.timing`, `evaluation.active_cue_tokens`, and
  `evaluation.tokens_per_sequence` to the result JSON. Metrics are unchanged.
- **Docs:** `ABLATION_CONDITIONING.md` became this file; references in the
  ablation 1 scripts and `config.yaml` point here.

### What did not change

- The frozen protocol, model architecture, loss curriculum, and official
  metrics.
- `history_condition=full` produces exactly the pre-ablation tokens.
- The cue-budget changes touch no fingerprinted preparation code, so every
  prepared folder built since `3d98434` stays valid.
- The WP-D runtime (`backbone_runtime.py`) still encodes contexts as `full`
  with the ranked cue table. Only Full checkpoints should be used for
  synthesis.

### Consequences

- `genplaylist_tokenizer.py`, `prepare_wp_c_data.py`, and `prepared_data.py`
  are fingerprinted by every prepared cache. **All prepared folders built
  before `3d98434` are rejected and must be rebuilt.** Tuteng's v3-u500
  Music4All cache is one of them; its three source folders are reused, not the
  cache itself.

### One-off scripts kept outside the repository

| Script | Purpose |
|---|---|
| `move_mpd_history_runs.sh [--apply]` | Moved the first four dated MPD seed-1 runs to `ablation-history-cond/mpd-<condition>` |
| `collect_results.sh [experiment]` | Bundles every model's `results/*.json` and `runs.csv` into one archive for download |
