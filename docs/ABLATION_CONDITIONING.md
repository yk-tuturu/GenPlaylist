# Conditioning-channel ablation

This ablation asks which part of the reference history drives GenPlaylist's
predictions: the latent musical references (RVQ/conflict tokens), the creative
cues, or merely the presence of extra cue tokens. It is the first required
ablation and runs on both MPD and Music4All-Onion under the frozen 15->5
protocol in [`WP_C_TRAIN_EVAL_PROTOCOL.md`](WP_C_TRAIN_EVAL_PROTOCOL.md).

## Variants

Every variant uses the main 8-cue layout (13 tokens per item, 262-token
sequences), the same splits, catalog, warm start, loss curriculum, 20,000
optimization steps, and final-EMA-checkpoint rule. Only the **15 reference
items** change. The five target items always carry their real RVQ, conflict,
and cue tokens, so every variant is trained and scored on the same objective.

| `history_condition` | Reference RVQ + conflict tokens | Reference cue tokens |
|---|---|---|
| `full` | real | real |
| `latent_only` | real | cue-null token |
| `cue_only` | RVQ-null token | real |
| `shuffled_cue` | real | real cues of a donor history |

The existing 0-cue model (DDBC-SFT) is a different layout, with no cues
anywhere, and can be reported as an extra "no cues" row. It is not the
`latent_only` variant.

### Null tokens

The runtime vocabulary (IDs 0-2893) has no spare IDs, and adding one would
change the embedding size and break the `spotify30.ckpt` warm start.

- **Cue-null** is cue ID 0 (`<unk>`, token 845). The frozen cue table never
  assigns `<unk>` to a real item ([`CUE_VOCAB_FREEZE.md`](CUE_VOCAB_FREEZE.md)),
  so it never occurs in real histories. The validator enforces this.
- **RVQ-null** is token 0, the padding/BOS token. MASK is not used: reference
  contexts containing MASK are rejected, and MASK means "to be denoised".

Training loss is computed only on target payloads, and the `subs`
parameterization copies every unmasked position, so null tokens in the history
never enter the loss or the sampler.

For `cue_only`, the per-row CLHE statistics (`context_emb`, `mu_c`,
`sigma_c2`) are also zeroed. They are unused while CFG and structure
conditioning are off, which is the default. Zeroing them guarantees that no
latent history information leaks in.

### Shuffled-cue donors

`shuffled_cue` keeps the number, position, and format of cue tokens but takes
them from another history. Reference position *p* receives the cues of
reference item *p* of the donor row.

- Donors always come from a **different group**: a different playlist for MPD
  (rows are `<playlist>:joint5:<start>`) and a different user for Music4All
  (rows are `m4a-<split>-<user pseudonym>-...`). This matters because MPD
  training windows overlap with stride one; a naive shuffle would often pick a
  near-copy of the same history.
- The assignment is a seeded permutation, so every row donates exactly once.
  Train and test are assigned independently: test uses `--donor-seed`, train
  uses `--donor-seed + 1`.
- The mapping is computed once over the full split and stored as
  `history_donors.json` in the prepared directory, together with each donor's
  row ID. The file's hash is recorded in the manifest.

## Code changes

| File | Change |
|---|---|
| `src/03_backbone_recommender/genplaylist_tokenizer.py` | `HISTORY_CONDITIONS`, null-token constants, `history_group`, `build_history_donors`, `load_history_donors`; `set_history_condition` and `encode_references` apply the variant to history items in `encode_playlist`, train/test `tokenize`, and prepared test vectors. `from_dataset_config` reads `history_condition` and loads donors from the prepared directory. |
| `scripts/prepare_wp_c_data.py` | `--history-condition` and `--donor-seed`; writes `history_donors.json` for `shuffled_cue`; records `history_condition` in `prepared_manifest.json`; test context vectors use the variant. |
| `src/03_backbone_recommender/prepared_data.py` | Training and evaluation refuse a prepared cache whose `history_condition` differs from the configured one. Caches without the field count as `full`. |
| `scripts/validate_wp_c_prepared_data.py` | `--history-condition`. Independently re-derives **every** train and test row from catalog tokens and checks the variant: blanks where expected, real target tokens, donor cues, cross-group donors, permutation, zeroed CLHE statistics for `cue_only`, and Arrow test contexts equal to the stored vectors. |
| `src/03_backbone_recommender/configs/config.yaml` | `history_condition: full` and `eval.allow_history_condition_mismatch: false`. |
| `src/03_backbone_recommender/main.py` | `rec_eval` reads the checkpoint's trained `history_condition` and refuses test contexts of a different condition, unless the mismatch override is set. The result JSON records both conditions under `evaluation.history_condition`; a mismatch is marked `official_protocol: false`. |
| `scripts/train_spotify.sh` | `GENPLAYLIST_HISTORY_CONDITION` (default `full`) and `GENPLAYLIST_SEED` (default `1`); both appear in the run name. With `GENPLAYLIST_OUTPUT_NAME`, the run uses that folder and is logged to `runs.csv` (see [Run folders](#run-folders)). |
| `scripts/eval_spotify.sh` | `GENPLAYLIST_HISTORY_CONDITION`, `GENPLAYLIST_EVAL_ALLOW_HISTORY_MISMATCH`; the condition appears in the result filename. The checkpoint path is resolved to an absolute path, because Hydra changes directory. With `GENPLAYLIST_OUTPUT_NAME`, it evaluates that folder's checkpoint and writes results into it. |
| `scripts/run_layout.sh` | Shared run-folder naming for both runners. |
| `test_history_condition.py`, `test_prepared_data.py` | Tests covering all four variants, donor rules, and the manifest guard. |

With `history_condition=full`, the tokens are byte-identical to the
pre-ablation code; `test_full_matches_plain_item_encoding` checks this.
However, `genplaylist_tokenizer.py`, `prepare_wp_c_data.py`, and
`prepared_data.py` are fingerprinted by every prepared cache, so **all existing
prepared directories, including Full, must be rebuilt** with this code.

The WP-D runtime (`backbone_runtime.py`) still encodes contexts as `full`.
Only the Full checkpoint should be used for synthesis.

## Running it

Commit the code first: `prepare_wp_c_data.py` refuses a dirty worktree. The
commands below assume the repository at `$REPO` contains `data/`, the cue
outputs, and `checkpoints/pretrained/ddbc/spotify30.ckpt`. Always pass
absolute paths.

```bash
export REPO=$HOME/tt_workspace/waikei-ablation/GenPlaylist OUT=$HOME/tt_workspace/waikei-ablation/GenPlaylist/src/03_backbone_recommender/outputs
```

### 1. Prepare and validate (once per variant and dataset)

MPD:

```bash
for V in full latent_only cue_only shuffled_cue; do python $REPO/scripts/prepare_wp_c_data.py --data-dir $REPO/data/dataset --artifact-dir $REPO/data/dataset --output-dir $REPO/data/processed/ablation-mpd-8cue-$V --active-cues 8 --history-condition $V --donor-seed 42 && python $REPO/scripts/validate_wp_c_prepared_data.py --data-dir $REPO/data/dataset --artifact-dir $REPO/data/dataset --prepared-dir $REPO/data/processed/ablation-mpd-8cue-$V --active-cues 8 --history-condition $V; done
```

For Music4All, use the expanded **v3-u500** catalog built at commit `613bdf5`
(10,950 songs, 1,227,307 training windows, 19,771 test histories; empty
validation split). Its three source folders are, on the server under
`/home/wjzhang/tt_workspace/model/GenPlaylist`:

| Role | Folder | Key file SHA-256 |
|---|---|---|
| `--data-dir` / `GENPLAYLIST_DATA_ROOT` | `data/dataset-music4all-onion-v3-u500` | `catalog_metadata.json` `8f8abf7f…` |
| `--artifact-dir` / `GENPLAYLIST_ARTIFACT_ROOT` | `data/artifacts-music4all-onion-v3-u500` | `item_id_to_row.json` `cf86c136…` |
| `--cue-dir` / `GENPLAYLIST_CUE_ROOT` | `data/processed/cues-music4all-onion-v3-u500/latest` | `item2cues.json` `7e8c5642…` |

The cue vocabulary is byte-identical to MPD's (`cue_vocab.json` `cd229492…`),
so cue IDs mean the same on both datasets. Always pass `--cue-dir`; otherwise
prep falls back to the MPD cues.

```bash
D=$REPO/data/dataset-music4all-onion-v3-u500; A=$REPO/data/artifacts-music4all-onion-v3-u500; C=$REPO/data/processed/cues-music4all-onion-v3-u500/latest
```

```bash
for V in full latent_only cue_only shuffled_cue; do python $REPO/scripts/prepare_wp_c_data.py --data-dir $D --artifact-dir $A --cue-dir $C --output-dir $REPO/data/processed/ablation-m4a-v3-u500-8cue-$V --active-cues 8 --history-condition $V --donor-seed 42 && python $REPO/scripts/validate_wp_c_prepared_data.py --data-dir $D --artifact-dir $A --cue-dir $C --prepared-dir $REPO/data/processed/ablation-m4a-v3-u500-8cue-$V --active-cues 8 --history-condition $V; done
```

Each Music4All prepared folder is about 2.5 GB, and evaluation covers ~20x
more test histories than MPD.

### Run folders

Set `GENPLAYLIST_OUTPUT_NAME` to choose a run's folder yourself, instead of
Hydra's dated `outputs/<data>/<date>/<time>/`. Slashes group runs:

```
src/03_backbone_recommender/outputs/
├── runs.csv                              one row per finished named training run
└── ablation-cond/mpd/cue_only-seed2/     GENPLAYLIST_OUTPUT_NAME
    ├── .hydra/
    ├── checkpoints/{last,step-500,...}.ckpt
    ├── results/last-steps256-evalseed1.json
    └── eval-runs/...                     Hydra folders of evaluation runs
```

- The name is used exactly as given (letters, digits, `.`, `_`, `-`, and `/`
  between names). Nothing else about the run is encoded in it, so put the
  variant, seed, and anything non-default (cue count, loss schedule, a
  500-step smoke run) into the name yourself.
- Training refuses to start in a folder that already has `last.ckpt`.
  `GENPLAYLIST_TRAIN_MODE=resume` continues a crashed run in the same folder;
  resume with the same variables as the original run.
- `runs.csv` records time, output name, dataset, condition, cues, loss variant,
  seed, steps, train mode, git commit, dirty flag, prepared-manifest SHA-256,
  checkpoint path, and checkpoint SHA-256. A resumed run adds a new row; the
  latest row for a name is the final one.
- Evaluation with the same `GENPLAYLIST_OUTPUT_NAME` evaluates that folder's
  `last.ckpt` (`GENPLAYLIST_EVAL_CKPT_FILE=step-10000.ckpt` picks an
  intermediate one) and writes to its `results/` folder. It refuses to
  overwrite an existing result. `GENPLAYLIST_EVAL_SEED` stays the sampling
  seed (1).
- `GENPLAYLIST_OUTPUT_ROOT` moves the whole tree (default
  `src/03_backbone_recommender/outputs`).

Without `GENPLAYLIST_OUTPUT_NAME`, both runners behave exactly as before.

Runs trained before this layout can be linked into it once. The link only
points at the old dated folder; nothing is copied:

```bash
mkdir -p $OUT/ablation-cond/mpd; for V in full latent_only cue_only shuffled_cue; do R=$(grep -lx -- "- history_condition=$V" $OUT/spotify/*/*/.hydra/overrides.yaml | xargs grep -lx -- "- seed=1" | xargs grep -lx -- "- trainer.max_steps=20000" | xargs -n1 dirname | xargs -n1 dirname); [ $(echo "$R" | grep -c .) -eq 1 ] && ln -sfn "$R" $OUT/ablation-cond/mpd/$V-seed1 || echo "check $V: $R"; done
```

Linked runs are not in `runs.csv`; record their checkpoint hashes by hand.

### 2. Smoke test (500 steps per new variant)

```bash
CUDA_VISIBLE_DEVICES=7 GENPLAYLIST_OUTPUT_NAME=smoke/mpd/cue_only GENPLAYLIST_HISTORY_CONDITION=cue_only GENPLAYLIST_PREPARED_DATA_ROOT=$REPO/data/processed/ablation-mpd-8cue-cue_only GENPLAYLIST_MAX_STEPS=500 bash $REPO/src/03_backbone_recommender/scripts/train_spotify.sh 2>&1 | tee smoke_cue_only.log
```

### 3. Train (4 variants x 3 seeds x 2 datasets = 24 runs)

One seed of all four MPD variants, sequentially on one GPU:

```bash
SEED=1; for V in full latent_only cue_only shuffled_cue; do CUDA_VISIBLE_DEVICES=7 GENPLAYLIST_OUTPUT_NAME=ablation-cond/mpd/$V-seed$SEED GENPLAYLIST_HISTORY_CONDITION=$V GENPLAYLIST_SEED=$SEED GENPLAYLIST_PREPARED_DATA_ROOT=$REPO/data/processed/ablation-mpd-8cue-$V bash $REPO/src/03_backbone_recommender/scripts/train_spotify.sh 2>&1 | tee train_mpd_${V}_s$SEED.log; done
```

Repeat with `SEED=2` and `3`. For Music4All, name the runs
`ablation-cond/m4a/$V-seed$SEED`, also set `GENPLAYLIST_DATA_CONFIG=music4all`,
`GENPLAYLIST_DATA_ROOT=$D`, `GENPLAYLIST_ARTIFACT_ROOT=$A`,
`GENPLAYLIST_CUE_ROOT=$C`, and use the `ablation-m4a-v3-u500-8cue-$V` prepared
folders.

### 4. Evaluate

Use the same output name and variables as in training, without a checkpoint
path:

```bash
SEED=1; for V in full latent_only cue_only shuffled_cue; do CUDA_VISIBLE_DEVICES=7 GENPLAYLIST_OUTPUT_NAME=ablation-cond/mpd/$V-seed$SEED GENPLAYLIST_HISTORY_CONDITION=$V GENPLAYLIST_PREPARED_DATA_ROOT=$REPO/data/processed/ablation-mpd-8cue-$V bash $REPO/src/03_backbone_recommender/scripts/eval_spotify.sh; done
```

Then compute the MERT metrics and bootstrap intervals for each result:

```bash
SEED=1; for V in full latent_only cue_only shuffled_cue; do R=$OUT/ablation-cond/mpd/$V-seed$SEED/results/last-steps256-evalseed1.json; python $REPO/scripts/evaluate_mert_proxy.py --prediction-result $R --mert-dir /path/to/mert-v1-95m-catalog-v1 --output ${R%.json}-mert.json; done
```

**Diagnostic (no training):** the Full checkpoint evaluated on shuffled
test cues tests whether Full reads its cues at inference. The result is marked
unofficial and saved as `on-shuffled_cue-last-steps256-evalseed1.json` in the
Full run's folder.

```bash
CUDA_VISIBLE_DEVICES=7 GENPLAYLIST_OUTPUT_NAME=ablation-cond/mpd/full-seed1 GENPLAYLIST_HISTORY_CONDITION=shuffled_cue GENPLAYLIST_EVAL_ALLOW_HISTORY_MISMATCH=true GENPLAYLIST_PREPARED_DATA_ROOT=$REPO/data/processed/ablation-mpd-8cue-shuffled_cue bash $REPO/src/03_backbone_recommender/scripts/eval_spotify.sh
```

## Reading the results

| Comparison | Question answered |
|---|---|
| Full vs shuffled-cue | Do cue *semantics* matter, beyond extra conditioning tokens? |
| Full vs latent-only | Do cues add anything over latent references? |
| Full vs cue-only | Do latent references add anything over cues? |
| Latent-only vs cue-only | Which channel carries more preference signal? |

Report the mean and spread over seeds, with paired history-level bootstrap
differences against Full (10,000 samples, seed 42), for Recall@5, N1-MERT,
M2M-MERT, Coverage@5, and cue F1. Record for each run: dataset, variant, seed,
git commit, prepared-manifest SHA-256, donor-file SHA-256 (shuffled only),
checkpoint SHA-256, and result JSON paths.
