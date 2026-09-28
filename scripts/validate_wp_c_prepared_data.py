#!/usr/bin/env python3
"""Exhaustively validate a prepared WP-C dataset and its offline vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
WP_ROOT = SRC_ROOT / "03_backbone_recommender"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(WP_ROOT))

from dataset import AbstractDataset  # noqa: E402
from config_composition import compose_wp_c_config  # noqa: E402
from genplaylist_tokenizer import (  # noqa: E402
    CUE_NULL_TOKEN,
    HISTORY_CONDITIONS,
    HISTORY_DONOR_FILE,
    SEMANTIC_NULL_TOKEN,
    GenPlaylistTokenizer,
    history_group,
    load_history_donors,
)
from prepared_data import (  # noqa: E402
    expected_split_counts,
    load_prepared_tokenized_dataset,
)
from shared.artifacts import sha256_file  # noqa: E402
from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402
from shared.schema import TOKEN_LAYOUT  # noqa: E402


def _configure(args):
    config = compose_wp_c_config()
    config.data_root = str(args.data_dir.expanduser().resolve())
    artifact_dir = args.artifact_dir.expanduser().resolve()
    cue_dir = args.cue_dir.expanduser().resolve()
    config.catalog_embeddings_path = str(artifact_dir / "catalog_item_embeddings.npy")
    config.item_id_to_row_path = str(artifact_dir / "item_id_to_row.json")
    config.semantic_tokens_path = str(artifact_dir / "semantic_tokens.json")
    config.codebook_weights_path = str(artifact_dir / "rvq_codebook_weights.npy")
    config.item2cues_path = str(cue_dir / "item2cues.json")
    config.cue_vocab_path = str(cue_dir / "cue_vocab.json")
    config.cue_manifest_path = str(cue_dir / "cue_manifest.json")
    config.active_cue_tokens = args.active_cues
    config.history_condition = args.history_condition
    config.prepared_dataset_path = str(args.prepared_dir.expanduser().resolve())
    config.model.length = FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
        1 + TOKEN_LAYOUT.rq_n_codebooks + 1 + args.active_cues)
    return config


def _assert_array_equal(actual, expected, name: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"{name} differs: actual={actual.shape}, expected={expected.shape}")


def _assert_array_close(actual, expected, name: str, atol: float = 1e-6) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=1e-5, atol=atol):
        difference = float(np.max(np.abs(actual - expected)))
        raise ValueError(f"{name} differs: max_abs_difference={difference}")


def _validate_output_hashes(root: Path, manifest: dict) -> None:
    expected = manifest.get("outputs", {})
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "prepared_manifest.json"
    }
    if actual_files != set(expected):
        raise ValueError(
            f"Prepared output file set differs: missing={sorted(set(expected) - actual_files)}, "
            f"extra={sorted(actual_files - set(expected))}")
    for relative, entry in expected.items():
        path = root / relative
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Prepared output size mismatch: {relative}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared output hash mismatch: {relative}")


def _validate_arrow(root: Path, dataset, tokenizer, tokenized, expected_counts: dict) -> None:
    from datasets import Dataset, load_from_disk

    raw = load_from_disk(str(root / "raw_dataset"))
    raw_counts = {split: len(raw[split]) for split in raw}
    if raw_counts != expected_counts:
        raise ValueError(f"Raw Arrow split counts drifted: {raw_counts}")

    for split, count in expected_counts.items():
        indices = sorted({0, count // 2, count - 1})
        source_rows = []
        for index in indices:
            source_row = raw[split][index]
            source_rows.append(source_row)
            if source_row != dataset.split_data[split][index]:
                raise ValueError(f"Raw {split}[{index}] differs from source expansion")

        # Rebuild a tiny in-memory Dataset. Running filter/map directly on the
        # disk-backed prepared Dataset would create cache-*.arrow files inside
        # the immutable release directory and invalidate its output manifest.
        sampled = Dataset.from_dict({
            column: [row[column] for row in source_rows]
            for column in raw[split].column_names
        })
        fresh = tokenizer.tokenize({split: sampled})[split]
        for sample_index, source_index in enumerate(indices):
            for field in fresh.column_names:
                left = fresh[sample_index][field]
                right = tokenized[split][source_index][field]
                if field in {"context_emb", "mu_c", "sigma_c2"}:
                    _assert_array_close(right, left, f"tokenized {split}[{source_index}].{field}")
                else:
                    _assert_array_equal(right, left, f"tokenized {split}[{source_index}].{field}")

    train_count = expected_counts["train"]
    train_examples = [
        tokenized["train"][index] for index in (0, train_count // 2, train_count - 1)]
    train_batch = tokenizer.collate_batch(train_examples)
    if train_batch["input_ids"].shape[1] > FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
            tokenizer.tokens_per_item):
        raise ValueError("Collated train sequence exceeds the configured model length")
    expected_target_tokens = (
        FROZEN_NEXT_SONG_PROTOCOL.train_target_items * (tokenizer.tokens_per_item - 1))
    if not np.all(
            train_batch["target_mask"].numpy().sum(axis=1) == expected_target_tokens):
        raise ValueError(
            f"Each train row must expose exactly {expected_target_tokens} target tokens")

    test_batch = tokenizer.collate_batch([tokenized["test"][0], tokenized["test"][-1]])
    expected_context_length = FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
        tokenizer.tokens_per_item, items=FROZEN_NEXT_SONG_PROTOCOL.eval_reference_items)
    if tuple(test_batch["input_ids"].shape) != (2, expected_context_length):
        raise ValueError(f"Test context shape drifted: {tuple(test_batch['input_ids'].shape)}")
    if tuple(test_batch["labels"].shape) != (2, 5, 4):
        raise ValueError(f"Test label shape drifted: {tuple(test_batch['labels'].shape)}")
    if test_batch["target_mask"].any():
        raise ValueError("Prepared test context must not expose target tokens")


def _load_vector(vector_dir: Path, manifest: dict, name: str):
    path = vector_dir / name
    array = np.load(path, allow_pickle=False)
    entry = manifest["vectors"].get(name)
    if entry is None:
        raise ValueError(f"Vector is absent from manifest: {name}")
    if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
        raise ValueError(f"Vector metadata differs for {name}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"Vector contains NaN or infinity: {name}")
    return array


def _validate_vectors(
    root: Path, manifest: dict, dataset, tokenizer, expected_counts: dict,
) -> None:
    vector_dir = root / "vectors"
    row_to_item = json.loads((vector_dir / "catalog_item_ids.json").read_text("utf-8"))
    expected_items = [None] * len(tokenizer.item_id_to_row)
    for item_id, row in tokenizer.item_id_to_row.items():
        expected_items[row] = item_id
    if row_to_item != expected_items:
        raise ValueError("catalog_item_ids.json differs from item_id_to_row")

    catalog = np.asarray(tokenizer.catalog_embeddings, dtype=np.float32)
    semantic = _load_vector(vector_dir, manifest, "catalog_semantic_tokens.npy")
    stored_cues = _load_vector(vector_dir, manifest, "catalog_stored_cues.npy")
    active_cues = _load_vector(vector_dir, manifest, "catalog_active_cues.npy")
    expected_semantic = np.asarray(
        [tokenizer.semantic_tokens[item_id] for item_id in row_to_item], dtype=np.int16)
    expected_stored_cues = np.asarray(
        [tokenizer.stored_item2cues[item_id] for item_id in row_to_item], dtype=np.int16)
    _assert_array_equal(semantic, expected_semantic, "catalog semantic tokens")
    _assert_array_equal(stored_cues, expected_stored_cues, "catalog stored cues")
    _assert_array_equal(
        active_cues, expected_stored_cues[:, :tokenizer.active_cues],
        "catalog active cues")

    catalog_l2 = _load_vector(vector_dir, manifest, "catalog_embeddings_l2.npy")
    _assert_array_close(catalog_l2, catalog / np.linalg.norm(catalog, axis=1, keepdims=True),
                        "normalized catalog embeddings")
    reconstructed = _load_vector(vector_dir, manifest, "catalog_rvq_reconstructed.npy")
    expected_reconstructed = np.stack([
        tokenizer._token_to_feature(tokens) for tokens in semantic
    ]).astype(np.float32)
    _assert_array_close(reconstructed, expected_reconstructed, "RVQ reconstructions")
    reconstructed_l2 = _load_vector(
        vector_dir, manifest, "catalog_rvq_reconstructed_l2.npy")
    _assert_array_close(
        reconstructed_l2,
        reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True),
        "normalized RVQ reconstructions")
    _assert_array_equal(
        _load_vector(vector_dir, manifest, "full_sequence_type_mask.npy"),
        tokenizer.make_type_mask(FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
            tokenizer.tokens_per_item)),
        "full legal-token type mask")

    bundle_ids = _load_vector(vector_dir, manifest, "eval_bundle_ids.npy")
    ref_ids = _load_vector(vector_dir, manifest, "eval_reference_item_ids.npy")
    target_ids = _load_vector(vector_dir, manifest, "eval_target_item_ids.npy")
    ref_rows = _load_vector(vector_dir, manifest, "eval_reference_rows.npy")
    target_rows = _load_vector(vector_dir, manifest, "eval_target_rows.npy")
    expected_bundles = []
    expected_ref_ids = []
    expected_target_ids = []
    for row in dataset.split_data["test"]:
        references, targets = FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(row["item_seq"])
        expected_bundles.append(str(row["bundle"]))
        expected_ref_ids.append(references)
        expected_target_ids.append(targets)
    _assert_array_equal(bundle_ids, expected_bundles, "evaluation bundle IDs")
    _assert_array_equal(ref_ids, expected_ref_ids, "evaluation reference IDs")
    _assert_array_equal(target_ids, expected_target_ids, "evaluation target IDs")
    expected_ref_rows = np.asarray([
        [tokenizer.item_id_to_row[item_id] for item_id in row] for row in expected_ref_ids
    ], dtype=np.int32)
    expected_target_rows = np.asarray([
        [tokenizer.item_id_to_row[item_id] for item_id in row] for row in expected_target_ids
    ], dtype=np.int32)
    _assert_array_equal(ref_rows, expected_ref_rows, "evaluation reference rows")
    _assert_array_equal(target_rows, expected_target_rows, "evaluation target rows")

    ref_embeddings = _load_vector(vector_dir, manifest, "eval_reference_embeddings.npy")
    target_embeddings = _load_vector(vector_dir, manifest, "eval_target_embeddings.npy")
    _assert_array_equal(ref_embeddings, catalog[ref_rows], "evaluation reference embeddings")
    _assert_array_equal(target_embeddings, catalog[target_rows], "evaluation target embeddings")
    ref_l2 = _load_vector(vector_dir, manifest, "eval_reference_embeddings_l2.npy")
    target_l2 = _load_vector(vector_dir, manifest, "eval_target_embeddings_l2.npy")
    _assert_array_close(
        ref_l2, ref_embeddings / np.linalg.norm(ref_embeddings, axis=2, keepdims=True),
        "normalized evaluation reference embeddings")
    _assert_array_close(
        target_l2, target_embeddings / np.linalg.norm(target_embeddings, axis=2, keepdims=True),
        "normalized evaluation target embeddings")

    mu_c = _load_vector(vector_dir, manifest, "eval_mu_c.npy")
    expected_mu = ref_embeddings.mean(axis=1, dtype=np.float32)
    _assert_array_close(mu_c, expected_mu, "evaluation context means")
    _assert_array_close(
        _load_vector(vector_dir, manifest, "eval_mu_c_l2.npy"),
        expected_mu / np.linalg.norm(expected_mu, axis=1, keepdims=True),
        "normalized evaluation context means")
    expected_sigma = np.mean(np.sum((ref_embeddings - expected_mu[:, None]) ** 2, axis=2), axis=1)
    _assert_array_close(
        _load_vector(vector_dir, manifest, "eval_sigma_c2.npy"), expected_sigma,
        "evaluation context variances")
    _assert_array_equal(
        _load_vector(vector_dir, manifest, "eval_target_semantic_tokens.npy"),
        semantic[target_rows], "evaluation target semantic tokens")
    _assert_array_equal(
        _load_vector(vector_dir, manifest, "eval_reference_active_cues.npy"),
        active_cues[ref_rows], "evaluation reference cues")
    _assert_array_equal(
        _load_vector(vector_dir, manifest, "eval_target_active_cues.npy"),
        active_cues[target_rows], "evaluation target cues")

    context_ids = _load_vector(vector_dir, manifest, "eval_context_input_ids.npy")
    completion_ids = _load_vector(vector_dir, manifest, "eval_completion_input_ids.npy")
    completion_mask = _load_vector(vector_dir, manifest, "eval_completion_mask.npy")
    expected_test = expected_counts["test"]
    context_length = FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
        tokenizer.tokens_per_item, items=FROZEN_NEXT_SONG_PROTOCOL.eval_reference_items)
    completion_length = FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
        tokenizer.tokens_per_item)
    if context_ids.shape != (expected_test, context_length) or completion_ids.shape != (
            expected_test, completion_length):
        raise ValueError("Evaluation context/completion token shapes drifted")
    expected_masked = (
        FROZEN_NEXT_SONG_PROTOCOL.eval_generated_items * (tokenizer.tokens_per_item - 1))
    if not np.all(completion_mask.sum(axis=1) == expected_masked):
        raise ValueError(
            f"Each evaluation completion must mask exactly {expected_masked} tokens")
    if not np.all(completion_ids[completion_mask] == TOKEN_LAYOUT.mask_token):
        raise ValueError("Evaluation completion payload contains a non-MASK token")
    context_payload_end = context_length - 1
    _assert_array_equal(
        completion_ids[:, :context_payload_end], context_ids[:, :context_payload_end],
                        "evaluation completion reference prefix")
    for item_index in range(5):
        boi_position = context_payload_end + item_index * tokenizer.tokens_per_item
        if not np.all(completion_ids[:, boi_position] == TOKEN_LAYOUT.boi_token):
            raise ValueError(
                f"Evaluation completion target {item_index} is missing BOI")
    if not np.all(completion_ids[:, -1] == TOKEN_LAYOUT.eos_token):
        raise ValueError("Evaluation completion does not end in EOS")

    cosine = _load_vector(vector_dir, manifest, "eval_reference_target_cosine.npy")
    _assert_array_close(
        cosine, np.matmul(ref_l2, np.swapaxes(target_l2, 1, 2)),
        "evaluation reference-target cosine")


def _column_matrix(dataset, column: str, dtype=np.int64) -> np.ndarray:
    values = dataset.with_format("numpy")[column]
    array = np.asarray(values)
    if array.dtype == object:
        array = np.stack(values)
    return array.astype(dtype)


def _check_blocks(
    name: str, blocks: np.ndarray, rows: np.ndarray, expected_cue_rows: np.ndarray | None,
    *, null_semantic: bool, null_cues: bool, semantic_table: np.ndarray,
    cue_table: np.ndarray,
) -> None:
    """Check ``[N, items, tokens_per_item]`` blocks against catalog tokens."""
    semantic_end = 1 + TOKEN_LAYOUT.rq_n_codebooks + 1
    if not np.all(blocks[:, :, 0] == TOKEN_LAYOUT.boi_token):
        raise ValueError(f"{name}: an item block does not start with BOI")
    semantic = blocks[:, :, 1:semantic_end]
    cues = blocks[:, :, semantic_end:]
    expected_semantic = (
        np.full_like(semantic, SEMANTIC_NULL_TOKEN) if null_semantic
        else semantic_table[rows])
    if null_cues:
        expected_cues = np.full_like(cues, CUE_NULL_TOKEN)
    else:
        expected_cues = cue_table[rows if expected_cue_rows is None else expected_cue_rows]
    _assert_array_equal(semantic, expected_semantic, f"{name} RVQ/conflict tokens")
    _assert_array_equal(cues, expected_cues, f"{name} cue tokens")


def _validate_history_condition(
    root: Path, dataset, tokenizer, tokenized, condition: str,
) -> None:
    """Independently re-derive every train/test row under the history ablation.

    References must match the condition; train targets must always carry
    real RVQ and cue tokens.
    """
    tokens_per_item = tokenizer.tokens_per_item
    reference_items = FROZEN_NEXT_SONG_PROTOCOL.train_reference_items
    target_items = FROZEN_NEXT_SONG_PROTOCOL.train_target_items
    row_of = tokenizer.item_id_to_row
    catalog_ids = [None] * len(row_of)
    for item_id, row in row_of.items():
        catalog_ids[row] = item_id
    semantic_table = np.asarray(
        [tokenizer.semantic_tokens[item_id] for item_id in catalog_ids], dtype=np.int64)
    cue_table = np.asarray(
        [[TOKEN_LAYOUT.cue_token(cue) for cue in tokenizer.item2cues[item_id]]
         for item_id in catalog_ids], dtype=np.int64)
    if condition == "latent_only" and np.any(cue_table == CUE_NULL_TOKEN):
        raise ValueError("A real catalog cue equals the cue-null token; latent_only is ambiguous")

    donor_payload = None
    if condition == "shuffled_cue":
        donor_path = root / HISTORY_DONOR_FILE
        if not donor_path.is_file():
            raise ValueError(f"shuffled_cue cache is missing {HISTORY_DONOR_FILE}")
        donor_payload = json.loads(donor_path.read_text(encoding="utf-8"))["splits"]
    elif (root / HISTORY_DONOR_FILE).exists():
        raise ValueError(f"{condition} cache must not contain {HISTORY_DONOR_FILE}")

    context_ids = np.load(root / "vectors" / "eval_context_input_ids.npy").astype(np.int64)
    for split in ("train", "test"):
        raw = dataset.split_data[split]
        row_ids = [str(row_id) for row_id in raw["bundle"]]
        sequences = raw["item_seq"]
        item_rows = np.asarray(
            [[row_of[str(item_id)] for item_id in sequence] for sequence in sequences],
            dtype=np.int64)
        reference_rows = item_rows[:, :reference_items]

        donor_rows = None
        if donor_payload is not None:
            split_donors = donor_payload[split]
            if set(split_donors) != set(row_ids):
                raise ValueError(f"{split} donor map does not cover exactly the split rows")
            position = {row_id: index for index, row_id in enumerate(row_ids)}
            donor_indices = []
            for row_id in row_ids:
                entry = split_donors[row_id]
                donor = entry["donor"]
                if history_group(donor) == history_group(row_id):
                    raise ValueError(f"{split} row {row_id} has a same-group donor {donor}")
                donor_index = position[donor]
                if list(entry["items"]) != [
                        str(item_id) for item_id in sequences[donor_index][:reference_items]]:
                    raise ValueError(f"{split} donor items for {row_id} differ from {donor}")
                donor_indices.append(donor_index)
            if len(set(donor_indices)) != len(donor_indices):
                raise ValueError(f"{split} donor assignment is not a permutation")
            donor_rows = reference_rows[np.asarray(donor_indices, dtype=np.int64)]

        input_ids = _column_matrix(tokenized[split], "input_ids")
        if split == "test":
            _assert_array_equal(input_ids, context_ids, "test Arrow contexts vs vectors")
        expected_items = reference_items + (target_items if split == "train" else 0)
        expected_length = 2 + expected_items * tokens_per_item
        if input_ids.shape != (len(row_ids), expected_length):
            raise ValueError(f"{split} input shape drifted: {input_ids.shape}")
        blocks = input_ids[:, 1:-1].reshape(len(row_ids), expected_items, tokens_per_item)
        _check_blocks(
            f"{split} references ({condition})", blocks[:, :reference_items],
            reference_rows, donor_rows,
            null_semantic=condition == "cue_only",
            null_cues=condition == "latent_only",
            semantic_table=semantic_table, cue_table=cue_table)
        if split == "train":
            _check_blocks(
                f"{split} targets", blocks[:, reference_items:],
                item_rows[:, reference_items:], None,
                null_semantic=False, null_cues=False,
                semantic_table=semantic_table, cue_table=cue_table)
        if condition == "cue_only":
            for column in ("mu_c", "context_emb"):
                if np.any(_column_matrix(tokenized[split], column, np.float32) != 0.0):
                    raise ValueError(f"{split} cue_only rows must zero {column}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "data" / "dataset")
    parser.add_argument(
        "--cue-dir", type=Path,
        default=SRC_ROOT / "02_creative_cues" / "outputs" / "production" / "latest")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--active-cues", type=int, default=TOKEN_LAYOUT.cue_tokens,
        choices=(0, 4, 8, 16),
        help="Cue count used when this prepared dataset was built.")
    parser.add_argument(
        "--history-condition", default="full", choices=HISTORY_CONDITIONS,
        help="History-condition ablation used when this prepared dataset was built.")
    args = parser.parse_args()

    root = args.prepared_dir.expanduser().resolve()
    config = _configure(args)
    dataset = AbstractDataset(config)
    expected_counts = expected_split_counts(dataset)
    tokenizer = GenPlaylistTokenizer.from_dataset_config(config, dataset)
    tokenized, manifest = load_prepared_tokenized_dataset(root, config, dataset, tokenizer)
    if manifest.get("git_dirty") is not False:
        raise ValueError(f"Release cache must have git_dirty=false, got {manifest.get('git_dirty')}")
    _validate_output_hashes(root, manifest)
    _validate_arrow(root, dataset, tokenizer, tokenized, expected_counts)
    _validate_vectors(root, manifest, dataset, tokenizer, expected_counts)
    _validate_history_condition(root, dataset, tokenizer, tokenized, args.history_condition)
    print(json.dumps({
        "status": "ok",
        "prepared_dir": str(root),
        "history_condition": args.history_condition,
        "git_commit": manifest.get("git_commit"),
        "split_counts": manifest["split_counts"],
        "output_files_verified": len(manifest["outputs"]),
        "vector_files_verified": len(manifest["vectors"]),
        "manifest_sha256": sha256_file(root / "prepared_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
