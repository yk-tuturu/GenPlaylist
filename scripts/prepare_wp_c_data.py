#!/usr/bin/env python3
"""Prepare all checkpoint-independent WP-C train/test data and vectors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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
    HISTORY_CONDITIONS,
    HISTORY_DONOR_FILE,
    HISTORY_DONOR_SCHEMA,
    GenPlaylistTokenizer,
    build_history_donors,
)
from prepared_data import (  # noqa: E402
    PREPARED_DATA_VERSION,
    configured_source_paths,
    expected_split_counts,
    preparation_code_manifest,
    source_manifest,
)
from shared.artifacts import sha256_file  # noqa: E402
from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402
from shared.schema import SCHEMA_VERSION, TOKEN_LAYOUT  # noqa: E402


def _save_npy(path: Path, values, dtype=None) -> dict:
    array = np.asarray(values, dtype=dtype)
    np.save(path, array, allow_pickle=False)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            check=True, capture_output=True)
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _directory_outputs(root: Path) -> dict:
    outputs = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "prepared_manifest.json":
            relative = str(path.relative_to(root))
            outputs[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return outputs


def _cast_tokenized(tokenized):
    from datasets import Features, Sequence, Value

    base = {
        "input_ids": Sequence(Value("int32")),
        "sequence_mask": Sequence(Value("bool")),
        "attention_mask": Sequence(Value("bool")),
        "target_mask": Sequence(Value("bool")),
        "context_emb": Sequence(Value("float32"), length=64),
        "mu_c": Sequence(Value("float32"), length=64),
        "sigma_c2": Value("float32"),
    }
    output = {}
    for split, dataset in tokenized.items():
        dataset.reset_format()
        features = dict(base)
        if split == "test":
            features["labels"] = Sequence(
                Sequence(Value("int32"), length=4), length=5)
        output[split] = dataset.cast(Features(features))
    return output


def _build_vectors(root: Path, dataset, tokenizer) -> dict:
    vector_dir = root / "vectors"
    vector_dir.mkdir(parents=True)
    metadata = {}

    row_to_item = [None] * len(tokenizer.item_id_to_row)
    for item_id, row in tokenizer.item_id_to_row.items():
        row_to_item[row] = item_id
    if any(item_id is None for item_id in row_to_item):
        raise ValueError("item_id_to_row is not contiguous")
    (vector_dir / "catalog_item_ids.json").write_text(
        json.dumps(row_to_item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    catalog = np.asarray(tokenizer.catalog_embeddings, dtype=np.float32)
    semantic = np.asarray(
        [tokenizer.semantic_tokens[item_id] for item_id in row_to_item], dtype=np.int16)
    stored_cues = np.asarray(
        [tokenizer.stored_item2cues[item_id] for item_id in row_to_item], dtype=np.int16)
    active_cues = stored_cues[:, :tokenizer.active_cues]
    rvq_reconstructed = np.stack([
        tokenizer._token_to_feature(tokens) for tokens in semantic
    ]).astype(np.float32)
    catalog_arrays = {
        "catalog_embeddings_l2.npy": _l2_normalize(catalog),
        "catalog_rvq_reconstructed.npy": rvq_reconstructed,
        "catalog_rvq_reconstructed_l2.npy": _l2_normalize(rvq_reconstructed),
        "catalog_semantic_tokens.npy": semantic,
        "catalog_stored_cues.npy": stored_cues,
        "catalog_active_cues.npy": active_cues,
        "full_sequence_type_mask.npy": tokenizer.make_type_mask(
            FROZEN_NEXT_SONG_PROTOCOL.model_token_length(tokenizer.tokens_per_item)),
    }
    for filename, values in catalog_arrays.items():
        metadata[filename] = _save_npy(vector_dir / filename, values)

    bundles = []
    reference_ids = []
    target_ids = []
    reference_rows = []
    target_rows = []
    context_ids = []
    completion_ids = []
    completion_masks = []
    reference_embeddings = []
    target_embeddings = []
    mu_values = []
    sigma_values = []
    target_semantic = []
    reference_cues = []
    target_cues = []

    for row in dataset.split_data["test"]:
        references, targets = FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(
            row["item_seq"])
        ref_rows = [tokenizer.item_id_to_row[item_id] for item_id in references]
        tgt_rows = [tokenizer.item_id_to_row[item_id] for item_id in targets]
        context = [
            tokenizer.bos_token,
            *tokenizer.encode_references(
                references, split="test", row_id=str(row["bundle"])),
            tokenizer.eos_token,
        ]
        completed, completion_mask = tokenizer.build_item_completion(
            context, num_items=FROZEN_NEXT_SONG_PROTOCOL.eval_generated_items)
        ref_emb = catalog[ref_rows]
        mu_c = ref_emb.mean(axis=0, dtype=np.float32)

        bundles.append(str(row["bundle"]))
        reference_ids.append(references)
        target_ids.append(targets)
        reference_rows.append(ref_rows)
        target_rows.append(tgt_rows)
        context_ids.append(context)
        completion_ids.append(completed)
        completion_masks.append(completion_mask)
        reference_embeddings.append(ref_emb)
        target_embeddings.append(catalog[tgt_rows])
        mu_values.append(mu_c)
        sigma_values.append(np.mean(np.sum((ref_emb - mu_c) ** 2, axis=1)))
        target_semantic.append([tokenizer.semantic_tokens[item_id] for item_id in targets])
        reference_cues.append([tokenizer.item2cues[item_id] for item_id in references])
        target_cues.append([tokenizer.item2cues[item_id] for item_id in targets])

    string_arrays = {
        "eval_bundle_ids.npy": bundles,
        "eval_reference_item_ids.npy": reference_ids,
        "eval_target_item_ids.npy": target_ids,
    }
    numeric_arrays = {
        "eval_reference_rows.npy": (reference_rows, np.int32),
        "eval_target_rows.npy": (target_rows, np.int32),
        "eval_context_input_ids.npy": (context_ids, np.int16),
        "eval_completion_input_ids.npy": (completion_ids, np.int16),
        "eval_completion_mask.npy": (completion_masks, np.bool_),
        "eval_reference_embeddings.npy": (reference_embeddings, np.float32),
        "eval_reference_embeddings_l2.npy": (
            _l2_normalize(np.asarray(reference_embeddings)), np.float32),
        "eval_target_embeddings.npy": (target_embeddings, np.float32),
        "eval_target_embeddings_l2.npy": (
            _l2_normalize(np.asarray(target_embeddings)), np.float32),
        "eval_mu_c.npy": (mu_values, np.float32),
        "eval_mu_c_l2.npy": (_l2_normalize(np.asarray(mu_values)), np.float32),
        "eval_sigma_c2.npy": (sigma_values, np.float32),
        "eval_target_semantic_tokens.npy": (target_semantic, np.int16),
        "eval_reference_active_cues.npy": (reference_cues, np.int16),
        "eval_target_active_cues.npy": (target_cues, np.int16),
    }
    for filename, values in string_arrays.items():
        metadata[filename] = _save_npy(vector_dir / filename, values)
    for filename, (values, dtype) in numeric_arrays.items():
        metadata[filename] = _save_npy(vector_dir / filename, values, dtype=dtype)

    refs_l2 = np.load(vector_dir / "eval_reference_embeddings_l2.npy", allow_pickle=False)
    targets_l2 = np.load(vector_dir / "eval_target_embeddings_l2.npy", allow_pickle=False)
    metadata["eval_reference_target_cosine.npy"] = _save_npy(
        vector_dir / "eval_reference_target_cosine.npy",
        np.einsum("brid,brjd->brij", refs_l2[:, None], targets_l2[:, None])[:, 0],
        dtype=np.float32,
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=REPO_ROOT / "data" / "dataset")
    parser.add_argument(
        "--cue-dir", type=Path,
        default=SRC_ROOT / "02_creative_cues" / "outputs" / "production" / "latest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--active-cues", type=int, default=TOKEN_LAYOUT.cue_tokens,
        choices=(0, 4, 8, 16),
        help="Ranked cues encoded per item; each value defines a separate model layout.")
    parser.add_argument(
        "--history-condition", default="full", choices=HISTORY_CONDITIONS,
        help="Conditioning-channel ablation applied to the 15 reference items.")
    parser.add_argument(
        "--donor-seed", type=int, default=42,
        help="Seed for the shuffled_cue donor permutation (ignored otherwise).")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow generation from an uncommitted worktree (recorded in the manifest).")
    args = parser.parse_args()

    git_dirty = _git_dirty()
    if git_dirty and not args.allow_dirty:
        raise RuntimeError(
            "Refusing to prepare data from a dirty worktree. Commit the preparation "
            "code first, or pass --allow-dirty for an explicitly non-release cache.")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared data: {output}. Use a new versioned directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        from datasets import DatasetDict
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
        config.model.length = FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
            1 + TOKEN_LAYOUT.rq_n_codebooks + 1 + args.active_cues)
        FROZEN_NEXT_SONG_PROTOCOL.validate_config(config)
        if args.history_condition != "full" and args.active_cues == 0:
            raise ValueError("History-condition ablations require active cues")

        dataset = AbstractDataset(config)
        counts = {split: len(rows) for split, rows in dataset.split_data.items()}
        expected_counts = expected_split_counts(dataset)
        if counts != expected_counts:
            raise ValueError(f"Frozen split counts changed: {counts}")
        tokenizer = GenPlaylistTokenizer.from_dataset_config(config, dataset)
        history_record = {"condition": args.history_condition}
        if args.history_condition == "shuffled_cue":
            donor_splits = {
                split: build_history_donors(
                    list(zip(rows["bundle"], rows["item_seq"])),
                    reference_items=FROZEN_NEXT_SONG_PROTOCOL.train_reference_items,
                    seed=args.donor_seed + index)
                for index, (split, rows) in enumerate(sorted(dataset.split_data.items()))
            }
            (temp / HISTORY_DONOR_FILE).write_text(json.dumps({
                "schema": HISTORY_DONOR_SCHEMA,
                "seed": args.donor_seed,
                "reference_items": FROZEN_NEXT_SONG_PROTOCOL.train_reference_items,
                "splits": donor_splits,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            tokenizer.set_history_condition("shuffled_cue", {
                split: {row_id: entry["items"] for row_id, entry in rows.items()}
                for split, rows in donor_splits.items()
            })
            history_record.update({
                "donor_seed": args.donor_seed,
                "donor_file": HISTORY_DONOR_FILE,
                "donor_sha256": sha256_file(temp / HISTORY_DONOR_FILE),
            })

        raw = DatasetDict(dataset.split_data)
        raw.save_to_disk(str(temp / "raw_dataset"))
        tokenized = DatasetDict(_cast_tokenized(tokenizer.tokenize(dataset.split_data)))
        for split in tokenized:
            tokenized[split].set_format(type="torch")
        tokenized.save_to_disk(str(temp / "tokenized_dataset"))
        vector_metadata = _build_vectors(temp, dataset, tokenizer)

        manifest = {
            "prepared_data_version": PREPARED_DATA_VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "git_dirty": git_dirty,
            "preparation_code": preparation_code_manifest(REPO_ROOT),
            "protocol": {
                "min_reference_items": FROZEN_NEXT_SONG_PROTOCOL.min_reference_items,
                "train_total_items": FROZEN_NEXT_SONG_PROTOCOL.train_total_items,
                "train_target_items": FROZEN_NEXT_SONG_PROTOCOL.train_target_items,
                "eval_total_items": FROZEN_NEXT_SONG_PROTOCOL.eval_total_items,
                "eval_reference_items": FROZEN_NEXT_SONG_PROTOCOL.eval_reference_items,
                "eval_target_items": FROZEN_NEXT_SONG_PROTOCOL.eval_target_items,
                "eval_num_samples": FROZEN_NEXT_SONG_PROTOCOL.eval_num_samples,
                "eval_generated_items": FROZEN_NEXT_SONG_PROTOCOL.eval_generated_items,
            },
            "token_layout": {
                "tokens_per_item": tokenizer.tokens_per_item,
                "runtime_vocab_size": TOKEN_LAYOUT.runtime_vocab_size,
                "model_length": FROZEN_NEXT_SONG_PROTOCOL.model_token_length(
                    tokenizer.tokens_per_item),
            },
            "history_condition": history_record,
            "split_counts": counts,
            "source_artifacts": source_manifest(configured_source_paths(config, dataset)),
            "vectors": vector_metadata,
        }
        manifest["outputs"] = _directory_outputs(temp)
        manifest_path = temp / "prepared_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, output)
        print(json.dumps({
            "output": str(output),
            "split_counts": counts,
            "vector_files": len(vector_metadata),
            "output_files": len(manifest["outputs"]),
            "manifest_sha256": sha256_file(output / "prepared_manifest.json"),
        }, indent=2))
        return 0
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
