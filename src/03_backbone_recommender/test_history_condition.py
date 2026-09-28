"""Contract tests for the conditioning-channel (history_condition) ablation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "genplaylist_tokenizer", HERE / "genplaylist_tokenizer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GenPlaylistTokenizer = module.GenPlaylistTokenizer

from shared.schema import CatalogItem, TOKEN_LAYOUT  # noqa: E402

PROTOCOL_CONFIG = {
    "seq_len": 20,
    "rq_codebook_size": 256,
    "protocol": {
        "min_reference_items": 15,
        "train_target_items": 5,
        "eval_reference_items": 15,
        "eval_target_items": 5,
        "eval_num_samples": 1,
        "eval_generated_items": 5,
    },
}
SEMANTIC_END = 1 + TOKEN_LAYOUT.rq_n_codebooks + 1


def make_tokenizer(item_count=40):
    """Items with distinct RVQ tokens and distinct, never-<unk> cues."""
    item_ids = [str(20000 + index) for index in range(item_count)]
    items = [CatalogItem(item_id) for item_id in item_ids]
    semantic = {
        item_id: [1 + index, 257 + index, 513 + index, 769 + index]
        for index, item_id in enumerate(item_ids)
    }
    cues = {
        item_id: list(range(1 + 16 * index, 17 + 16 * index))
        for index, item_id in enumerate(item_ids)
    }
    embeddings = np.stack([
        np.full(64, float(index + 1), dtype=np.float32) for index in range(item_count)])
    weights = np.arange(768 * 64, dtype=np.float32).reshape(768, 64)
    tokenizer = GenPlaylistTokenizer(
        semantic, cues, items, embeddings,
        {item_id: index for index, item_id in enumerate(item_ids)}, weights)
    tokenizer.config = PROTOCOL_CONFIG
    tokenizer.max_items = 20
    return tokenizer, item_ids


class FakeRows:
    """Small subset of the HF Dataset interface used by tokenize()."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def filter(self, function):
        return FakeRows([row for row in self.rows if function(row)])

    def map(self, function, **_kwargs):
        return FakeRows([function(row) for row in self.rows])

    def set_format(self, **_kwargs):
        return None

    def __getitem__(self, index):
        return self.rows[index]


def blocks(input_ids, tokenizer):
    values = np.asarray(input_ids)[1:-1]
    return values.reshape(-1, tokenizer.tokens_per_item)


def two_rows(item_ids):
    return [
        {"bundle": "pA:joint5:0", "item_seq": item_ids[:20]},
        {"bundle": "pB:joint5:0", "item_seq": item_ids[20:40]},
    ]


def donors_for(rows):
    return {
        row["bundle"]: {"donor": other["bundle"], "items": other["item_seq"][:15]}
        for row, other in zip(rows, reversed(rows))
    }


def test_full_matches_plain_item_encoding():
    tokenizer, item_ids = make_tokenizer()
    row = tokenizer.tokenize({"train": FakeRows(two_rows(item_ids)[:1])})["train"][0]
    expected = [tokenizer.bos_token]
    for item_id in item_ids[:20]:
        expected.extend(tokenizer.encode_item(item_id))
    expected.append(tokenizer.eos_token)
    assert row["input_ids"] == expected
    assert len(row["input_ids"]) == 262


def test_latent_only_blanks_reference_cues_and_keeps_targets():
    tokenizer, item_ids = make_tokenizer()
    tokenizer.set_history_condition("latent_only")
    row = tokenizer.tokenize({"train": FakeRows(two_rows(item_ids)[:1])})["train"][0]
    item_blocks = blocks(row["input_ids"], tokenizer)
    assert len(row["input_ids"]) == 262
    for position in range(15):
        original = tokenizer.encode_item(item_ids[position])
        assert list(item_blocks[position, :SEMANTIC_END]) == original[:SEMANTIC_END]
        assert set(item_blocks[position, SEMANTIC_END:]) == {module.CUE_NULL_TOKEN}
    for position in range(15, 20):
        assert list(item_blocks[position]) == tokenizer.encode_item(item_ids[position])
    assert sum(row["target_mask"]) == 5 * (tokenizer.tokens_per_item - 1)


def test_cue_only_blanks_reference_latents_and_clhe_context():
    tokenizer, item_ids = make_tokenizer()
    tokenizer.set_history_condition("cue_only")
    row = tokenizer.tokenize({"train": FakeRows(two_rows(item_ids)[:1])})["train"][0]
    item_blocks = blocks(row["input_ids"], tokenizer)
    for position in range(15):
        original = tokenizer.encode_item(item_ids[position])
        assert set(item_blocks[position, 1:SEMANTIC_END]) == {module.SEMANTIC_NULL_TOKEN}
        assert list(item_blocks[position, SEMANTIC_END:]) == original[SEMANTIC_END:]
        assert item_blocks[position, 0] == tokenizer.boi_token
    for position in range(15, 20):
        assert list(item_blocks[position]) == tokenizer.encode_item(item_ids[position])
    assert np.all(np.asarray(row["mu_c"]) == 0.0)
    assert np.all(np.asarray(row["context_emb"]) == 0.0)
    assert row["sigma_c2"] == 0.0


def test_cue_only_context_is_accepted_by_joint_completion():
    tokenizer, item_ids = make_tokenizer()
    tokenizer.set_history_condition("cue_only")
    row = tokenizer.tokenize({"test": FakeRows(two_rows(item_ids)[:1])})["test"][0]
    completed, completion_mask = tokenizer.build_item_completion(
        row["input_ids"], num_items=5)
    assert len(completed) == 262
    assert completion_mask.sum() == 5 * (tokenizer.tokens_per_item - 1)
    assert list(completed[:len(row["input_ids"]) - 1]) == row["input_ids"][:-1]


def test_shuffled_cue_uses_donor_cues_on_train_and_test():
    tokenizer, item_ids = make_tokenizer()
    rows = two_rows(item_ids)
    donors = donors_for(rows)
    tokenizer.set_history_condition("shuffled_cue", {
        split: {row_id: entry["items"] for row_id, entry in donors.items()}
        for split in ("train", "test")
    })
    for split in ("train", "test"):
        row = tokenizer.tokenize({split: FakeRows(rows)})[split][0]
        item_blocks = blocks(row["input_ids"], tokenizer)
        donor_items = rows[1]["item_seq"][:15]
        for position in range(15):
            own = tokenizer.encode_item(item_ids[position])
            donor = tokenizer.encode_item(donor_items[position])
            assert list(item_blocks[position, :SEMANTIC_END]) == own[:SEMANTIC_END]
            assert list(item_blocks[position, SEMANTIC_END:]) == donor[SEMANTIC_END:]
        if split == "train":
            for position in range(15, 20):
                assert list(item_blocks[position]) == tokenizer.encode_item(item_ids[position])


def test_shuffled_cue_without_donors_is_rejected():
    tokenizer, item_ids = make_tokenizer()
    tokenizer.set_history_condition("shuffled_cue")
    try:
        tokenizer.tokenize({"train": FakeRows(two_rows(item_ids)[:1])})
    except ValueError as exc:
        assert "donor" in str(exc)
    else:
        raise AssertionError("Expected shuffled_cue without donors to be rejected")


def test_unknown_condition_and_stray_donors_are_rejected():
    tokenizer, _ = make_tokenizer()
    for condition, donors in (("no_cues", None), ("full", {"train": {"x": []}})):
        try:
            tokenizer.set_history_condition(condition, donors)
        except ValueError:
            continue
        raise AssertionError(f"Expected {condition!r} with {donors!r} to be rejected")


def test_history_group_parses_mpd_and_music4all_row_ids():
    assert module.history_group("918:joint5:12") == "918"
    assert module.history_group("918") == "918"
    assert module.history_group(
        "m4a-train-0123456789abcdef-r0003-s00000042:joint5:0") == "0123456789abcdef"
    assert module.history_group("m4a-test-fedcba9876543210-r0001-s00000001") == (
        "fedcba9876543210")


def synthetic_rows(groups=30, windows=6):
    rows = []
    for group in range(groups):
        for window in range(windows):
            items = [f"{group}-{window}-{index}" for index in range(20)]
            rows.append((f"playlist{group}:joint5:{window}", items))
    return rows


def test_donors_are_a_cross_group_permutation():
    rows = synthetic_rows()
    donors = module.build_history_donors(rows, reference_items=15, seed=42)
    assert set(donors) == {row_id for row_id, _ in rows}
    assigned = [entry["donor"] for entry in donors.values()]
    assert sorted(assigned) == sorted(donors)
    by_id = dict(rows)
    for row_id, entry in donors.items():
        assert module.history_group(entry["donor"]) != module.history_group(row_id)
        assert entry["items"] == by_id[entry["donor"]][:15]


def test_donors_are_deterministic_per_seed():
    rows = synthetic_rows()
    first = module.build_history_donors(rows, reference_items=15, seed=7)
    second = module.build_history_donors(rows, reference_items=15, seed=7)
    other = module.build_history_donors(rows, reference_items=15, seed=8)
    assert first == second
    assert first != other


def test_donors_reject_a_dominant_group():
    rows = [(f"big:joint5:{index}", [str(index)] * 20) for index in range(10)]
    rows.append(("small:joint5:0", ["x"] * 20))
    try:
        module.build_history_donors(rows, reference_items=15, seed=1)
    except ValueError as exc:
        assert "no cross-group" in str(exc)
    else:
        raise AssertionError("Expected an impossible donor assignment to be rejected")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
