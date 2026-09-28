"""Dependency-light checks for prepared-dataset manifest validation."""

from __future__ import annotations

from prepared_data import (
    EXPECTED_SPLIT_COUNTS,
    PREPARED_DATA_VERSION,
    expected_split_counts,
    validate_prepared_manifest,
)
from shared.schema import SCHEMA_VERSION, TOKEN_LAYOUT


class FakeTokenizer:
    tokens_per_item = TOKEN_LAYOUT.tokens_per_item


class FakeDataset:
    dataset_card = {"wp_c_split_counts": {"train": 231422, "test": 4725}}


def valid_manifest():
    return {
        "prepared_data_version": PREPARED_DATA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "token_layout": {
            "tokens_per_item": 13,
            "runtime_vocab_size": TOKEN_LAYOUT.runtime_vocab_size,
            "model_length": 262,
        },
    }


def test_valid_manifest_matches_frozen_protocol():
    validate_prepared_manifest(
        valid_manifest(),
        {"seq_len": 20, "protocol": {}},
        FakeTokenizer(),
    )


def test_wrong_split_count_is_rejected():
    manifest = valid_manifest()
    manifest["split_counts"]["test"] = 468
    try:
        validate_prepared_manifest(
            manifest, {"seq_len": 20, "protocol": {}}, FakeTokenizer())
    except ValueError as exc:
        assert "split counts drifted" in str(exc)
    else:
        raise AssertionError("Expected drifting prepared split counts to fail")


def test_dataset_specific_counts_are_supported():
    dataset = FakeDataset()
    manifest = valid_manifest()
    manifest["split_counts"] = expected_split_counts(dataset)
    validate_prepared_manifest(
        manifest, {"seq_len": 20, "protocol": {}}, FakeTokenizer(), dataset)


def test_history_condition_must_match_the_prepared_cache():
    manifest = valid_manifest()
    manifest["history_condition"] = {"condition": "cue_only"}
    validate_prepared_manifest(
        manifest, {"seq_len": 20, "protocol": {}, "history_condition": "cue_only"},
        FakeTokenizer())
    for configured in ("full", "shuffled_cue"):
        try:
            validate_prepared_manifest(
                manifest,
                {"seq_len": 20, "protocol": {}, "history_condition": configured},
                FakeTokenizer())
        except ValueError as exc:
            assert "history condition mismatch" in str(exc)
        else:
            raise AssertionError(f"Expected {configured!r} config on a cue_only cache to fail")


def test_pre_ablation_caches_count_as_full():
    validate_prepared_manifest(
        valid_manifest(), {"seq_len": 20, "protocol": {}, "history_condition": "full"},
        FakeTokenizer())


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
