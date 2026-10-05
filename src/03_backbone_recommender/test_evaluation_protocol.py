"""Dependency-light checks for official stochastic evaluation settings."""

from evaluation_protocol import (
    OFFICIAL_EVALUATION_PROTOCOL, history_mask_positions, select_test_subset)


def test_official_stochastic_eval_settings_are_frozen():
    protocol = OFFICIAL_EVALUATION_PROTOCOL
    assert protocol.as_dict() == {
        "sampling_steps": 256,
        "seed": 1,
        "use_ema": True,
    }
    protocol.validate_config({
        "seed": 1,
        "sampling": {"steps": 256},
        "eval": {"disable_ema": False},
    })
    try:
        protocol.validate_config({
            "seed": 1,
            "sampling": {"steps": 25},
            "eval": {"disable_ema": False},
        })
    except ValueError as exc:
        assert "sampling.steps" in str(exc)
    else:
        raise AssertionError("Expected official sampling-step drift to be rejected")

    protocol.validate_config({
        "seed": 99,
        "sampling": {"steps": 1},
        "eval": {"disable_ema": True},
    }, allow_override=True)


def test_test_subset_is_fixed_sorted_and_spread_over_the_test_set():
    subset = select_test_subset(19771, 2000, 0)
    assert subset == select_test_subset(19771, 2000, 0)
    assert subset != select_test_subset(19771, 2000, 1)
    assert len(subset) == len(set(subset)) == 2000
    assert subset == sorted(subset)
    assert 0 <= subset[0] and subset[-1] < 19771
    # A random subset, not the first rows of the file.
    assert subset != list(range(2000))
    assert sum(index >= 19771 // 2 for index in subset) > 800


def test_test_subset_keeps_all_rows_when_large_enough_and_rejects_nonsense():
    assert select_test_subset(941, 941, 0) == list(range(941))
    assert select_test_subset(941, 5000, 3) == list(range(941))
    for total, size in ((0, 10), (10, 0), (10, -1)):
        try:
            select_test_subset(total, size, 0)
        except ValueError:
            continue
        raise AssertionError(f"total={total}, size={size} should be rejected")


def test_history_mask_blanks_only_the_oldest_reference_payloads():
    # 8-cue layout: 13 tokens per item, BOI + 4 semantic + 8 cues.
    semantic, cues = history_mask_positions(
        reference_items=15, tokens_per_item=13, semantic_tokens=4, keep=5)
    assert len(semantic) == 10 * 4 and len(cues) == 10 * 8
    assert semantic[:4] == [2, 3, 4, 5] and cues[:8] == list(range(6, 14))
    boi_positions = {1 + block * 13 for block in range(15)}
    assert not boi_positions & (set(semantic) | set(cues))
    # The last blanked block is block 9; blocks 10-14 (the most recent five)
    # and EOS are untouched.
    assert max(semantic + cues) == 1 + 9 * 13 + 12
    context_length = 2 + 15 * 13
    assert max(semantic + cues) < context_length - 1 - 5 * 13


def test_history_mask_edges_and_zero_cue_layout():
    assert history_mask_positions(
        reference_items=15, tokens_per_item=13, semantic_tokens=4, keep=15) == ([], [])
    semantic, cues = history_mask_positions(
        reference_items=15, tokens_per_item=13, semantic_tokens=4, keep=1)
    assert len(semantic) == 14 * 4 and len(cues) == 14 * 8
    semantic, cues = history_mask_positions(
        reference_items=15, tokens_per_item=5, semantic_tokens=4, keep=10)
    assert len(semantic) == 5 * 4 and cues == []
    for keep in (0, 16):
        try:
            history_mask_positions(
                reference_items=15, tokens_per_item=13, semantic_tokens=4, keep=keep)
        except ValueError:
            continue
        raise AssertionError(f"keep={keep} should be rejected")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
