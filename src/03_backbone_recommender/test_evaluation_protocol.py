"""Dependency-light checks for official stochastic evaluation settings."""

from evaluation_protocol import OFFICIAL_EVALUATION_PROTOCOL, select_test_subset


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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
