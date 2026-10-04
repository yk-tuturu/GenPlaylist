"""Tests for the training-schedule curve exporter (no tensorboard needed)."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "export_training_curves", HERE / "export_training_curves.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def result_payload(*, ema=True, steps=256, examples=941, subset=None, official=True):
    return {
        "evaluation": {"ema_enabled": ema, "sampling_steps": steps,
                       "test_examples": examples, "test_subset": subset,
                       "official_protocol": official},
        "metrics": {"m2m_recall": 0.02, "m2m_cue_f1": 0.28, "m2m_cosine": 0.58},
    }


def test_run_names_are_parsed():
    assert module.parse_run_name("m4a-fixed-seed2") == {
        "dataset": "m4a", "schedule": "fixed", "seed": 2}
    assert module.parse_run_name("mpd-warmup-seed10")["seed"] == 10
    assert module.parse_run_name("curves") is None


def test_eval_points_carry_step_and_settings():
    curve = module.eval_point(
        "m4a-uniform-seed1", Path("step-5000-steps64-evalseed1-raw-n2000s0.json"),
        result_payload(ema=False, steps=64, examples=2000,
                       subset={"seed": 0, "size": 2000}, official=False))
    assert (curve["step"], curve["sampling_steps"], curve["weights"]) == (5000, 64, "raw")
    assert (curve["test_examples"], curve["subset_seed"], curve["official"]) == (2000, 0, False)
    assert curve["schedule"] == "uniform" and curve["m2m_recall"] == 0.02

    final = module.eval_point(
        "mpd-warmup-seed1", Path("last-steps256-evalseed1.json"), result_payload())
    assert (final["checkpoint"], final["step"], final["weights"]) == ("last", 20000, "ema")
    assert final["official"] is True

    assert module.eval_point(
        "mpd-warmup-seed1", Path("last-steps256-evalseed1-mert.json"), {}) is None
    assert module.eval_point("mpd-warmup-seed1", Path("notes.json"), {}) is None


def test_scalar_events_become_one_row_per_step():
    columns, rows = module.wide_rows([
        ("train/layer_nll/cues", 10, 3.0), ("train/grad_norm", 10, 0.7),
        ("train/layer_nll/cues", 20, 2.5), ("train/layer_nll/cues", 10, 2.9),
    ])
    assert columns == ["train/grad_norm", "train/layer_nll/cues"]
    assert rows == [
        {"step": 10, "train/layer_nll/cues": 2.9, "train/grad_norm": 0.7},
        {"step": 20, "train/layer_nll/cues": 2.5},
    ]


def test_cli_exports_eval_points_for_every_run():
    with tempfile.TemporaryDirectory() as tmp:
        outputs = Path(tmp)
        experiment = outputs / "ablation-training-schedule"
        for run, files in {
            "mpd-warmup-seed1": {
                "last-steps256-evalseed1.json": result_payload(),
                "last-steps256-evalseed1-mert.json": {"metrics": {}},
                "step-1000-steps64-evalseed1-raw.json": result_payload(
                    ema=False, steps=64, official=False),
            },
            "mpd-fixed-seed1": {
                "step-1000-steps64-evalseed1-raw.json": result_payload(
                    ema=False, steps=64, official=False),
            },
        }.items():
            (experiment / run / "results").mkdir(parents=True)
            for name, payload in files.items():
                (experiment / run / "results" / name).write_text(json.dumps(payload))
        (experiment / "not-a-run").mkdir()

        subprocess.run(
            [sys.executable, str(HERE / "export_training_curves.py"),
             "--outputs", str(outputs), "--skip-scalars"],
            check=True, capture_output=True)
        with (experiment / "curves" / "eval_points.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [(row["run"], row["step"], row["weights"]) for row in rows] == [
            ("mpd-fixed-seed1", "1000", "raw"),
            ("mpd-warmup-seed1", "1000", "raw"),
            ("mpd-warmup-seed1", "20000", "ema"),
        ]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
