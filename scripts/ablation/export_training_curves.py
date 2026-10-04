#!/usr/bin/env python3
"""Export training curves of the training-schedule ablation to CSV.

For every run folder ``<outputs>/<experiment>/<dataset>-<schedule>-seed<N>``:

- ``train_scalars.csv`` (one per run, in ``--output-dir/<run>/``): every
  TensorBoard scalar under ``train/`` plus ``trainer/loss``, one row per
  logged step. Compare schedules on the *unweighted* ``train/layer_nll/*``
  columns and on ``train/grad_norm*``; ``trainer/loss`` is weighted by the
  schedule itself and is not comparable across schedules.
- ``eval_points.csv`` (one for all runs): every evaluation result in the
  runs' ``results/`` folders, with its checkpoint step, settings (sampling
  steps, EMA or raw, test subset, official flag) and WP-C metrics.

Reading TensorBoard logs needs the ``tensorboard`` package (in
requirements.txt); ``--skip-scalars`` exports only the evaluation points.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RUN_PATTERN = re.compile(r"^(?P<dataset>[a-z0-9]+)-(?P<schedule>[a-z_]+)-seed(?P<seed>\d+)$")
CKPT_PATTERN = re.compile(r"^(?P<label>last|step-(?P<step>\d+))-steps(?P<steps>\d+)-evalseed\d+")
EVAL_METRICS = (
    "m2m_recall", "m2m_hit", "m2m_cosine", "m2m_cue_f1",
    "m2m_unique_ratio", "m2m_exact_matches",
)
SCALAR_PREFIXES = ("train/", "trainer/loss")


def parse_run_name(name: str) -> dict | None:
    match = RUN_PATTERN.match(name)
    if match is None:
        return None
    return {**match.groupdict(), "seed": int(match["seed"])}


def eval_point(run: str, result_path: Path, payload: dict, max_steps: int = 20000) -> dict | None:
    """One evaluation result as a flat row; None for files that are not results."""
    match = CKPT_PATTERN.match(result_path.name)
    if match is None or result_path.name.endswith("-mert.json"):
        return None
    evaluation = payload.get("evaluation", {})
    subset = evaluation.get("test_subset") or {}
    step = int(match["step"]) if match["step"] else max_steps
    row = {
        "run": run,
        **(parse_run_name(run) or {}),
        "checkpoint": match["label"],
        "step": step,
        "sampling_steps": int(evaluation.get("sampling_steps", match["steps"])),
        "weights": "ema" if evaluation.get("ema_enabled", True) else "raw",
        "test_examples": evaluation.get("test_examples"),
        "subset_seed": subset.get("seed"),
        "official": evaluation.get("official_protocol"),
        "file": result_path.name,
    }
    metrics = payload.get("metrics", {})
    row.update({name: metrics.get(name) for name in EVAL_METRICS})
    return row


def wide_rows(events: list[tuple[str, int, float]]) -> tuple[list[str], list[dict]]:
    """(tag, step, value) events -> sorted step rows; later events win."""
    by_step: dict[int, dict] = {}
    tags: set[str] = set()
    for tag, step, value in events:
        tags.add(tag)
        by_step.setdefault(int(step), {})[tag] = value
    columns = sorted(tags)
    return columns, [{"step": step, **by_step[step]} for step in sorted(by_step)]


def read_scalar_events(run_dir: Path) -> list[tuple[str, int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = []
    # A resumed run writes a second event file; read them in time order.
    files = sorted((run_dir / "tensorboard").rglob("events.out.tfevents.*"),
                   key=lambda path: path.stat().st_mtime)
    for event_file in files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if tag.startswith(SCALAR_PREFIXES):
                events.extend((tag, event.step, event.value)
                              for event in accumulator.Scalars(tag))
    return events


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--outputs", type=Path,
                        default=repo / "src" / "03_backbone_recommender" / "outputs")
    parser.add_argument("--experiment", default="ablation-training-schedule")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <outputs>/<experiment>/curves")
    parser.add_argument("--skip-scalars", action="store_true")
    args = parser.parse_args()

    root = args.outputs.expanduser().resolve() / args.experiment
    output = (args.output_dir or root / "curves").expanduser().resolve()
    runs = sorted(path for path in root.iterdir()
                  if path.is_dir() and parse_run_name(path.name)) if root.is_dir() else []
    if not runs:
        raise SystemExit(f"No <dataset>-<schedule>-seed<N> run folders under {root}")

    points = []
    for run_dir in runs:
        for result in sorted((run_dir / "results").glob("*.json")):
            row = eval_point(run_dir.name, result, json.loads(result.read_text("utf-8")))
            if row is not None:
                points.append(row)
        if not args.skip_scalars:
            columns, rows = wide_rows(read_scalar_events(run_dir))
            write_csv(output / run_dir.name / "train_scalars.csv", ["step", *columns], rows)
            print(f"{run_dir.name}: {len(rows)} logged steps, {len(columns)} scalars")

    point_columns = ["run", "dataset", "schedule", "seed", "checkpoint", "step",
                     "sampling_steps", "weights", "test_examples", "subset_seed",
                     "official", *EVAL_METRICS, "file"]
    points.sort(key=lambda row: (row["run"], row["step"], row["sampling_steps"], row["weights"]))
    write_csv(output / "eval_points.csv", point_columns, points)
    print(f"{len(points)} evaluation points -> {output / 'eval_points.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
