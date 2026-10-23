#!/usr/bin/env python3
"""Collect the key metrics of ablations 1-3 into one JSON file for comparison.

Reads every model's ``results/*.json`` under an outputs root (the live
``src/03_backbone_recommender/outputs`` tree or an extracted results bundle)
and writes, per ablation, dataset, and variant:

- per-seed official metrics (``last-steps256-evalseed1.json`` and its
  ``-mert.json``), with MERT 95% bootstrap intervals, tokens per sequence,
  and generation seconds per history
- mean, std (sample), min, and max over seeds
- the difference from the reference variant (Full, ranked-8, warm-up), both
  as a difference of means and as per-seed paired differences
- every other evaluation of the run (training-curve points, model-study
  history lengths and sampling steps, inference-time condition swaps), tagged
  by what made it unofficial
- finished runs (``runs.csv``) that have no official result yet, and results
  older than their run's finish time (stale)

Usage:
    python scripts/ablation/collect_ablation_metrics.py [ROOT] [-o OUT.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "src" / "03_backbone_recommender" / "outputs"

OFFICIAL_STEM = "last-steps256-evalseed1"

# Headline metrics, in reporting order. "mert" metrics come from the -mert.json.
MERT_METRICS = ["n1_mert", "recall_at_5", "m2m_mert", "coverage_at_5", "unique_predicted_items"]
EXACT_METRICS = [
    "m2m_hit", "m2m_f1", "m2m_cosine", "m2m_cue_f1",
    "m2m_unique_ratio", "m2m_cue_unique_ratio", "m2m_exact_matches",
]
COST_METRICS = ["tokens_per_sequence", "generation_seconds_per_history"]
SUMMARY_METRICS = MERT_METRICS + EXACT_METRICS + COST_METRICS

ABLATIONS = {
    "history_cond": {
        "folder": "ablation-history-cond",
        "question": "Which part of the history drives predictions: latent references, cues, or extra cue tokens?",
        "variants": ["full", "latent_only", "cue_only", "shuffled_cue"],
        "reference": "full",
    },
    "cue_budget": {
        "folder": "ablation-cue-budget",
        "question": "Does cue ranking matter, and is 8 cues a good balance of information and sequence length?",
        "variants": ["cues0", "cues4", "cues8", "cues8random", "cues16"],
        "reference": "cues8",
        # Ranked-8 is the history-conditioning Full model (see common.sh).
        "borrowed": {"cues8": ("ablation-history-cond", "full")},
    },
    "training_schedule": {
        "folder": "ablation-training-schedule",
        "question": "Does warming up the cue loss help over fixed loss weights from the start?",
        "variants": ["warmup", "fixed", "uniform"],
        "reference": "warmup",
    },
}

RUN_RE = re.compile(r"^(?P<dataset>mpd|m4a)-(?P<variant>.+?)(?:-seed(?P<seed>\d+))?$")
RESULT_RE = re.compile(
    r"^(?:on-(?P<on>[a-z_]+)-)?"
    r"(?P<checkpoint>last|step-\d+)-steps(?P<steps>\d+)-evalseed(?P<evalseed>\d+)"
    r"(?P<suffix>(?:-raw|-n\d+s\d+|-hist\d+)*)$"
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_runs_csv(root: Path) -> dict[str, dict]:
    """Last runs.csv row per output name (a resumed run adds another row)."""
    path = root / "runs.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["output_name"]: row for row in csv.DictReader(f)}


def parse_result_stem(stem: str) -> dict | None:
    m = RESULT_RE.match(stem)
    if not m:
        return None
    suffix = m["suffix"]
    subset = re.search(r"-n(\d+)s(\d+)", suffix)
    hist = re.search(r"-hist(\d+)", suffix)
    ckpt = m["checkpoint"]
    return {
        "checkpoint": ckpt,
        "step": None if ckpt == "last" else int(ckpt.split("-")[1]),
        "sampling_steps": int(m["steps"]),
        "eval_seed": int(m["evalseed"]),
        "ema": "-raw" not in suffix,
        "test_subset": {"size": int(subset[1]), "seed": int(subset[2])} if subset else None,
        "history_length": int(hist[1]) if hist else None,
        "test_contexts": m["on"],
    }


def extract_evaluation(result_path: Path) -> dict:
    """Metrics and metadata of one prediction result plus its -mert.json, if any."""
    result = load_json(result_path)
    ev = result.get("evaluation", {})
    timing = ev.get("timing") or {}
    metrics = {k: result.get("metrics", {}).get(k) for k in EXACT_METRICS}
    metrics["tokens_per_sequence"] = ev.get("tokens_per_sequence")
    metrics["generation_seconds_per_history"] = timing.get("generation_seconds_per_history")

    out = {
        "file": result_path.name,
        "official_protocol": ev.get("official_protocol"),
        "created_utc": result.get("created_utc"),
        "checkpoint_sha256": (result.get("checkpoint") or {}).get("sha256"),
        "git_commit": result.get("git_commit"),
        "test_examples": ev.get("test_examples"),
        "active_cue_tokens": ev.get("active_cue_tokens"),
        "history_condition": ev.get("history_condition"),
        "timing_device": timing.get("device"),
        "mert_file": None,
        "metrics": metrics,
        "confidence_intervals_95": {},
    }
    mert_path = result_path.with_name(result_path.stem + "-mert.json")
    if mert_path.exists():
        mert = load_json(mert_path)
        out["mert_file"] = mert_path.name
        for k in MERT_METRICS:
            metrics[k] = mert.get("metrics", {}).get(k)
        out["confidence_intervals_95"] = mert.get("confidence_intervals_95", {})
        out["official_protocol"] = bool(out["official_protocol"] and mert.get("official_evaluation"))
    else:
        for k in MERT_METRICS:
            metrics[k] = None
    return out


def collect_run(run_dir: Path, output_name: str, runs_csv: dict) -> dict:
    results_dir = run_dir / "results"
    row = runs_csv.get(output_name)
    finished = parse_time(row["finished_utc"]) if row else None
    run = {
        "output_name": output_name,
        "finished_utc": row["finished_utc"] if row else None,
        "loss_variant": row["loss_variant"] if row else None,
        "official": None,
        "other_evaluations": [],
        "warnings": [],
    }
    if not results_dir.is_dir():
        run["warnings"].append("no results/ folder")
        return run

    for path in sorted(results_dir.glob("*.json")):
        if path.stem.endswith("-mert"):
            continue  # read together with its prediction result
        tags = parse_result_stem(path.stem)
        if tags is None:
            run["warnings"].append(f"unrecognised result file {path.name}")
            continue
        ev = extract_evaluation(path)
        created = parse_time(ev["created_utc"])
        if finished and created and created < finished:
            run["warnings"].append(f"STALE: {path.name} was evaluated before the run finished")
        if path.stem == OFFICIAL_STEM:
            if not ev["official_protocol"]:
                run["warnings"].append(f"{path.name} is not marked official_protocol")
            if ev["mert_file"] is None:
                run["warnings"].append(f"{path.name} has no -mert.json")
            run["official"] = ev
        else:
            ev["tags"] = tags
            run["other_evaluations"].append(ev)

    run["other_evaluations"].sort(
        key=lambda e: (
            e["tags"]["test_contexts"] or "",
            e["tags"]["history_length"] or 0,
            e["tags"]["test_subset"] is not None,
            e["tags"]["ema"],
            e["tags"]["sampling_steps"],
            e["tags"]["step"] if e["tags"]["step"] is not None else float("inf"),
        )
    )
    if run["official"] is None:
        run["warnings"].append(f"no official result ({OFFICIAL_STEM}.json)")
    return run


def summarize(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def compare(variant: dict, reference: dict) -> dict:
    """Variant minus reference: difference of means and paired per-seed differences."""
    ref_by_seed = {r["seed"]: r for r in reference["runs"] if r["official"]}
    out = {}
    for metric in SUMMARY_METRICS:
        v_sum = variant["summary"].get(metric)
        r_sum = reference["summary"].get(metric)
        if not v_sum or not r_sum:
            continue
        paired = {}
        for run in variant["runs"]:
            ref = ref_by_seed.get(run["seed"])
            if not run["official"] or not ref:
                continue
            a = run["official"]["metrics"].get(metric)
            b = ref["official"]["metrics"].get(metric)
            if a is not None and b is not None:
                paired[str(run["seed"])] = a - b
        out[metric] = {
            "diff_of_means": v_sum["mean"] - r_sum["mean"],
            "relative_diff": (v_sum["mean"] - r_sum["mean"]) / r_sum["mean"] if r_sum["mean"] else None,
            "paired_seed_diffs": paired,
            "paired_mean_diff": statistics.fmean(paired.values()) if paired else None,
        }
    return out


def discover_runs(root: Path, folder: str) -> list[tuple[str, str, int, Path]]:
    base = root / folder
    if not base.is_dir():
        return []
    found = []
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        found.append((m["dataset"], m["variant"], int(m["seed"] or 1), run_dir))
    return found


def collect_ablation(root: Path, key: str, spec: dict, runs_csv: dict) -> dict:
    sources = [(spec["folder"], v, v) for v in spec["variants"]]
    for variant, (folder, src_variant) in spec.get("borrowed", {}).items():
        sources = [s for s in sources if s[2] != variant] + [(folder, src_variant, variant)]

    datasets: dict[str, dict] = {}
    for folder, src_variant, variant in sources:
        for dataset, found_variant, seed, run_dir in discover_runs(root, folder):
            if found_variant != src_variant:
                continue
            output_name = f"{folder}/{run_dir.name}"
            run = collect_run(run_dir, output_name, runs_csv)
            run["seed"] = seed
            if folder != spec["folder"]:
                run["borrowed_from"] = output_name
            ds = datasets.setdefault(dataset, {"variants": {}})
            ds["variants"].setdefault(variant, {"runs": []})["runs"].append(run)

    # Finished training runs (runs.csv) whose folder or official result is missing.
    for name in runs_csv:
        if not name.startswith(spec["folder"] + "/"):
            continue
        m = RUN_RE.match(name.split("/", 1)[1])
        if not m:
            continue
        dataset, variant = m["dataset"], m["variant"]
        runs = datasets.get(dataset, {}).get("variants", {}).get(variant, {}).get("runs", [])
        if not any(r["output_name"] == name and r["official"] for r in runs):
            datasets.setdefault(dataset, {"variants": {}}).setdefault("missing_results", []).append(name)

    order = {v: i for i, v in enumerate(spec["variants"])}
    for dataset, ds in datasets.items():
        ds["variants"] = dict(sorted(ds["variants"].items(), key=lambda kv: order.get(kv[0], 99)))
        for variant in ds["variants"].values():
            variant["runs"].sort(key=lambda r: r["seed"])
            official = [r for r in variant["runs"] if r["official"]]
            variant["seeds"] = [r["seed"] for r in official]
            variant["summary"] = {
                m: summarize([r["official"]["metrics"].get(m) for r in official])
                for m in SUMMARY_METRICS
            }
        ref = ds["variants"].get(spec["reference"])
        for name, variant in ds["variants"].items():
            if ref and name != spec["reference"]:
                variant["vs_reference"] = compare(variant, ref)
        ds["comparison_table"] = [
            {"variant": name, "seeds": v["seeds"]}
            | {m: (v["summary"][m] or {}).get("mean") for m in SUMMARY_METRICS}
            | {f"{m}_std": (v["summary"][m] or {}).get("std") for m in MERT_METRICS[:4]}
            for name, v in ds["variants"].items()
        ]
        if "missing_results" in ds:
            ds["missing_results"].sort()

    return {
        "folder": spec["folder"],
        "question": spec["question"],
        "reference_variant": spec["reference"],
        "datasets": dict(sorted(datasets.items())),
    }


def strip_runs(ablation: dict, keep_other: bool) -> None:
    if keep_other:
        return
    for ds in ablation["datasets"].values():
        for variant in ds["variants"].values():
            for run in variant["runs"]:
                run.pop("other_evaluations", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT,
                        help="outputs root containing ablation-*/ folders and runs.csv "
                             f"(default: {DEFAULT_ROOT.relative_to(REPO_ROOT)})")
    parser.add_argument("-o", "--output", type=Path, help="output JSON path (default: stdout)")
    parser.add_argument("--ablations", nargs="+", choices=list(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--no-other", action="store_true",
                        help="omit non-official evaluations (curves, model-study files)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    runs_csv = load_runs_csv(root)
    if not runs_csv:
        print(f"warning: no runs.csv in {root}; finished/stale checks are skipped", file=sys.stderr)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_root": str(root),
        "official_result": f"{OFFICIAL_STEM}.json (+ -mert.json)",
        "metrics": {
            "mert": MERT_METRICS,
            "exact_match": EXACT_METRICS,
            "cost": COST_METRICS,
            "note": "std is the sample std over seeds; generation seconds are only comparable on the same GPU type",
        },
        "ablations": {},
    }
    for key in args.ablations:
        ablation = collect_ablation(root, key, ABLATIONS[key], runs_csv)
        strip_runs(ablation, keep_other=not args.no_other)
        report["ablations"][key] = ablation

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    for key, ablation in report["ablations"].items():
        for dataset, ds in ablation["datasets"].items():
            print(f"[{key}/{dataset}] " + ", ".join(
                f"{n}: {len(v['seeds'])} seed(s)" for n, v in ds["variants"].items()), file=sys.stderr)
            for name in ds.get("missing_results", []):
                print(f"  finished but no official result: {name}", file=sys.stderr)
            for v in ds["variants"].values():
                for run in v["runs"]:
                    for w in run["warnings"]:
                        print(f"  {run['output_name']}: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
