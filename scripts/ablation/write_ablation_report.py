#!/usr/bin/env python3
"""Render the JSON from collect_ablation_metrics.py as a Markdown report.

The report states what each ablation measures and lists the numbers exactly
as collected: per-variant means and spreads over seeds, differences from the
reference variant, and (for the training schedule) the training-curve points.
It draws no conclusions.

Usage:
    python scripts/ablation/write_ablation_report.py ablation_metrics.json [-o REPORT.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

DATASET_NAMES = {"mpd": "MPD", "m4a": "Music4All-Onion"}

# (key, column label, decimals)
HEADLINE = [
    ("n1_mert", "N1-MERT", 4),
    ("recall_at_5", "Recall@5", 4),
    ("m2m_mert", "M2M-MERT", 4),
    ("coverage_at_5", "Coverage@5", 4),
]
SECONDARY = [
    ("m2m_hit", "Hit", 4),
    ("m2m_f1", "Item F1", 4),
    ("m2m_cosine", "Cosine", 4),
    ("m2m_cue_f1", "Cue F1", 4),
    ("m2m_unique_ratio", "Unique items", 4),
    ("m2m_cue_unique_ratio", "Unique cues", 4),
    ("tokens_per_sequence", "Tokens/seq", 0),
    ("generation_seconds_per_history", "Gen s/history", 4),
]

ABLATION_TEXT = {
    "history_cond": {
        "title": "Ablation 1: conditioning channels",
        "aim": (
            "Measures which part of the 15-item reference history drives the predictions: "
            "the latent musical references (RVQ and conflict tokens), the creative cues, or "
            "only the presence of extra cue tokens. Every variant uses the 8-cue layout "
            "(262-token sequences); only the reference items change, and the five target "
            "items always carry their real tokens."
        ),
        "variants": {
            "full": "real latent references and real cues",
            "latent_only": "real latent references, cues replaced by the cue-null token",
            "cue_only": "real cues, latent references replaced by the RVQ-null token",
            "shuffled_cue": "real latent references, cues taken from a donor history",
        },
        "comparisons": [
            ("Full vs shuffled-cue", "whether cue semantics matter beyond extra conditioning tokens"),
            ("Full vs latent-only", "whether cues add anything over latent references"),
            ("Full vs cue-only", "whether latent references add anything over cues"),
            ("Latent-only vs cue-only", "which channel carries more preference signal"),
        ],
    },
    "cue_budget": {
        "title": "Ablation 2: cue budget",
        "aim": (
            "Measures whether the relevance ranking of cues matters, and whether eight cues "
            "per song balance information against conditioning length. A separate model is "
            "trained per budget. Ranked-8 is the Full model of ablation 1."
        ),
        "variants": {
            "cues0": "no cues (102 tokens per sequence)",
            "cues4": "ranked cues 1-4 (182 tokens)",
            "cues8": "ranked cues 1-8 (262 tokens); the Full model of ablation 1",
            "cues8random": "8 random cues of the 16 stored, kept in rank order (262 tokens)",
            "cues16": "all 16 ranked cues (422 tokens)",
        },
        "comparisons": [
            ("Ranked-8 vs random-8", "whether relevance ranking matters"),
            ("0 vs 4 vs 8 vs 16", "how much each additional block of cues adds"),
            ("Metric change vs tokens and seconds per history", "whether 8 is a reasonable balance"),
        ],
        "notes": [
            "Cue F1 is only comparable within a budget.",
            "Ranked-8 results were evaluated before timing was logged, so its tokens and seconds are blank.",
        ],
    },
    "training_schedule": {
        "title": "Ablation 3: training schedule",
        "aim": (
            "Measures whether warming up the cue loss helps compared with fixed loss weights "
            "from the start, and whether any effect comes from training stability or only "
            "from training time. Warm-up vs fixed is the main comparison; uniform also "
            "removes the role weights."
        ),
        "variants": {
            "warmup": "role weights; cue weight 0.1 until step 1,000, then linear to 1.0 by step 5,000",
            "fixed": "role weights at their final values from step 0",
            "uniform": "every target token weighted 1.0",
        },
        "comparisons": [
            ("Final official metrics", "whether warm-up ends with a better score"),
            ("Training curves", "whether fixed looks like warm-up shifted earlier (a training-time effect only)"),
            ("Spread over seeds", "whether one schedule is less stable"),
        ],
        "notes": [
            "Curve points use raw (non-EMA) weights and are not official-protocol results.",
        ],
    },
}


def fmt(value, decimals: int) -> str:
    if value is None:
        return "-"
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def fmt_signed(value, decimals: int) -> str:
    if value is None:
        return "-"
    return f"{value:+.{decimals}f}"


def mean_std(summary: dict | None, decimals: int) -> str:
    if not summary:
        return "-"
    text = fmt(summary["mean"], decimals)
    if summary.get("std") is not None:
        text += f" ± {fmt(summary['std'], decimals)}"
    return text


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def seeds_label(seeds: list[int]) -> str:
    return ", ".join(map(str, seeds)) if seeds else "none"


def metric_table(variants: dict, metrics: list) -> list[str]:
    header = ["Variant", "Seeds"] + [label for _, label, _ in metrics]
    rows = [
        [f"`{name}`", seeds_label(v["seeds"])]
        + [mean_std(v["summary"].get(key), d) for key, _, d in metrics]
        for name, v in variants.items()
    ]
    return table(header, rows)


def delta_table(variants: dict, reference: str, metrics: list) -> list[str]:
    header = ["Variant", "Paired seeds"] + [label for _, label, _ in metrics]
    rows = []
    for name, v in variants.items():
        comp = v.get("vs_reference")
        if name == reference or not comp:
            continue
        paired = next((c["paired_seed_diffs"] for c in comp.values() if c["paired_seed_diffs"]), {})
        cells = []
        for key, _, d in metrics:
            c = comp.get(key)
            if not c:
                cells.append("-")
                continue
            cell = fmt_signed(c["diff_of_means"], d)
            if c["relative_diff"] is not None:
                cell += f" ({c['relative_diff'] * 100:+.1f}%)"
            cells.append(cell)
        rows.append([f"`{name}`", seeds_label(sorted(int(s) for s in paired))] + cells)
    return table(header, rows) if rows else []


def per_seed_table(variants: dict, metrics: list) -> list[str]:
    header = ["Variant", "Seed"] + [label for _, label, _ in metrics] + ["Recall@5 95% CI"]
    rows = []
    for name, v in variants.items():
        for run in v["runs"]:
            off = run.get("official")
            if not off:
                continue
            ci = off.get("confidence_intervals_95", {}).get("recall_at_5")
            ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "-"
            rows.append(
                [f"`{name}`", str(run["seed"])]
                + [fmt(off["metrics"].get(key), d) for key, _, d in metrics]
                + [ci_text]
            )
    return table(header, rows)


def curve_tables(variants: dict) -> list[str]:
    """Training-curve points (step-K checkpoints), one table per evaluation setting."""
    groups: dict[tuple, dict] = {}
    for name, v in variants.items():
        for run in v["runs"]:
            for ev in run.get("other_evaluations", []):
                tags = ev["tags"]
                if tags["step"] is None or tags["history_length"] or tags["test_contexts"]:
                    continue
                subset = tags["test_subset"]
                key = (tags["sampling_steps"], tags["ema"], subset["size"] if subset else None)
                groups.setdefault(key, {}).setdefault((name, run["seed"]), {})[tags["step"]] = ev["metrics"]

    lines = []
    for (sampling_steps, ema, subset), runs in sorted(groups.items(), key=lambda kv: (kv[0][2] or 0, kv[0][0])):
        steps = sorted({s for points in runs.values() for s in points})
        weights = "EMA" if ema else "raw"
        test = f"{subset}-history test subset" if subset else "full test set"
        lines += ["", f"**{sampling_steps} sampling steps, {weights} weights, {test}** (Hit / Cue F1)", ""]
        header = ["Variant", "Seed"] + [f"{s:,}" for s in steps]
        rows = []
        for (name, seed), points in sorted(runs.items(), key=lambda kv: (list(variants).index(kv[0][0]), kv[0][1])):
            cells = []
            for s in steps:
                m = points.get(s)
                cells.append(f"{fmt(m.get('m2m_hit'), 4)} / {fmt(m.get('m2m_cue_f1'), 4)}" if m else "-")
            rows.append([f"`{name}`", str(seed)] + cells)
        lines += table(header, rows)
    return lines


def render(report: dict) -> str:
    out = [
        "# GenPlaylist ablation results",
        "",
        f"Generated from `{Path(report['results_root']).name}` "
        f"(collected {report['generated_utc']}).",
        "",
        "All numbers come from the official evaluation (final EMA checkpoint, 256 sampling "
        "steps, sampling seed 1, full test set). Values are the mean ± sample standard "
        "deviation over the seeds listed; a single seed has no spread. Differences are "
        "variant minus reference, as a difference of means (relative change in brackets).",
        "",
        "Metrics: N1-MERT, Recall@5, M2M-MERT, and Coverage@5 are the MERT-proxy paper "
        "metrics. Hit, item F1, cosine, cue F1, and unique ratios are the exact-match "
        "metrics of the 5x5 Hungarian matching. Generation seconds per history are only "
        "comparable between runs on the same GPU type.",
    ]

    for key, ablation in report["ablations"].items():
        text = ABLATION_TEXT.get(key, {"title": key, "aim": ablation.get("question", ""), "variants": {}})
        ref = ablation["reference_variant"]
        out += ["", f"## {text['title']}", "", "### What it measures", "", text["aim"], ""]
        out += table(
            ["Variant", "Definition"],
            [[f"`{n}`" + (" (reference)" if n == ref else ""), d] for n, d in text["variants"].items()],
        )
        if text.get("comparisons"):
            out += ["", "| Comparison | Measures |", "|---|---|"]
            out += [f"| {c} | {m} |" for c, m in text["comparisons"]]
        for note in text.get("notes", []):
            out += ["", f"> {note}"]

        order = list(DATASET_NAMES)
        datasets = sorted(ablation["datasets"].items(),
                          key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))
        for dataset, ds in datasets:
            variants = ds["variants"]
            out += ["", f"### {DATASET_NAMES.get(dataset, dataset)}", ""]
            out += ["#### Headline metrics", ""] + metric_table(variants, HEADLINE)
            out += ["", "#### Exact-match and cost metrics", ""] + metric_table(variants, SECONDARY)
            deltas = delta_table(variants, ref, HEADLINE + SECONDARY[:4])
            if deltas:
                out += ["", f"#### Difference from `{ref}`", ""] + deltas
            out += ["", "#### Per seed", ""] + per_seed_table(variants, HEADLINE)
            if key == "training_schedule":
                curves = curve_tables(variants)
                if curves:
                    out += ["", "#### Training curves"] + curves
            if ds.get("missing_results"):
                out += ["", "#### Not yet evaluated", "",
                        "Finished training but has no official result:", ""]
                out += [f"- `{name}`" for name in ds["missing_results"]]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", type=Path, help="JSON written by collect_ablation_metrics.py")
    parser.add_argument("-o", "--output", type=Path, help="output Markdown path (default: stdout)")
    args = parser.parse_args(argv)

    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    text = render(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
