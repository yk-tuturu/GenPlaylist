#!/usr/bin/env python3
"""Random-retrieval baseline for the MERT proxy metrics.

For every test history in a prediction result, draw 5 distinct songs uniformly
from the MERT catalog and score them against that history's true next songs
with the same code as ``scripts/evaluate_mert_proxy.py``. The result gives the
floor of N1-MERT, M2M-MERT, Recall@5, and Coverage@5 on that dataset, so the
narrow range of the MERT metrics can be read against it.

- Only the target songs are read from the prediction result; its predictions
  are ignored, so any result of the right dataset works.
- ``--draws`` independent random draws (seeds ``--seed`` ... ``--seed +
  draws - 1``) are scored; the output reports each draw and their mean.
- Nothing is overwritten unless ``--force`` is given.

Example (MPD; for Music4All use the m4a-full-seed1 result and its MERT folder):

    python scripts/ablation/mert_random_baseline.py \
        --prediction-result src/03_backbone_recommender/outputs/ablation-history-cond/mpd-full/results/last-steps256-evalseed1.json \
        --mert-dir data/processed/mert-v1-95m-catalog-v1 \
        --output src/03_backbone_recommender/outputs/baselines/mpd-random-mert.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_mert_proxy import (  # noqa: E402
    EXPECTED_ITEMS,
    _bootstrap_mean_interval,
    _calculate_metrics,
)
from shared.artifacts import sha256_file  # noqa: E402

SCHEMA = "genplaylist-mert-random-baseline-v1"


def load_mert_catalog(mert_dir: Path) -> tuple[list[str], np.ndarray, Path]:
    """Load and check the catalog embeddings the same way evaluate_mert_proxy does."""
    manifest_path = mert_dir / "mert_manifest.json"
    embedding_path = mert_dir / "catalog_mert_embeddings_l2.npy"
    ids_path = mert_dir / "catalog_item_ids.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete_catalog", False):
        raise ValueError("MERT artifact is a smoke-test subset, not the complete catalog")
    if manifest["outputs"][embedding_path.name] != sha256_file(embedding_path):
        raise ValueError("MERT embedding hash differs from its manifest")
    if manifest["outputs"][ids_path.name] != sha256_file(ids_path):
        raise ValueError("MERT item-ID hash differs from its manifest")
    item_ids = [str(item_id) for item_id in json.loads(ids_path.read_text("utf-8"))]
    embeddings_l2 = np.load(embedding_path, allow_pickle=False).astype(np.float32)
    if len(item_ids) != len(embeddings_l2) or len(set(item_ids)) != len(item_ids):
        raise ValueError("MERT catalog IDs are not unique and aligned with embedding rows")
    return item_ids, embeddings_l2, manifest_path


def random_predictions(item_ids: list[str], histories: int, seed: int) -> np.ndarray:
    """5 distinct uniformly random catalog songs per history."""
    rng = np.random.default_rng(seed)
    catalog = np.asarray(item_ids, dtype=str)
    rows = np.stack([
        rng.choice(len(catalog), size=EXPECTED_ITEMS, replace=False)
        for _ in range(histories)
    ])
    return catalog[rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prediction-result", type=Path, required=True,
                        help="Any evaluation result of the dataset; only its targets are used")
    parser.add_argument("--mert-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.draws <= 0:
        raise ValueError("--draws must be positive")
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to overwrite")
    prediction_path = args.prediction_result.expanduser().resolve()
    mert_dir = args.mert_dir.expanduser().resolve()

    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    target_ids = np.asarray(payload["predictions"]["target_item_ids"], dtype=str)
    item_ids, embeddings_l2, manifest_path = load_mert_catalog(mert_dir)
    item_to_row = {item_id: row for row, item_id in enumerate(item_ids)}

    draws = []
    for draw in range(args.draws):
        seed = args.seed + draw
        predictions = random_predictions(item_ids, len(target_ids), seed)
        metrics, per_history = _calculate_metrics(
            predictions, target_ids, item_to_row, embeddings_l2)
        draws.append({
            "seed": seed,
            "metrics": metrics,
            "confidence_intervals_95": {
                name: _bootstrap_mean_interval(
                    values, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
                for name, values in per_history.items()
            },
        })

    names = ("n1_mert", "recall_at_5", "m2m_mert", "coverage_at_5")
    summary = {
        name: {
            "mean": float(np.mean([d["metrics"][name] for d in draws])),
            "min": float(np.min([d["metrics"][name] for d in draws])),
            "max": float(np.max([d["metrics"][name] for d in draws])),
        }
        for name in names
    }
    result = {
        "result_schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "uniform random 5 distinct catalog songs per history",
        "targets_from": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        "mert_artifact": {"path": str(mert_dir), "manifest_sha256": sha256_file(manifest_path)},
        "examples": int(len(target_ids)),
        "catalog_items": len(item_ids),
        "summary_over_draws": summary,
        "draws": draws,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    for name in names:
        s = summary[name]
        print(f"{name:14s} mean {s['mean']:.4f}  (draws {s['min']:.4f}-{s['max']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
