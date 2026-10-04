#!/usr/bin/env python3
"""Derive a random-subset cue table for the cue-budget ablation.

WP-C always uses the first ``--active-cues`` entries of each song's stored
ranked cue list. This script writes a new cue folder in which, for every song,
the first ``--select`` entries are a random subset of its stored cues and the
remaining stored cues follow. Preparing that folder with
``--active-cues <select>`` therefore gives "N random cues from the same stored
set" without any change to the tokenizer.

- The subset is drawn once per song from ``sha256(seed, item_id)``, so it is
  fixed across histories, splits, and reruns.
- The selected cues keep their original rank order, and so do the remaining
  ones, so only *which* cues are used differs from the ranked table.
- ``cue_vocab.json`` is copied byte-for-byte, so cue IDs keep their meaning.
- ``cue_manifest.json`` keeps the source fields the tokenizer checks, points
  ``item2cues_sha256`` at the new table, and records how it was derived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "genplaylist-random-cue-table-v1"
CUE_FILES = ("item2cues.json", "cue_vocab.json", "cue_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_positions(item_id: str, stored: int, select: int, seed: int) -> list[int]:
    """Rank positions (0-based, ascending) of the cues chosen for one song."""
    if not 0 < select <= stored:
        raise ValueError(f"Cannot select {select} of {stored} stored cues")
    digest = hashlib.sha256(f"{seed}\0{item_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return sorted(int(position) for position in rng.choice(stored, size=select, replace=False))


def reorder_cues(cues: list[int], positions: list[int]) -> list[int]:
    """Selected cues first, then the rest, both in their original rank order."""
    chosen = set(positions)
    return ([cues[index] for index in positions]
            + [cue for index, cue in enumerate(cues) if index not in chosen])


def build_random_table(
    item2cues: dict[str, list[int]], *, select: int, seed: int,
) -> dict[str, list[int]]:
    lengths = {len(cues) for cues in item2cues.values()}
    if len(lengths) != 1:
        raise ValueError(f"Songs store different cue counts: {sorted(lengths)}")
    stored = lengths.pop()
    return {
        str(item_id): reorder_cues(
            [int(cue) for cue in cues], selected_positions(str(item_id), stored, select, seed))
        for item_id, cues in item2cues.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cue-dir", type=Path, required=True,
                        help="Ranked cue folder (item2cues.json, cue_vocab.json, cue_manifest.json)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--select", type=int, default=8,
                        help="Random cues placed first; prepare with --active-cues equal to this")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.cue_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use a new folder")
    for name in CUE_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    item2cues = json.loads((source / "item2cues.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "cue_manifest.json").read_text(encoding="utf-8"))
    stored = int(manifest.get("stored_cues_per_item", manifest.get("cues_per_item", 0)))
    table = build_random_table(item2cues, select=args.select, seed=args.seed)
    if {len(cues) for cues in table.values()} != {stored}:
        raise ValueError(f"Cue manifest declares {stored} stored cues per song")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temp / "item2cues.json").write_text(
            json.dumps(table, ensure_ascii=False) + "\n", encoding="utf-8")
        shutil.copyfile(source / "cue_vocab.json", temp / "cue_vocab.json")
        derived = dict(manifest)
        derived.pop("item2cue_scores_sha256", None)
        derived["item2cues_sha256"] = sha256_file(temp / "item2cues.json")
        derived["derived_cue_table"] = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "random subset per song, selected first, original rank order kept",
            "select": args.select,
            "seed": args.seed,
            "source_dir": str(source),
            "source_item2cues_sha256": sha256_file(source / "item2cues.json"),
            "source_cue_manifest_sha256": sha256_file(source / "cue_manifest.json"),
        }
        (temp / "cue_manifest.json").write_text(
            json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    changed = sum(
        table[item_id][:args.select] != [int(cue) for cue in cues[:args.select]]
        for item_id, cues in item2cues.items())
    print(json.dumps({
        "output": str(output),
        "songs": len(table),
        "stored_cues_per_song": stored,
        "select": args.select,
        "seed": args.seed,
        "songs_whose_first_cues_changed": changed,
        "item2cues_sha256": sha256_file(output / "item2cues.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
