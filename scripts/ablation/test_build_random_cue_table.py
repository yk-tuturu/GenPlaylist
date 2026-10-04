"""Tests for the cue-budget ablation's random cue table builder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "build_random_cue_table", HERE / "build_random_cue_table.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ranked_table(songs=200, stored=16):
    return {str(1000 + song): list(range(1 + song * stored, 1 + (song + 1) * stored))
            for song in range(songs)}


def test_first_cues_are_a_rank_ordered_subset_followed_by_the_rest():
    ranked = ranked_table()
    table = module.build_random_table(ranked, select=8, seed=42)
    for item_id, cues in ranked.items():
        new = table[item_id]
        assert sorted(new) == sorted(cues)
        chosen, rest = new[:8], new[8:]
        assert chosen == sorted(chosen, key=cues.index)
        assert rest == [cue for cue in cues if cue not in chosen]


def test_selection_is_deterministic_per_seed_and_song():
    ranked = ranked_table()
    first = module.build_random_table(ranked, select=8, seed=42)
    assert first == module.build_random_table(ranked, select=8, seed=42)
    assert first != module.build_random_table(ranked, select=8, seed=43)
    # A song's subset does not depend on which other songs are in the table.
    alone = module.build_random_table({"1003": ranked["1003"]}, select=8, seed=42)
    assert alone["1003"] == first["1003"]


def test_selection_is_not_just_the_top_ranks():
    ranked = ranked_table()
    table = module.build_random_table(ranked, select=8, seed=42)
    unchanged = sum(table[item][:8] == cues[:8] for item, cues in ranked.items())
    assert unchanged < len(ranked) // 10
    positions = [index for item, cues in ranked.items()
                 for index in module.selected_positions(item, 16, 8, 42)]
    assert max(positions) == 15 and min(positions) == 0


def test_invalid_requests_are_rejected():
    for select in (0, 17):
        try:
            module.selected_positions("1", 16, select, 42)
        except ValueError:
            continue
        raise AssertionError(f"select={select} should be rejected")
    try:
        module.build_random_table({"a": [1, 2, 3], "b": [1, 2]}, select=1, seed=0)
    except ValueError as exc:
        assert "different cue counts" in str(exc)
    else:
        raise AssertionError("Uneven cue lists should be rejected")


def test_cli_writes_a_valid_cue_folder_and_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "ranked"
        source.mkdir()
        ranked = ranked_table(songs=20)
        (source / "item2cues.json").write_text(json.dumps(ranked), encoding="utf-8")
        (source / "cue_vocab.json").write_text('{"0": "<unk>"}', encoding="utf-8")
        (source / "cue_manifest.json").write_text(json.dumps({
            "schema_version": "genplaylist-v1", "wp_d_compatible": True,
            "stored_cues_per_item": 16, "item2cues_sha256": "old",
            "item2cue_scores_sha256": "scores"}), encoding="utf-8")
        output = Path(tmp) / "random8"
        command = [sys.executable, str(HERE / "build_random_cue_table.py"),
                   "--cue-dir", str(source), "--output-dir", str(output), "--seed", "7"]
        subprocess.run(command, check=True, capture_output=True)

        table = json.loads((output / "item2cues.json").read_text(encoding="utf-8"))
        assert table == module.build_random_table(ranked, select=8, seed=7)
        assert ((output / "cue_vocab.json").read_bytes()
                == (source / "cue_vocab.json").read_bytes())
        manifest = json.loads((output / "cue_manifest.json").read_text(encoding="utf-8"))
        assert manifest["wp_d_compatible"] is True
        assert manifest["stored_cues_per_item"] == 16
        assert manifest["item2cues_sha256"] == module.sha256_file(output / "item2cues.json")
        assert "item2cue_scores_sha256" not in manifest
        derived = manifest["derived_cue_table"]
        assert (derived["select"], derived["seed"]) == (8, 7)
        assert derived["source_item2cues_sha256"] == module.sha256_file(
            source / "item2cues.json")

        again = subprocess.run(command, capture_output=True, text=True)
        assert again.returncode != 0 and "Refusing to overwrite" in again.stderr


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
