"""GenPlaylist-v1 tokenizer independent of the legacy DISCO tokenizer.

This is the cross-WP boundary implementation.  It consumes already-built RVQ
semantic IDs and WP-B cue IDs; training the RVQ codebook remains an offline
artifact-building step.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

from shared.schema import (  # noqa: E402
    CLHE_EMB_DIM,
    CUE_CANDIDATES_PER_ITEM,
    CUE_TOKENS,
    RQ_N_CODEBOOKS,
    CatalogItem,
    GeneratedItem,
    TOKEN_LAYOUT,
)
from shared.artifacts import validate_catalog_alignment  # noqa: E402
from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402


# Conditioning-channel ablation. Only the reference (history) items change;
# every target item keeps its real RVQ and cue tokens.
#   full          real RVQ/conflict tokens + real cues
#   latent_only   real RVQ/conflict tokens, cues replaced by CUE_NULL_TOKEN
#   cue_only      RVQ/conflict replaced by SEMANTIC_NULL_TOKEN, real cues
#   shuffled_cue  real RVQ/conflict tokens, cues from a donor history of
#                 another playlist/user (see build_history_donors)
HISTORY_CONDITIONS = ("full", "latent_only", "cue_only", "shuffled_cue")
# Cue ID 0 is <unk>; the frozen cue table never assigns it to a real item.
CUE_NULL_TOKEN = TOKEN_LAYOUT.cue_token(0)
# The padding/BOS token. The vocabulary has no spare IDs, and MASK would be
# denoised by the sampler, so the existing padding token marks "no content".
SEMANTIC_NULL_TOKEN = 0
HISTORY_DONOR_FILE = "history_donors.json"
HISTORY_DONOR_SCHEMA = "genplaylist-history-donors-v1"


def history_group(row_id: str) -> str:
    """Return the playlist (MPD) or user (Music4All) that produced a row.

    MPD training rows are ``<playlist>:joint5:<start>`` rolling windows and
    Music4All rows are ``m4a-<split>-<user pseudonym>-r....-s....``.
    """
    base = str(row_id).split(":joint5:", 1)[0]
    if base.startswith("m4a-"):
        parts = base.split("-")
        if len(parts) < 3 or not parts[2]:
            raise ValueError(f"Malformed Music4All row ID: {row_id!r}")
        return parts[2]
    return base


def build_history_donors(
    rows: list[tuple[str, list[str]]], *, reference_items: int, seed: int,
    max_rounds: int = 1000,
) -> dict[str, dict]:
    """Assign every row a donor row from a different playlist/user.

    The assignment is a seeded permutation, so every row donates its
    reference cues exactly once. Returns ``{row_id: {"donor": donor_row_id,
    "items": donor reference item IDs}}``.
    """
    if not rows:
        return {}
    row_ids = [str(row_id) for row_id, _ in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Donor assignment requires unique row IDs")
    group_names = [history_group(row_id) for row_id in row_ids]
    group_index = {name: index for index, name in enumerate(dict.fromkeys(group_names))}
    if len(group_index) < 2:
        raise ValueError("Donor assignment needs rows from at least two groups")
    groups = np.asarray([group_index[name] for name in group_names], dtype=np.int64)
    largest = int(np.bincount(groups).max())
    if 2 * largest > len(rows):
        raise ValueError(
            f"A single playlist/user owns {largest} of {len(rows)} rows; no "
            "cross-group donor permutation exists")

    rng = np.random.default_rng(seed)
    donors = rng.permutation(len(rows))
    for _ in range(max_rounds):
        bad = np.flatnonzero(groups[donors] == groups)
        if bad.size == 0:
            break
        partners = rng.integers(0, len(rows), size=bad.size)
        for index, partner in zip(bad.tolist(), partners.tolist()):
            donors[index], donors[partner] = donors[partner], donors[index]
    else:
        raise RuntimeError("Could not find a cross-group donor permutation")

    output = {}
    for index, donor in enumerate(donors.tolist()):
        items = [str(item_id) for item_id in rows[donor][1][:reference_items]]
        if len(items) != reference_items:
            raise ValueError(
                f"Donor row {row_ids[donor]} has {len(items)} references, "
                f"expected {reference_items}")
        output[row_ids[index]] = {"donor": row_ids[donor], "items": items}
    return output


def load_history_donors(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Read a donor file into split -> row ID -> donor reference item IDs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != HISTORY_DONOR_SCHEMA:
        raise ValueError(f"Unsupported donor file schema: {payload.get('schema')!r}")
    return {
        split: {row_id: entry["items"] for row_id, entry in rows.items()}
        for split, rows in payload["splits"].items()
    }


@dataclass(frozen=True)
class TokenizedPlaylist:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    target_mask: np.ndarray
    context_emb: np.ndarray
    mu_c: np.ndarray
    sigma_c2: np.float32


class GenPlaylistTokenizer:
    """Encode a configurable-cue item representation and decode candidates."""

    bos_token = 0
    boi_token = TOKEN_LAYOUT.boi_token
    eos_token = TOKEN_LAYOUT.eos_token
    bos_token_id = bos_token
    boi_token_id = boi_token
    eos_token_id = eos_token
    mask_token_id = TOKEN_LAYOUT.mask_token
    vocab_size = TOKEN_LAYOUT.vocab_size
    tokens_per_item = TOKEN_LAYOUT.tokens_per_item

    def __init__(
        self,
        semantic_tokens: dict[str, list[int]],
        item2cues: dict[str, list[int]],
        catalog_items: list[CatalogItem],
        catalog_embeddings: np.ndarray,
        item_id_to_row: dict[str, int],
        codebook_weights: np.ndarray,
        active_cues: int = CUE_TOKENS,
    ):
        active_cues = int(active_cues)
        if not 0 <= active_cues <= CUE_CANDIDATES_PER_ITEM:
            raise ValueError(
                f"active_cues must be in [0, {CUE_CANDIDATES_PER_ITEM}], "
                f"got {active_cues}")
        self.active_cues = active_cues
        self.tokens_per_item = 1 + RQ_N_CODEBOOKS + 1 + active_cues
        self.semantic_tokens = {
            str(item_id): [int(token) for token in tokens]
            for item_id, tokens in semantic_tokens.items()
        }
        self.stored_item2cues = {
            str(item_id): [int(cue) for cue in cues]
            for item_id, cues in item2cues.items()
        }
        self.item2cues = {
            item_id: cues[:active_cues]
            for item_id, cues in self.stored_item2cues.items()
        }
        self.catalog_items = catalog_items
        self.catalog_embeddings = np.asarray(catalog_embeddings, dtype=np.float32)
        self.item_id_to_row = {str(key): int(value) for key, value in item_id_to_row.items()}
        self.codebook_weights = np.asarray(codebook_weights, dtype=np.float32)
        self.max_items = 30
        self.allow_repeated_items = False
        self.config = {"rq_codebook_size": TOKEN_LAYOUT.rq_codebook_size}
        self.dataset_dir = None
        self.history_condition = "full"
        self.history_donors = None
        self._validate_artifacts()
        self.collate_fn = {
            "train": self.collate_batch,
            "test": self.collate_batch,
        }

    @classmethod
    def from_dataset_config(cls, config, dataset) -> "GenPlaylistTokenizer":
        """Construct from the canonical repository artifacts named in config."""
        FROZEN_NEXT_SONG_PROTOCOL.validate_config(config)
        data_root = Path(dataset.dir)
        repo_root = Path(__file__).resolve().parents[2]
        cue_root = repo_root / "src" / "02_creative_cues" / "outputs" / "production" / "latest"

        def configured(name: str, default: Path) -> Path:
            value = config.get(name, None)
            return Path(value).expanduser() if value else default

        catalog_items = CatalogItem.load_catalog(str(data_root / "catalog_metadata.json"))
        mapping_path = configured(
            "item_id_to_row_path", data_root / "item_id_to_row.json")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        embeddings_path = configured(
            "catalog_embeddings_path", data_root / "catalog_item_embeddings.npy")
        tokenizer = cls.from_files(
            semantic_tokens_path=configured(
                "semantic_tokens_path", data_root / "semantic_tokens.json"),
            item2cues_path=configured("item2cues_path", cue_root / "item2cues.json"),
            cue_manifest_path=configured("cue_manifest_path", cue_root / "cue_manifest.json"),
            catalog_items=catalog_items,
            catalog_embeddings=np.load(embeddings_path, allow_pickle=False),
            item_id_to_row=mapping,
            codebook_weights_path=configured(
                "codebook_weights_path", data_root / "rvq_codebook_weights.npy"),
            active_cues=int(config.get("active_cue_tokens", CUE_TOKENS)),
        )
        tokenizer.max_items = int(config.get("seq_len", 30))
        repeat_setting = dataset.dataset_card.get("allow_repeated_items")
        if repeat_setting is None:
            repeat_setting = (
                dataset.dataset_card.get("sequence_protocol", {}).get(
                    "repeated_listens") == "retained"
            )
        tokenizer.allow_repeated_items = bool(repeat_setting)
        tokenizer.config = config
        tokenizer.dataset_dir = str(data_root)
        condition = str(config.get("history_condition", "full"))
        donors = None
        prepared_path = config.get("prepared_dataset_path", None)
        if condition == "shuffled_cue" and prepared_path:
            donor_path = Path(prepared_path).expanduser() / HISTORY_DONOR_FILE
            if donor_path.is_file():
                donors = load_history_donors(donor_path)
        tokenizer.set_history_condition(condition, donors)
        return tokenizer

    def set_history_condition(
        self, condition: str, donors: dict[str, dict] | None = None,
    ) -> None:
        """Select the reference-history ablation variant.

        ``donors`` maps split -> row ID -> donor reference item IDs and is
        required before a ``shuffled_cue`` row can be encoded.
        """
        if condition not in HISTORY_CONDITIONS:
            raise ValueError(
                f"history_condition must be one of {HISTORY_CONDITIONS}, got {condition!r}")
        if condition != "shuffled_cue" and donors:
            raise ValueError("Only the shuffled_cue condition uses donor histories")
        self.history_condition = condition
        self.history_donors = (
            {
                str(split): {
                    str(row_id): [str(item_id) for item_id in items]
                    for row_id, items in rows.items()
                }
                for split, rows in donors.items()
            }
            if donors is not None else None
        )

    def _donor_references(self, split: str, row_id: str, count: int) -> list[str]:
        if self.history_donors is None:
            raise ValueError(
                "shuffled_cue needs donor histories; build them with "
                "scripts/prepare_wp_c_data.py --history-condition shuffled_cue")
        try:
            items = self.history_donors[str(split)][str(row_id)]
        except KeyError as exc:
            raise KeyError(f"No donor history for {split} row {row_id!r}") from exc
        if len(items) != count:
            raise ValueError(
                f"Donor history for {row_id!r} has {len(items)} items, expected {count}")
        return items

    def encode_references(
        self, reference_ids: list[str], *, split: str | None = None,
        row_id: str | None = None,
    ) -> list[int]:
        """Encode history items under the configured history condition."""
        semantic_end = 1 + RQ_N_CODEBOOKS + 1
        donors = None
        if self.history_condition == "shuffled_cue":
            donors = self._donor_references(split, row_id, len(reference_ids))
        tokens = []
        for position, item_id in enumerate(reference_ids):
            encoded = self.encode_item(item_id)
            if self.history_condition == "latent_only":
                encoded[semantic_end:] = [CUE_NULL_TOKEN] * self.active_cues
            elif self.history_condition == "cue_only":
                encoded[1:semantic_end] = [SEMANTIC_NULL_TOKEN] * (semantic_end - 1)
            elif self.history_condition == "shuffled_cue":
                encoded[semantic_end:] = [
                    TOKEN_LAYOUT.cue_token(cue) for cue in self.item2cues[donors[position]]]
            tokens.extend(encoded)
        return tokens

    @classmethod
    def from_files(
        cls,
        semantic_tokens_path: str | Path,
        item2cues_path: str | Path,
        cue_manifest_path: str | Path,
        catalog_items: list[CatalogItem],
        catalog_embeddings: np.ndarray,
        item_id_to_row: dict[str, int],
        codebook_weights_path: str | Path,
        active_cues: int = CUE_TOKENS,
    ) -> "GenPlaylistTokenizer":
        manifest = json.loads(Path(cue_manifest_path).read_text(encoding="utf-8"))
        if not manifest.get("wp_d_compatible", False):
            raise ValueError("Cue artifact is marked wp_d_compatible=false")
        if manifest.get("schema_version") != TOKEN_LAYOUT.schema_version:
            raise ValueError(
                f"Cue schema {manifest.get('schema_version')!r} does not match "
                f"{TOKEN_LAYOUT.schema_version!r}")
        if not 0 <= active_cues <= CUE_CANDIDATES_PER_ITEM:
            raise ValueError(
                f"active_cues must be in [0, {CUE_CANDIDATES_PER_ITEM}], "
                f"got {active_cues}")
        stored_cues = int(manifest.get(
            "stored_cues_per_item", manifest.get("cues_per_item", 0)))
        if stored_cues < active_cues:
            raise ValueError(
                f"Cue artifact stores {stored_cues} cues/item but WP-C needs the "
                f"first {active_cues}")
        semantic_tokens = json.loads(Path(semantic_tokens_path).read_text(encoding="utf-8"))
        item2cues = json.loads(Path(item2cues_path).read_text(encoding="utf-8"))
        bad_lengths = {
            str(item_id): len(cues)
            for item_id, cues in item2cues.items()
            if len(cues) != stored_cues
        }
        if bad_lengths:
            first = next(iter(bad_lengths.items()))
            raise ValueError(
                f"Cue artifact declares {stored_cues} stored cues/item but "
                f"{first[0]} has {first[1]}")
        weights = np.load(codebook_weights_path, allow_pickle=False)
        return cls(
            semantic_tokens, item2cues, catalog_items, catalog_embeddings,
            item_id_to_row, weights, active_cues=active_cues)

    def _validate_artifacts(self) -> None:
        validate_catalog_alignment(
            self.catalog_items, self.catalog_embeddings, self.item_id_to_row)
        expected_weights = (
            TOKEN_LAYOUT.rq_n_codebooks * TOKEN_LAYOUT.rq_codebook_size,
            CLHE_EMB_DIM,
        )
        if self.codebook_weights.shape != expected_weights:
            raise ValueError(
                f"RVQ codebook weights must be {expected_weights}, got "
                f"{self.codebook_weights.shape}")
        if not np.isfinite(self.codebook_weights).all():
            raise ValueError("RVQ codebook weights contain NaN or infinity")

        catalog_ids = set(self.item_id_to_row)
        for name, artifact in (
            ("semantic_tokens", self.semantic_tokens), ("item2cues", self.item2cues)):
            missing = sorted(catalog_ids - set(artifact))
            extra = sorted(set(artifact) - catalog_ids)
            if missing or extra:
                raise ValueError(
                    f"{name} ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
        for item_id in catalog_ids:
            self._validated_semantic_tokens(item_id)
            stored_cues = self.stored_item2cues[item_id]
            if len(stored_cues) < self.active_cues:
                raise ValueError(
                    f"Item {item_id} stores only {len(stored_cues)} cues; "
                    f"at least {self.active_cues} are required")
            cues = self.item2cues[item_id]
            if len(cues) != self.active_cues or any(
                cue < 0 or cue >= TOKEN_LAYOUT.cue_vocab_size for cue in cues):
                raise ValueError(f"Invalid cue IDs for item {item_id}: {cues}")

    def _validated_semantic_tokens(self, item_id: str) -> list[int]:
        tokens = self.semantic_tokens[item_id]
        if len(tokens) != RQ_N_CODEBOOKS + 1:
            raise ValueError(f"Item {item_id} needs 3 RVQ tokens + 1 conflict token")
        for level, token in enumerate(tokens[:RQ_N_CODEBOOKS]):
            lower = TOKEN_LAYOUT.rvq_token(level, 0)
            upper = TOKEN_LAYOUT.rvq_token(level, TOKEN_LAYOUT.rq_codebook_size - 1)
            if not lower <= token <= upper:
                raise ValueError(
                    f"Item {item_id} RVQ level {level} token {token} outside {lower}..{upper}")
        conflict = tokens[-1]
        lower = TOKEN_LAYOUT.conflict_token(0)
        upper = TOKEN_LAYOUT.conflict_token(TOKEN_LAYOUT.conflict_vocab_size - 1)
        if not lower <= conflict <= upper:
            raise ValueError(
                f"Item {item_id} conflict token {conflict} outside {lower}..{upper}")
        return tokens

    def encode_item(self, item_id: str) -> list[int]:
        item_id = str(item_id)
        if item_id not in self.semantic_tokens:
            raise KeyError(f"Unknown item ID: {item_id}")
        semantic = self._validated_semantic_tokens(item_id)
        cues = [TOKEN_LAYOUT.cue_token(cue) for cue in self.item2cues[item_id]]
        return [self.boi_token, *semantic, *cues]

    def encode_playlist(
        self, item_ids: list[str], context_items: int, *, split: str | None = None,
        row_id: str | None = None,
    ) -> TokenizedPlaylist:
        ids = [str(item_id) for item_id in item_ids]
        if len(ids) < 3:
            raise ValueError(
                "Playlist completion needs at least two references and one target")
        if not 2 <= context_items < len(ids):
            raise ValueError(
                "context_items must contain at least two references and leave targets")
        if not self.allow_repeated_items and len(set(ids)) != len(ids):
            raise ValueError("Sequence item IDs must be unique for this dataset")

        sequence = [self.bos_token]
        sequence.extend(self.encode_references(
            ids[:context_items], split=split, row_id=row_id))
        for item_id in ids[context_items:]:
            sequence.extend(self.encode_item(item_id))
        sequence.append(self.eos_token)
        target_mask = [False] * (1 + context_items * self.tokens_per_item)
        for _ in ids[context_items:]:
            target_mask.extend([False] + [True] * (self.tokens_per_item - 1))
        target_mask.append(False)

        input_ids = np.asarray(sequence, dtype=np.int64)
        special = np.isin(input_ids, [self.bos_token, self.boi_token, self.eos_token])
        attention_mask = ~special
        context_rows = [self.item_id_to_row[item_id] for item_id in ids[:context_items]]
        context_emb = self.catalog_embeddings[context_rows]
        mu_c = context_emb.mean(axis=0, dtype=np.float32)
        sigma_c2 = np.float32(np.mean(np.sum((context_emb - mu_c) ** 2, axis=1)))
        if self.history_condition == "cue_only":
            # These CLHE statistics are unused while CFG and structure
            # conditioning are off; zero them so the latent history cannot leak.
            context_emb = np.zeros_like(context_emb)
            mu_c = np.zeros_like(mu_c)
            sigma_c2 = np.float32(0.0)
        return TokenizedPlaylist(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=np.asarray(target_mask, dtype=bool),
            context_emb=context_emb,
            mu_c=mu_c,
            sigma_c2=sigma_c2,
        )

    def build_item_completion(
        self, context_tokens: list[int] | np.ndarray, *, num_items: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append ``num_items`` joint full-MASK item slots to references."""
        if num_items <= 0:
            raise ValueError(f"num_items must be positive, got {num_items}")
        values = np.asarray(context_tokens, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError(f"context_tokens must be 1-D, got {values.shape}")
        if len(values) < 2 or values[0] != self.bos_token or values[-1] != self.eos_token:
            raise ValueError("Context must be bounded by exactly one leading BOS and trailing EOS")
        if np.count_nonzero(values == self.eos_token) != 1:
            raise ValueError("Context must contain exactly one EOS")
        if np.any(values == self.mask_token_id):
            raise ValueError("Reference context must not contain MASK tokens")
        reference_width = len(values) - 2
        if reference_width % self.tokens_per_item != 0:
            raise ValueError("Reference context contains a partial item")
        if reference_width // self.tokens_per_item < 2:
            raise ValueError("Next-song completion requires at least two reference items")

        reference_items = reference_width // self.tokens_per_item
        if reference_items + num_items > self.max_items:
            raise ValueError(
                f"{reference_items} references + {num_items} targets exceed "
                f"max_items={self.max_items}")

        payload_width = self.tokens_per_item - 1
        target_tokens = []
        for _ in range(num_items):
            target_tokens.extend([
                self.boi_token,
                *([self.mask_token_id] * payload_width),
            ])
        completed = np.asarray(
            [*values[:-1], *target_tokens, self.eos_token], dtype=np.int64)
        completion_mask = np.zeros(len(completed), dtype=bool)
        first_target_boi = len(values) - 1
        for target_index in range(num_items):
            payload_start = (
                first_target_boi + target_index * self.tokens_per_item + 1)
            completion_mask[payload_start:payload_start + payload_width] = True
        return completed, completion_mask

    def build_next_item_completion(
        self, context_tokens: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible one-item completion used by the WP-D demo."""
        return self.build_item_completion(context_tokens, num_items=1)

    def make_type_mask(self, seq_len: int) -> np.ndarray:
        """Return ``[seq_len, runtime_vocab]`` legal-token positions.

        The final position is reserved for EOS; every complete item payload
        between BOS/EOS follows the configured item stride. MASK is never a
        legal clean prediction.
        """
        if seq_len < 2 or (seq_len - 2) % self.tokens_per_item != 0:
            raise ValueError(
                f"Sequence length must be 2 + n*{self.tokens_per_item}, got {seq_len}")
        legal = np.zeros((seq_len, TOKEN_LAYOUT.runtime_vocab_size), dtype=bool)
        legal[0, self.bos_token] = True
        legal[-1, self.eos_token] = True
        for position in range(1, seq_len - 1):
            offset = (position - 1) % self.tokens_per_item
            if offset == 0:
                legal[position, self.boi_token] = True
            elif 1 <= offset <= RQ_N_CODEBOOKS:
                level = offset - 1
                start = TOKEN_LAYOUT.rvq_token(level, 0)
                legal[position, start:start + TOKEN_LAYOUT.rq_codebook_size] = True
            elif offset == RQ_N_CODEBOOKS + 1:
                start = TOKEN_LAYOUT.conflict_token(0)
                legal[position, start:start + TOKEN_LAYOUT.conflict_vocab_size] = True
            else:
                start = TOKEN_LAYOUT.cue_token(0)
                legal[position, start:start + TOKEN_LAYOUT.cue_vocab_size] = True
        return legal

    def decode_item(
        self,
        tokens: list[int] | np.ndarray,
        *,
        mu_c: np.ndarray,
        sigma_c2: float,
        sample_idx: int = 0,
        context_prefix=None,
    ) -> GeneratedItem:
        values = [int(token) for token in tokens]
        if values and values[0] == self.boi_token:
            values = values[1:]
        if len(values) != self.tokens_per_item - 1:
            raise ValueError(
                f"Expected {self.tokens_per_item - 1} item payload tokens, got {len(values)}")
        semantic = values[:RQ_N_CODEBOOKS + 1]
        cue_tokens = values[RQ_N_CODEBOOKS + 1:]
        rvq_codes = tuple(
            semantic[level] - TOKEN_LAYOUT.rvq_token(level, 0)
            for level in range(RQ_N_CODEBOOKS)
        )
        conflict_code = semantic[-1] - TOKEN_LAYOUT.conflict_token(0)
        cue_ids = [token - TOKEN_LAYOUT.cue_token(0) for token in cue_tokens]
        # Reuse the same range checks used for catalog semantic tokens.
        synthetic_id = "<generated>"
        old = self.semantic_tokens.get(synthetic_id)
        self.semantic_tokens[synthetic_id] = semantic
        try:
            self._validated_semantic_tokens(synthetic_id)
        finally:
            if old is None:
                del self.semantic_tokens[synthetic_id]
            else:
                self.semantic_tokens[synthetic_id] = old
        if any(cue < 0 or cue >= TOKEN_LAYOUT.cue_vocab_size for cue in cue_ids):
            raise ValueError(f"Generated cue token outside cue range: {cue_tokens}")

        rows = [level * TOKEN_LAYOUT.rq_codebook_size + code
                for level, code in enumerate(rvq_codes)]
        z_hat = self.codebook_weights[rows].sum(axis=0).astype(np.float32)
        generated = GeneratedItem(
            rvq_codes=rvq_codes,
            conflict_code=conflict_code,
            z_hat_emb=z_hat,
            mu_c_emb=np.asarray(mu_c, dtype=np.float32),
            sigma_c2=float(sigma_c2),
            cue_ids=cue_ids,
            sample_idx=sample_idx,
            context_prefix=context_prefix,
        )
        return generated.validate()

    def _token_to_feature(self, semantic_tokens) -> np.ndarray:
        """Compatibility decoder for evaluator paths (3 RVQ + conflict)."""
        values = [int(token) for token in semantic_tokens]
        if len(values) != RQ_N_CODEBOOKS + 1:
            raise ValueError(f"Expected four semantic tokens, got {len(values)}")
        rvq_codes = [
            values[level] - TOKEN_LAYOUT.rvq_token(level, 0)
            for level in range(RQ_N_CODEBOOKS)
        ]
        for level, code in enumerate(rvq_codes):
            TOKEN_LAYOUT.rvq_token(level, code)
        TOKEN_LAYOUT.conflict_token(values[-1] - TOKEN_LAYOUT.conflict_token(0))
        rows = [level * TOKEN_LAYOUT.rq_codebook_size + code
                for level, code in enumerate(rvq_codes)]
        return self.codebook_weights[rows].sum(axis=0).astype(np.float32)

    @property
    def n_digit(self) -> int:
        return RQ_N_CODEBOOKS

    @property
    def padding_token(self) -> int:
        return self.bos_token

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.max_items * self.tokens_per_item + 1

    def tokenize(self, datasets: dict) -> dict:
        """Tokenize fixed joint 15-reference/5-target train and test rows."""
        tokenized = {}
        for split, source in datasets.items():
            protocol = FROZEN_NEXT_SONG_PROTOCOL.validate_config(self.config)
            usable = source.filter(
                lambda row: len(row["item_seq"]) == protocol.train_total_items)

            def encode_row(row):
                item_ids = [str(item_id) for item_id in row["item_seq"]]
                row_id = str(row["bundle"])
                if split == "test":
                    reference_count = protocol.eval_reference_items
                    target_count = protocol.eval_target_items
                    expected_count = protocol.eval_total_items
                    if len(item_ids) != expected_count:
                        raise ValueError(
                            f"Test rows must contain exactly {expected_count} items for "
                            f"{reference_count}->{target_count} evaluation, got {len(item_ids)}")
                    reference_ids = item_ids[:reference_count]
                    target_ids = item_ids[reference_count:]
                    encoded = self.encode_playlist(
                        [*reference_ids, *target_ids], context_items=reference_count,
                        split=split, row_id=row_id)
                else:
                    reference_count = protocol.train_reference_items
                    target_count = protocol.train_target_items
                    if len(item_ids) != protocol.train_total_items:
                        raise ValueError(
                            f"Train rows must contain exactly {protocol.train_total_items} "
                            f"items for {reference_count}->{target_count}, got {len(item_ids)}")
                    reference_ids = item_ids[:reference_count]
                    target_ids = item_ids[reference_count:]
                    encoded = self.encode_playlist(
                        item_ids, context_items=reference_count,
                        split=split, row_id=row_id)
                result = {
                    "input_ids": encoded.input_ids.tolist(),
                    "sequence_mask": [True] * len(encoded.input_ids),
                    "attention_mask": encoded.attention_mask.tolist(),
                    "target_mask": encoded.target_mask.tolist(),
                    # Mean context is the portable CFG path; mu_c is also fed
                    # independently into AdaLN structure conditioning.
                    "context_emb": encoded.mu_c.tolist(),
                    "mu_c": encoded.mu_c.tolist(),
                    "sigma_c2": float(encoded.sigma_c2),
                }
                if split == "test":
                    context_tokens = [
                        self.bos_token,
                        *self.encode_references(reference_ids, split=split, row_id=row_id),
                        self.eos_token,
                    ]
                    result["input_ids"] = context_tokens
                    result["sequence_mask"] = [True] * len(context_tokens)
                    result["attention_mask"] = [
                        token not in (self.bos_token, self.boi_token, self.eos_token)
                        for token in context_tokens]
                    result["target_mask"] = [False] * len(context_tokens)
                    result["labels"] = [
                        self._validated_semantic_tokens(item_id)
                        for item_id in target_ids]
                return result

            tokenized[split] = usable.map(
                encode_row,
                remove_columns=usable.column_names,
                desc=f"Tokenizing {split} set (GenPlaylist v1)",
            )
            tokenized[split].set_format(type="torch")
        return tokenized

    def collate_batch(self, examples: list[dict]):
        """Pad variable playlist lengths while keeping all padding/context fixed."""
        import torch

        if not examples:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(example["input_ids"]) for example in examples)
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, max_length), self.padding_token, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        target_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        sequence_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        for row, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[row, :length] = torch.as_tensor(example["input_ids"], dtype=torch.long)
            sequence_mask[row, :length] = True
            attention_mask[row, :length] = torch.as_tensor(
                example["attention_mask"], dtype=torch.bool)
            target_mask[row, :length] = torch.as_tensor(
                example["target_mask"], dtype=torch.bool)
        batch = {
            "input_ids": input_ids,
            "sequence_mask": sequence_mask,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "context_emb": torch.stack([
                torch.as_tensor(example["context_emb"], dtype=torch.float32)
                for example in examples]),
            "mu_c": torch.stack([
                torch.as_tensor(example["mu_c"], dtype=torch.float32)
                for example in examples]),
            "sigma_c2": torch.as_tensor(
                [example["sigma_c2"] for example in examples], dtype=torch.float32),
        }
        if "labels" in examples[0]:
            batch["labels"] = torch.stack([
                torch.as_tensor(example["labels"], dtype=torch.long)
                for example in examples])
        return batch
