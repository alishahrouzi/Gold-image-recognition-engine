"""Duplicate detection for dataset QA (Sprint S1.3).

This module is detection and reporting only. It never deletes, moves,
renames, or rewrites images, and it never modifies a manifest.

Classification
--------------
exact
    Identical file bytes (cryptographic content hash).
perceptual
    Identical perceptual hash (Hamming distance 0). Visually identical or
    effectively identical after encoding/resize differences. Not reported
    when the group is already explained by a single exact hash.
near
    Hamming distance in ``1 .. threshold`` on the 64-bit perceptual hash.
    These are candidate near-duplicates, not confirmed true duplicates.
cross_dataset
    A relationship whose files come from two dataset roots.

Near-duplicate search is candidate-based (multi-index / pigeonhole on hash
bands). It does not fall back to a silent full pairwise skip when N is large.
If a configured limit stops the search, the report records completed vs
skipped comparisons and must not be read as "no near duplicates".
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .constants import ALLOWED_SPLITS
from .loaders.image_loader import to_rgb_image

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

SCHEMA_VERSION = "1.0"

DEFAULT_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".webp",
)
DEFAULT_EXACT_HASH = "sha256"
DEFAULT_PERCEPTUAL_HASH = "phash"
# Maximum Hamming distance (64-bit pHash) counted as a near-duplicate.
# Distance 0 is perceptual identity, not a near-duplicate.
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 5
PHASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4
PHASH_BITS = PHASH_SIZE * PHASH_SIZE
HASH_BLOCK_SIZE = 65536
COMPARISON_MODES = frozenset({"intra", "cross", "both"})

DATASET_A = "dataset_a"
DATASET_B = "dataset_b"


@dataclass(frozen=True)
class DuplicateDetectionConfig:
    """Inputs that control duplicate detection.

    threshold
        Inclusive maximum Hamming distance for near-duplicates on a 64-bit
        perceptual hash. Lower is stricter. Distance 0 is classified as
        perceptual, not near.
    max_candidates
        If set, skip near-duplicate comparisons for an image whose candidate
        set is larger than this value. None / 0 means no candidate-set cap.
    max_pairs
        If set, stop after this many Hamming comparisons. None / 0 means no
        pairwise cap. Skipped comparisons are recorded in the report.
    """

    image_extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    exact_hash: str = DEFAULT_EXACT_HASH
    perceptual_hash: str = DEFAULT_PERCEPTUAL_HASH
    threshold: int = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    max_candidates: Optional[int] = None
    max_pairs: Optional[int] = None
    comparison_mode: str = "both"

    def normalized(self) -> "DuplicateDetectionConfig":
        extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in self.image_extensions
        )
        if not extensions:
            raise ValueError("image_extensions must not be empty.")
        exact_hash = self.exact_hash.lower().strip()
        if exact_hash not in {"sha256", "sha1", "md5"}:
            raise ValueError(
                f"Unsupported exact hash {self.exact_hash!r}. "
                "Expected sha256, sha1, or md5."
            )
        perceptual_hash = self.perceptual_hash.lower().strip()
        if perceptual_hash not in {"phash", "dhash", "ahash"}:
            raise ValueError(
                f"Unsupported perceptual hash {self.perceptual_hash!r}. "
                "Expected phash, dhash, or ahash."
            )
        if self.threshold < 0 or self.threshold > PHASH_BITS:
            raise ValueError(
                f"threshold must be between 0 and {PHASH_BITS} inclusive."
            )
        mode = self.comparison_mode.lower().strip()
        if mode not in COMPARISON_MODES:
            raise ValueError(
                f"comparison_mode must be one of {sorted(COMPARISON_MODES)}."
            )
        max_candidates = _unlimited_to_none(self.max_candidates)
        max_pairs = _unlimited_to_none(self.max_pairs)
        if max_candidates is not None and max_candidates < 0:
            raise ValueError("max_candidates must be >= 0 (0 means unlimited).")
        if max_pairs is not None and max_pairs < 0:
            raise ValueError("max_pairs must be >= 0 (0 means unlimited).")
        return DuplicateDetectionConfig(
            image_extensions=extensions,
            exact_hash=exact_hash,
            perceptual_hash=perceptual_hash,
            threshold=self.threshold,
            max_candidates=max_candidates,
            max_pairs=max_pairs,
            comparison_mode=mode,
        )


@dataclass
class ImageFingerprint:
    """Read-only fingerprint of one image file."""

    dataset: str
    relative_path: str
    image_id: Optional[str] = None
    category: Optional[str] = None
    split: Optional[str] = None
    group_id: Optional[str] = None
    exact_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    exact_hash_error: Optional[str] = None
    perceptual_hash_error: Optional[str] = None

    def to_dict(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image_id": self.image_id,
            "relative_path": self.relative_path,
            "dataset": self.dataset,
            "category": self.category,
            "split": self.split,
            "group_id": self.group_id,
            "exact_hash": self.exact_hash,
            "perceptual_hash": self.perceptual_hash,
        }
        if extra:
            payload.update(extra)
        return payload


def detect_duplicates(
    dataset_a: PathLike,
    dataset_b: Optional[PathLike] = None,
    *,
    config: Optional[DuplicateDetectionConfig] = None,
    manifest_a: Optional[PathLike] = None,
    manifest_b: Optional[PathLike] = None,
    dataset_a_name: Optional[str] = None,
    dataset_b_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect exact, perceptual, near, and optional cross-dataset duplicates.

    Dataset directories and manifests are read, never modified.
    """
    cfg = (config or DuplicateDetectionConfig()).normalized()
    root_a = _require_dataset_root(dataset_a, DATASET_A)
    root_b = _require_dataset_root(dataset_b, DATASET_B) if dataset_b is not None else None
    if cfg.comparison_mode == "cross" and root_b is None:
        raise ValueError("comparison_mode='cross' requires dataset_b.")

    name_a = dataset_a_name or root_a.name
    name_b = dataset_b_name or (root_b.name if root_b is not None else None)

    records_a = _index_dataset(
        dataset_key=DATASET_A,
        root=root_a,
        config=cfg,
        manifest_path=manifest_a,
    )
    records_b: List[ImageFingerprint] = []
    if root_b is not None:
        records_b = _index_dataset(
            dataset_key=DATASET_B,
            root=root_b,
            config=cfg,
            manifest_path=manifest_b,
        )

    include_intra = cfg.comparison_mode in {"intra", "both"}
    include_cross = cfg.comparison_mode in {"cross", "both"} and root_b is not None

    exact_groups: List[Dict[str, Any]] = []
    perceptual_groups: List[Dict[str, Any]] = []
    near_pairs: List[Dict[str, Any]] = []
    if include_intra:
        exact_groups.extend(_exact_groups(records_a, dataset_scope=DATASET_A))
        if records_b:
            exact_groups.extend(_exact_groups(records_b, dataset_scope=DATASET_B))
        perceptual_groups.extend(_perceptual_groups(records_a, dataset_scope=DATASET_A))
        if records_b:
            perceptual_groups.extend(_perceptual_groups(records_b, dataset_scope=DATASET_B))

    search_stats = {
        "status": "complete",
        "comparisons_completed": 0,
        "comparisons_skipped": 0,
        "skip_reasons": [],
    }
    limitations: List[Dict[str, Any]] = []

    if include_intra:
        intra_records = list(records_a) + list(records_b)
        intra_pairs, intra_stats, intra_limits = _near_duplicate_pairs(
            intra_records,
            threshold=cfg.threshold,
            max_candidates=cfg.max_candidates,
            max_pairs=cfg.max_pairs,
            require_cross_dataset=False,
        )
        near_pairs.extend(intra_pairs)
        _merge_search_stats(search_stats, intra_stats)
        limitations.extend(intra_limits)

    cross_matches: List[Dict[str, Any]] = []
    if include_cross and root_b is not None:
        combined = records_a + records_b
        cross_matches.extend(_cross_exact_groups(combined, name_a, name_b))
        cross_matches.extend(_cross_perceptual_groups(combined, name_a, name_b))
        cross_near, cross_stats, cross_limits = _near_duplicate_pairs(
            combined,
            threshold=cfg.threshold,
            max_candidates=cfg.max_candidates,
            max_pairs=cfg.max_pairs,
            require_cross_dataset=True,
        )
        for pair in cross_near:
            cross_matches.append(
                {
                    "duplicate_type": "cross_dataset",
                    "match_kind": "near",
                    "dataset_a": name_a,
                    "dataset_b": name_b,
                    "hamming_distance": pair["hamming_distance"],
                    "matches": [pair["file_a"], pair["file_b"]],
                }
            )
        _merge_search_stats(search_stats, cross_stats)
        limitations.extend(cross_limits)

    if search_stats["comparisons_skipped"] and search_stats["status"] != "partial":
        search_stats["status"] = "partial"

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    datasets_block: Dict[str, Any] = {
        DATASET_A: _dataset_block(name_a, records_a),
        DATASET_B: (
            _dataset_block(name_b or DATASET_B, records_b)
            if root_b is not None
            else None
        ),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "configuration": {
            "image_extensions": list(cfg.image_extensions),
            "exact_hash": cfg.exact_hash,
            "perceptual_hash": cfg.perceptual_hash,
            "threshold": cfg.threshold,
            "threshold_meaning": (
                f"Inclusive maximum Hamming distance on a {PHASH_BITS}-bit "
                f"{cfg.perceptual_hash} for near-duplicates. Distance 0 is "
                "perceptual identity and is reported separately."
            ),
            "max_candidates": cfg.max_candidates,
            "max_pairs": cfg.max_pairs,
            "comparison_mode": cfg.comparison_mode,
            "manifest_a": Path(manifest_a).name if manifest_a else None,
            "manifest_b": Path(manifest_b).name if manifest_b else None,
        },
        "datasets": datasets_block,
        "summary": {
            "total_images": len(records_a) + len(records_b),
            "dataset_a_images": len(records_a),
            "dataset_b_images": len(records_b) if root_b is not None else 0,
            "exact_duplicate_groups": len(exact_groups),
            "perceptual_duplicate_groups": len(perceptual_groups),
            "near_duplicate_pairs": len(near_pairs),
            "cross_dataset_matches": len(cross_matches),
            "unreadable_images": sum(
                1
                for item in records_a + records_b
                if item.perceptual_hash is None and item.perceptual_hash_error
            ),
            "near_duplicate_search": dict(search_stats),
        },
        "exact_duplicates": exact_groups,
        "perceptual_duplicates": perceptual_groups,
        "near_duplicates": near_pairs,
        "cross_dataset_duplicates": cross_matches,
        "limitations": limitations,
    }
    return report


def write_duplicate_report(report: Mapping[str, Any], output_path: PathLike) -> Path:
    """Write the JSON report. Does not modify images or manifests."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote duplicate report: %s", path)
    return path


def file_content_hash(path: Path, algorithm: str = DEFAULT_EXACT_HASH) -> str:
    """Cryptographic hash of file bytes. Does not interpret image pixels."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash_hex(
    path: Path,
    algorithm: str = DEFAULT_PERCEPTUAL_HASH,
) -> str:
    """Return the hex perceptual hash for a readable image file."""
    with Image.open(path) as image:
        image.load()
        rgb = to_rgb_image(image)
        value = _perceptual_hash_int(rgb, algorithm)
        if rgb is not image:
            rgb.close()
    return f"{value:016x}"


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _unlimited_to_none(value: Optional[int]) -> Optional[int]:
    if value is None or value == 0:
        return None
    return value


def _require_dataset_root(path: PathLike, label: str) -> Path:
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"{label} is not an existing directory: {root}")
    return root


def _index_dataset(
    *,
    dataset_key: str,
    root: Path,
    config: DuplicateDetectionConfig,
    manifest_path: Optional[PathLike],
) -> List[ImageFingerprint]:
    metadata = _load_manifest_index(manifest_path) if manifest_path else {}
    files = _find_images(root, config.image_extensions)
    records: List[ImageFingerprint] = []
    total = len(files)
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        meta = metadata.get(relative, {})
        split, category = _infer_split_category(relative)
        record = ImageFingerprint(
            dataset=dataset_key,
            relative_path=relative,
            image_id=meta.get("image_id") or None,
            category=meta.get("category") or category,
            split=meta.get("split") or split,
            group_id=meta.get("group_id") or None,
        )
        try:
            record.exact_hash = file_content_hash(path, config.exact_hash)
        except OSError as exc:
            record.exact_hash_error = str(exc)
        try:
            record.perceptual_hash = perceptual_hash_hex(path, config.perceptual_hash)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            record.perceptual_hash_error = str(exc).strip() or exc.__class__.__name__
        records.append(record)
        if index % 500 == 0 or index == total:
            logger.info(
                "Duplicate detection hashing %s: %s / %s",
                dataset_key,
                index,
                total,
            )
    records.sort(key=lambda item: item.relative_path)
    return records


def _find_images(root: Path, extensions: Sequence[str]) -> List[Path]:
    allowed = {ext.lower() for ext in extensions}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    files.sort()
    return files


def _load_manifest_index(manifest_path: PathLike) -> Dict[str, Dict[str, str]]:
    """Read-only lookup from relative image_path to identifiers."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")
    index: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = (row.get("image_path") or "").replace("\\", "/")
            if not image_path:
                continue
            index[image_path] = {
                "image_id": (row.get("image_id") or "").strip(),
                "group_id": (row.get("group_id") or "").strip(),
                "category": (row.get("category") or "").strip(),
                "split": (row.get("split") or "").strip(),
            }
    return index


def _infer_split_category(relative_path: str) -> Tuple[Optional[str], Optional[str]]:
    parts = Path(relative_path).parts
    if not parts:
        return None, None
    split = parts[0] if parts[0] in ALLOWED_SPLITS else None
    if split and len(parts) >= 3:
        return split, parts[1]
    if split and len(parts) == 2:
        return split, None
    if len(parts) >= 2:
        return None, parts[0]
    return None, None


def _perceptual_hash_int(image: Image.Image, algorithm: str) -> int:
    gray = image.convert("L")
    if algorithm == "phash":
        return _phash(gray)
    if algorithm == "dhash":
        return _dhash(gray)
    if algorithm == "ahash":
        return _ahash(gray)
    raise ValueError(f"Unsupported perceptual hash {algorithm!r}.")


def _phash(gray: Image.Image) -> int:
    """DCT perceptual hash compatible with the Sprint 0 imagehash.phash idea.

    Resize to 32x32, 2D type-II DCT, take the 8x8 low-frequency block, and
    threshold against the median. The result is 64 bits.
    """
    size = PHASH_SIZE * PHASH_HIGHFREQ_FACTOR
    pixels = np.asarray(
        gray.resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    dct_full = _dct2_type2(pixels)
    low = dct_full[:PHASH_SIZE, :PHASH_SIZE]
    median = float(np.median(low))
    bits = low > median
    return _bits_to_int(bits)


def _dhash(gray: Image.Image) -> int:
    pixels = np.asarray(
        gray.resize((PHASH_SIZE + 1, PHASH_SIZE), Image.Resampling.LANCZOS),
        dtype=np.int16,
    )
    diff = pixels[:, 1:] > pixels[:, :-1]
    return _bits_to_int(diff)


def _ahash(gray: Image.Image) -> int:
    pixels = np.asarray(
        gray.resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    return _bits_to_int(pixels > float(pixels.mean()))


def _dct2_type2(pixels: np.ndarray) -> np.ndarray:
    """Unnormalized separable type-II DCT (scipy.fftpack.dct default)."""
    return _dct1d_type2(_dct1d_type2(pixels, axis=0), axis=1)


def _dct1d_type2(values: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    length = moved.shape[-1]
    n_idx = np.arange(length)
    k_idx = np.arange(length)
    weights = 2.0 * np.cos(
        np.pi * np.outer(k_idx, 2 * n_idx + 1) / (2.0 * length)
    )
    transformed = moved @ weights.T
    return np.moveaxis(transformed, -1, axis)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=bool).flatten():
        value = (value << 1) | int(bit)
    return value


def _exact_groups(
    records: Sequence[ImageFingerprint],
    *,
    dataset_scope: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[ImageFingerprint]] = defaultdict(list)
    for record in records:
        if record.exact_hash:
            grouped[record.exact_hash].append(record)
    groups = []
    for digest, files in grouped.items():
        if len(files) < 2:
            continue
        groups.append(
            {
                "duplicate_type": "exact",
                "dataset": dataset_scope,
                "hash": digest,
                "files": [item.to_dict() for item in _sorted_files(files)],
            }
        )
    groups.sort(key=lambda item: item["hash"])
    return groups


def _perceptual_groups(
    records: Sequence[ImageFingerprint],
    *,
    dataset_scope: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[ImageFingerprint]] = defaultdict(list)
    for record in records:
        if record.perceptual_hash:
            grouped[record.perceptual_hash].append(record)
    groups = []
    for digest, files in grouped.items():
        if len(files) < 2:
            continue
        unique_exact = {item.exact_hash for item in files if item.exact_hash}
        if len(unique_exact) <= 1:
            continue
        groups.append(
            {
                "duplicate_type": "perceptual",
                "dataset": dataset_scope,
                "hash": digest,
                "hamming_distance": 0,
                "files": [item.to_dict() for item in _sorted_files(files)],
            }
        )
    groups.sort(key=lambda item: item["hash"])
    return groups


def _cross_exact_groups(
    records: Sequence[ImageFingerprint],
    name_a: str,
    name_b: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[ImageFingerprint]] = defaultdict(list)
    for record in records:
        if record.exact_hash:
            grouped[record.exact_hash].append(record)
    matches = []
    for digest, files in grouped.items():
        datasets = {item.dataset for item in files}
        if DATASET_A not in datasets or DATASET_B not in datasets:
            continue
        matches.append(
            {
                "duplicate_type": "cross_dataset",
                "match_kind": "exact",
                "dataset_a": name_a,
                "dataset_b": name_b,
                "hash": digest,
                "matches": [item.to_dict() for item in _sorted_files(files)],
            }
        )
    matches.sort(key=lambda item: item["hash"])
    return matches


def _cross_perceptual_groups(
    records: Sequence[ImageFingerprint],
    name_a: str,
    name_b: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[ImageFingerprint]] = defaultdict(list)
    for record in records:
        if record.perceptual_hash:
            grouped[record.perceptual_hash].append(record)
    matches = []
    for digest, files in grouped.items():
        datasets = {item.dataset for item in files}
        if DATASET_A not in datasets or DATASET_B not in datasets:
            continue
        unique_exact = {item.exact_hash for item in files if item.exact_hash}
        if len(unique_exact) <= 1:
            continue
        matches.append(
            {
                "duplicate_type": "cross_dataset",
                "match_kind": "perceptual",
                "dataset_a": name_a,
                "dataset_b": name_b,
                "hash": digest,
                "hamming_distance": 0,
                "matches": [item.to_dict() for item in _sorted_files(files)],
            }
        )
    matches.sort(key=lambda item: item["hash"])
    return matches


def _near_duplicate_pairs(
    records: Sequence[ImageFingerprint],
    *,
    threshold: int,
    max_candidates: Optional[int],
    max_pairs: Optional[int],
    require_cross_dataset: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    usable = [item for item in records if item.perceptual_hash]
    stats = {
        "status": "complete",
        "comparisons_completed": 0,
        "comparisons_skipped": 0,
        "skip_reasons": [],
    }
    limitations: List[Dict[str, Any]] = []
    if threshold < 1 or len(usable) < 2:
        return [], stats, limitations

    indexed = list(enumerate(usable))
    hash_ints = [int(item.perceptual_hash, 16) for _, item in indexed]
    # threshold == 64 means every pair is in range; band indexing cannot
    # represent that with 64 bits, so every later index is a candidate.
    full_candidate_scan = threshold >= PHASH_BITS
    layout = _band_layout(PHASH_BITS, threshold)
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    if not full_candidate_scan:
        for idx, hash_int in enumerate(hash_ints):
            for band_id, (start, width) in enumerate(layout):
                buckets[(band_id, _band_value(hash_int, start, width))].append(idx)

    pairs: List[Dict[str, Any]] = []
    remaining_after_stop: List[int] = []
    stopped = False
    skip_reason: Optional[str] = None

    for idx, record in indexed:
        if full_candidate_scan:
            candidates = set(range(idx + 1, len(usable)))
        else:
            candidates = set()
            hash_int = hash_ints[idx]
            for band_id, (start, width) in enumerate(layout):
                candidates.update(
                    buckets[(band_id, _band_value(hash_int, start, width))]
                )
            candidates = {other for other in candidates if other > idx}
        if require_cross_dataset:
            candidates = {
                other
                for other in candidates
                if usable[other].dataset != record.dataset
            }
        else:
            candidates = {
                other
                for other in candidates
                if usable[other].dataset == record.dataset
            }
        candidate_list = sorted(candidates)
        if stopped:
            remaining_after_stop.append(len(candidate_list))
            continue
        if max_candidates is not None and len(candidate_list) > max_candidates:
            stats["comparisons_skipped"] += len(candidate_list)
            stats["status"] = "partial"
            skip_reason = "max_candidates"
            if "max_candidates" not in stats["skip_reasons"]:
                stats["skip_reasons"].append("max_candidates")
            continue
        for position, other in enumerate(candidate_list):
            if max_pairs is not None and stats["comparisons_completed"] >= max_pairs:
                leftover = len(candidate_list) - position
                stats["comparisons_skipped"] += leftover
                stats["status"] = "partial"
                skip_reason = "max_pairs"
                if "max_pairs" not in stats["skip_reasons"]:
                    stats["skip_reasons"].append("max_pairs")
                stopped = True
                remaining_after_stop.append(0)
                break
            stats["comparisons_completed"] += 1
            distance = (hash_ints[idx] ^ hash_ints[other]).bit_count()
            if distance < 1 or distance > threshold:
                continue
            left = record
            right = usable[other]
            if left.exact_hash and left.exact_hash == right.exact_hash:
                continue
            if left.perceptual_hash == right.perceptual_hash:
                continue
            ordered = _sorted_files((left, right))
            pairs.append(
                {
                    "duplicate_type": "near",
                    "hamming_distance": distance,
                    "file_a": ordered[0].to_dict(),
                    "file_b": ordered[1].to_dict(),
                }
            )

    if remaining_after_stop:
        extra = sum(remaining_after_stop)
        stats["comparisons_skipped"] += extra
        stats["status"] = "partial"
        if skip_reason and skip_reason not in stats["skip_reasons"]:
            stats["skip_reasons"].append(skip_reason)

    if stats["status"] == "partial":
        reason = " and ".join(stats["skip_reasons"]) or "configured_limit"
        limitations.append(
            {
                "code": "near_duplicate_search_incomplete",
                "scope": "cross_dataset" if require_cross_dataset else "intra_dataset",
                "reason": reason,
                "comparisons_completed": stats["comparisons_completed"],
                "comparisons_skipped": stats["comparisons_skipped"],
                "message": (
                    "Near-duplicate search did not compare every candidate pair. "
                    "Do not interpret an empty or short near-duplicate list as "
                    "proof that no near duplicates exist."
                ),
            }
        )

    pairs.sort(
        key=lambda item: (
            item["hamming_distance"],
            item["file_a"]["relative_path"],
            item["file_b"]["relative_path"],
            item["file_a"]["dataset"],
        )
    )
    return pairs, stats, limitations


def _band_layout(hash_bits: int, threshold: int) -> List[Tuple[int, int]]:
    """Split the hash into threshold+1 bands (pigeonhole candidate index)."""
    n_bands = min(threshold + 1, hash_bits)
    base, remainder = divmod(hash_bits, n_bands)
    layout: List[Tuple[int, int]] = []
    start = 0
    for band in range(n_bands):
        width = base + (1 if band < remainder else 0)
        if width <= 0:
            continue
        layout.append((start, width))
        start += width
    return layout


def _band_value(hash_int: int, start: int, width: int, hash_bits: int = PHASH_BITS) -> int:
    shift = hash_bits - (start + width)
    mask = (1 << width) - 1
    return (hash_int >> shift) & mask


def _sorted_files(files: Iterable[ImageFingerprint]) -> List[ImageFingerprint]:
    return sorted(files, key=lambda item: (item.dataset, item.relative_path))


def _dataset_block(name: str, records: Sequence[ImageFingerprint]) -> Dict[str, Any]:
    return {
        "name": name,
        "label": name,
        "total_images": len(records),
        "hashed_images": sum(1 for item in records if item.exact_hash),
        "perceptual_hashed_images": sum(
            1 for item in records if item.perceptual_hash
        ),
        "unreadable_images": sum(
            1 for item in records if item.perceptual_hash_error
        ),
    }


def _merge_search_stats(target: Dict[str, Any], extra: Mapping[str, Any]) -> None:
    target["comparisons_completed"] += extra.get("comparisons_completed", 0)
    target["comparisons_skipped"] += extra.get("comparisons_skipped", 0)
    if extra.get("status") == "partial":
        target["status"] = "partial"
    for reason in extra.get("skip_reasons", []):
        if reason not in target["skip_reasons"]:
            target["skip_reasons"].append(reason)
