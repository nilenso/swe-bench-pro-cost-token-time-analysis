"""
Layer 3: File collection, parallel processing, and disk caching.

This is the "run the pipeline" layer. It finds .traj files, classifies them
in parallel, caches results, and returns structured data keyed by model.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .models import MODELS

# Cache lives at project-root/.cache/analysis/
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "analysis"


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_files(
    data_root: Path,
    models: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Find all .traj files under data_root, optionally filtered by model.

    Expected layout: data_root/<model>/traj/<instance_dir>/<file>.traj

    Returns list of (model_key, path) tuples.
    """
    target_models = models if models is not None else list(MODELS.keys())
    out: list[tuple[str, Path]] = []
    for model in target_models:
        base = data_root / model / "traj"
        if not base.exists():
            continue
        for p in sorted(base.glob("*/*.traj")):
            out.append((model, p))
    return out


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(path_str: str) -> str:
    """Deterministic cache key from file path + size + mtime."""
    st = os.stat(path_str)
    raw = f"{path_str}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(path_str: str) -> dict | None:
    """Read cached result if it exists and is valid."""
    key = _cache_key(path_str)
    cache_path = _CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_bytes())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(path_str: str, data: dict) -> None:
    """Write result to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(path_str)
        cache_path = _CACHE_DIR / f"{key}.json"
        cache_path.write_text(json.dumps(data, separators=(",", ":")))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Per-file worker (top-level function for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(args: tuple[str, str]) -> dict | None:
    """Process a single .traj file. Returns dict or None for empty files.

    This is a top-level function (not a method or closure) so it can be
    pickled by multiprocessing.
    """
    model, path_str = args

    # Check cache first
    cached = _read_cache(path_str)
    if cached is not None:
        cached["model"] = model
        return cached

    # Import here to avoid circular imports at module level in workers
    from .classify import classify_file, FileResult  # noqa: F811

    result = classify_file(model, path_str)
    if result is None:
        return None

    # Serialize to dict for caching. We store everything except the model
    # key (which is added back when reading from cache).
    data = asdict(result)
    cache_data = {k: v for k, v in data.items() if k != "model"}
    _write_cache(path_str, cache_data)

    data["model"] = model
    return data


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _dict_to_file_result(d: dict):
    """Reconstruct a FileResult from a dict (cached or freshly computed)."""
    from .classify import FileResult
    # Filter to only fields the dataclass accepts
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(FileResult)}
    filtered = {k: v for k, v in d.items() if k in field_names}
    return FileResult(**filtered)


def process_all(
    data_root: Path,
    models: list[str] | None = None,
    max_workers: int | None = None,
) -> dict[str, list]:
    """Classify all .traj files in parallel. Returns {model: [FileResult, ...]}.

    Results are keyed by whatever models are found in the data, not by a
    hardcoded list. If `models` is specified, only those models are processed.
    """
    files = collect_files(data_root, models)
    if not files:
        return {}

    tasks = [(model, str(path)) for model, path in files]
    workers = max_workers or min(8, os.cpu_count() or 1)

    results: dict[str, list] = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for data in ex.map(_process_one, tasks, chunksize=32):
            if data is None:
                continue
            model = data["model"]
            fr = _dict_to_file_result(data)
            results.setdefault(model, []).append(fr)

    return results
