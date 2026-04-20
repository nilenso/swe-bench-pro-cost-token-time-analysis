"""
Parallel Pi transcript processing + disk cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "analysis_pi"
_CACHE_VERSION = "v6"


def collect_files(data_root: Path) -> list[Path]:
    if data_root.is_file() and data_root.suffix == ".jsonl":
        return [data_root]
    return sorted(p for p in data_root.glob("*.jsonl") if p.is_file())


def _cache_key(path_str: str) -> str:
    st = os.stat(path_str)
    raw = f"{_CACHE_VERSION}:{path_str}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(path_str: str) -> dict | None:
    key = _cache_key(path_str)
    cache_path = _CACHE_DIR / f"{key}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path_str: str, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(path_str)
        (_CACHE_DIR / f"{key}.json").write_text(json.dumps(data, separators=(",", ":")))
    except OSError:
        pass


def _process_one(path_str: str) -> dict | None:
    cached = _read_cache(path_str)
    if cached is not None:
        return cached

    from .classify import classify_file

    result = classify_file(path_str)
    if result is None:
        return None
    data = asdict(result)
    _write_cache(path_str, data)
    return data


def _dict_to_file_result(d: dict):
    import dataclasses

    from .classify import FileResult

    field_names = {f.name for f in dataclasses.fields(FileResult)}
    filtered = {k: v for k, v in d.items() if k in field_names}
    return FileResult(**filtered)


def process_all(
    data_root: Path,
    models: list[str] | None = None,
    max_workers: int | None = None,
) -> dict[str, list]:
    files = collect_files(data_root)
    if not files:
        return {}

    tasks = [str(p) for p in files]
    workers = max_workers or min(8, os.cpu_count() or 1)
    results: dict[str, list] = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for data in ex.map(_process_one, tasks, chunksize=32):
            if data is None:
                continue
            model = data.get("model", "unknown")
            if models is not None and model not in models:
                continue
            fr = _dict_to_file_result(data)
            results.setdefault(model, []).append(fr)

    return results
