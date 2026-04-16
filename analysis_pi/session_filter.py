"""Helpers for configurable Pi session filtering."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .models import normalize_model_name


DEFAULT_EXACT_MODELS = [
    "gpt-5.4",
    "claude-opus-4-5",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "claude-opus-4-6",
]


@dataclass
class SessionMeta:
    path: str
    final_session_name: str = ""
    models: list[str] = field(default_factory=list)

    @property
    def single_model(self) -> str | None:
        return self.models[0] if len(self.models) == 1 else None


@dataclass
class SessionFilter:
    allowed_models: list[str] | None = None
    require_single_model: bool = True
    session_name_prefixes: list[str] | None = None

    def matches(self, meta: SessionMeta) -> bool:
        if self.require_single_model and meta.single_model is None:
            return False
        if self.session_name_prefixes:
            name = meta.final_session_name.lower()
            prefixes = [p.lower() for p in self.session_name_prefixes]
            if not any(name.startswith(prefix) for prefix in prefixes):
                return False
        if self.allowed_models is not None:
            model = meta.single_model if self.require_single_model else None
            if model not in set(self.allowed_models):
                return False
        return True


def scan_session(path: str | Path) -> SessionMeta:
    name = ""
    models: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            typ = obj.get("type")
            if typ == "session_info":
                value = obj.get("name", "") or ""
                if value:
                    name = value
            elif typ == "model_change":
                model = normalize_model_name(obj.get("modelId"))
                if model and model not in models:
                    models.append(model)
            elif typ == "message":
                msg = obj.get("message", {})
                if msg.get("role") == "assistant":
                    model = normalize_model_name(msg.get("model"))
                    if model and model not in models:
                        models.append(model)

    return SessionMeta(path=str(path), final_session_name=name, models=models)


def final_session_name(path: str | Path) -> str:
    return scan_session(path).final_session_name


def is_issue_session(path: str | Path) -> bool:
    return final_session_name(path).lower().startswith("issue:")


def distinct_models(path: str | Path) -> list[str]:
    return scan_session(path).models


def single_model_exact(path: str | Path) -> str | None:
    return scan_session(path).single_model


def collect_filtered_paths(
    data_root: Path,
    session_filter: SessionFilter,
) -> tuple[dict[str, set[str]], Counter, list[SessionMeta]]:
    selected: dict[str, set[str]] = {}
    counts: Counter = Counter()
    metas: list[SessionMeta] = []

    for path in sorted(data_root.glob("*.jsonl")):
        meta = scan_session(path)
        if not session_filter.matches(meta):
            continue
        metas.append(meta)
        if meta.single_model is not None:
            counts[meta.single_model] += 1
            selected.setdefault(meta.single_model, set()).add(meta.path)

    return selected, counts, metas


# Backwards-compatible wrapper used by the copied builders.
def collect_single_model_paths(
    data_root: Path,
    allowed_models: list[str] | None = None,
    require_issue: bool = False,
) -> tuple[dict[str, set[str]], Counter]:
    filt = SessionFilter(
        allowed_models=allowed_models,
        require_single_model=True,
        session_name_prefixes=["Issue:"] if require_issue else None,
    )
    selected, counts, _ = collect_filtered_paths(data_root, filt)
    return selected, counts
