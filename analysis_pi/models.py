"""
Shared taxonomy + dynamic model helpers for Pi transcript analysis.

The intent taxonomy is intentionally kept identical to the original SWE-Agent
analysis so the resulting charts stay comparable. Only the harness-specific
step parsing changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from analysis.models import (  # Reuse the canonical taxonomy unchanged.
    BASE_INTENTS,
    HIGH_LEVEL_CATEGORIES,
    HIGH_LEVEL_COLORS,
    HIGH_LEVEL_LETTER,
    INTENT_DESCRIPTIONS,
    INTENT_DISPLAY_NAMES,
    INTENT_TO_HIGH_LEVEL,
    LETTER_COLORS,
    LETTER_TO_NAME,
    ORDERED_LETTERS,
    PHASES,
)

# Kept for compatibility with copied scripts. Pi analysis discovers models
# dynamically from the transcript contents rather than a fixed registry.
MODELS: dict[str, dict] = {}

_FALLBACK_COLORS = [
    "#6a8da8",
    "#b8785e",
    "#6a9a6a",
    "#8a6a9a",
    "#9a7a5a",
    "#5a8a8a",
    "#a07a9a",
    "#7a8f5a",
]


def normalize_model_name(model: str | None) -> str:
    if not model:
        return "unknown"
    m = model.strip()
    for prefix in ("global.anthropic.", "us.anthropic."):
        if m.startswith(prefix):
            m = m[len(prefix):]
    # Normalize a couple of observed dot variants for Claude.
    m = m.replace("claude-opus-4.6", "claude-opus-4-6")
    m = m.replace("claude-sonnet-4.6", "claude-sonnet-4-6")
    m = m.replace("claude-opus-4.5", "claude-opus-4-5")
    m = m.replace("claude-sonnet-4.5", "claude-sonnet-4-5")
    return m


def model_label(model: str) -> str:
    return model


def model_color(model: str) -> str:
    key = model.lower()
    if "claude" in key:
        return "#b8785e"
    if "gpt" in key or "codex" in key:
        return "#6a8da8"
    if "gemini" in key or "google" in key:
        return "#6a9a6a"
    if "glm" in key:
        return "#8a6a9a"
    if "kimi" in key or "moonshot" in key:
        return "#9a7a5a"
    if "qwen" in key:
        return "#5a8a8a"
    idx = hashlib.md5(model.encode()).digest()[0] % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def build_model_registry(models: list[str]) -> dict[str, dict[str, str]]:
    return {
        m: {
            "label": model_label(m),
            "color": model_color(m),
        }
        for m in models
    }


def infer_repo_name(cwd: str | None) -> str:
    if not cwd:
        return "unknown"
    return Path(cwd).name or "unknown"
