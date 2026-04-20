"""
Shared taxonomy + dynamic model helpers for Pi transcript analysis.

The high-level taxonomy stays aligned with the original SWE-Agent analysis so
summary charts remain comparable. Pi uses a more semantic low-level git
subtaxonomy so the intent decomposition is more informative for this harness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from analysis.models import (
    HIGH_LEVEL_CATEGORIES,
    HIGH_LEVEL_COLORS,
    HIGH_LEVEL_LETTER,
    INTENT_DISPLAY_NAMES as _SWE_INTENT_DISPLAY_NAMES,
    INTENT_DESCRIPTIONS as _SWE_INTENT_DESCRIPTIONS,
    INTENT_TO_HIGH_LEVEL as _SWE_INTENT_TO_HIGH_LEVEL,
    LETTER_COLORS,
    LETTER_TO_NAME,
    ORDERED_LETTERS,
    PHASES,
)

# ---------------------------------------------------------------------------
# Pi-specific intents.
#
# Pi's harness exposes dedicated `read` / `edit` / `write` tools (not
# str_replace_editor), plus a bash tool that's often used for tmux,
# curl, and a brave-search skill. Pi also benefits from a more semantic
# decomposition of git behavior inside the low-level intent table: GitHub
# context, repo inspection, diff review, sync/integrate, local mutation, and
# publish.
# ---------------------------------------------------------------------------
_PI_EXTRA_INTENT_TO_HIGH_LEVEL: dict[str, str] = {
    "read-file-failed": "failed",
    "edit-source(failed)": "failed",
    "edit-test-or-repro(failed)": "failed",
    "run-inline-snippet": "verify",
    "web-search": "search",
    "tmux-session": "housekeeping",
    "fetch-url": "other",
    "git-github-context": "git",
    "git-repo-inspect": "git",
    "git-diff-review": "git",
    "git-sync-integrate": "git",
    "git-local-state-change": "git",
    "git-publish": "git",
}

INTENT_TO_HIGH_LEVEL: dict[str, str] = {
    **_SWE_INTENT_TO_HIGH_LEVEL,
    **_PI_EXTRA_INTENT_TO_HIGH_LEVEL,
}

BASE_INTENTS: list[str] = sorted(INTENT_TO_HIGH_LEVEL.keys())

# Pi-flavoured descriptions: strip str_replace_editor/applypatch language
# and name the actual Pi tool (read / edit / write / bash) where relevant.
_PI_INTENT_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "read-file-full": "Pi `read` tool: view an entire file",
    "read-file-range": "Pi `read` tool with `offset`/`limit`: view a line range",
    "read-file-full(truncated)": "Pi `read` tool: file too large, result got truncated",
    "read-test-file": "Pi `read` tool: view a test file (test_*, _test.*, conftest)",
    "read-config-file": "Pi `read` tool: view package.json, pytest.ini, go.mod, Makefile, etc.",
    "read-file-failed": "Pi `read` tool returned an error (missing path, permission, etc.)",
    "view-directory": "(n/a for Pi — Pi has no str_replace_editor view-directory)",
    "edit-source": "Pi `edit` tool: str_replace on a non-test, non-repro source file",
    "insert-source": "Pi `edit` tool with empty `oldText`: insert into a source file",
    "edit-test-or-repro": "Pi `edit` tool: modify an existing test or repro file; usually verification-support work, not an executed check",
    "edit-source(failed)": "Pi `edit` tool returned an error on a source file",
    "edit-test-or-repro(failed)": "Pi `edit` tool returned an error on a test or repro file",
    "apply-patch": "(n/a for Pi — no applypatch equivalent in this harness)",
    "undo-edit": "(n/a for Pi — no str_replace_editor undo_edit)",
    "create-file": "Pi `write` tool: create a file that doesn't match repro/test/verify/doc patterns",
    "create-repro-script": "Pi `write` tool: file named repro*, reproduce*, demo*",
    "create-test-script": "Pi `write` tool: create a repo test or regression file; verification-support work, not an executed check",
    "create-verify-script": "Pi `write` tool: create an ad hoc verify/check/validate script",
    "create-documentation": "Pi `write` tool: file named summary, readme, changes, implementation",
    "web-search": "Pi `brave-search` skill (bash call to search.js) for web lookups",
    "tmux-session": "Pi bash: `tmux` for background / long-running processes",
    "fetch-url": "Pi bash: `curl` for HTTP requests (APIs, downloads, webhooks)",
    "run-inline-snippet": "Pi bash: residual inline `tsx`/`node`/`python` snippet; in Pi this is treated as verification because these ad hoc one-offs are usually used to inspect or check behavior",
    "run-inline-verify": "Pi bash: inline `tsx`/`node`/`python` probe that imports repo code or runs ad hoc assertions / behavioral checks",
    "run-test-specific": "Pi bash: targeted test invocation aimed at a specific file, filter, or named subset",
    "run-test-suite": "Pi bash: broad test-runner invocation (`pytest`, `go test`, `npm test`, `jest`, `mocha`) when not recognized as targeted",
    "run-verify-script": "Pi bash: run a named verify/check/validate script",
    "run-custom-script": "Pi bash: run a named project script that does not match the repro/test/verify filename heuristics; sometimes still used for ad hoc verification",
    "compile-build": "Pi bash: repo-native check / build / typecheck command (often `npm run check`, `biome`, `eslint`, `tsc`, `tsgo`, `go build`, or `make`); usually verification-oriented, though some commands may auto-fix files",
    "git-github-context": "Pi git workflow: `gh issue` / `gh pr` / `gh api` commands for reading or updating GitHub task context",
    "git-repo-inspect": "Pi git workflow: inspect local repo state via `git status`, `log`, `show`, `branch`, `worktree`, etc.",
    "git-diff-review": "Pi git workflow: inspect current changes via `git diff`",
    "git-sync-integrate": "Pi git workflow: integrate upstream changes via `git fetch`, `pull`, `rebase`, `merge`, or `cherry-pick`",
    "git-local-state-change": "Pi git workflow: mutate local repo state via `git add`, `commit`, `stash`, `reset`, `checkout`, `switch`, etc.",
    "git-publish": "Pi git workflow: publish finished work via `git push`",
    "bash-command(failed)": "Pi bash tool exited non-zero (excluding specialised failure buckets)",
    "read-via-bash(failed)": "`cat`/`head`/`sed` run via Pi bash that hit shell errors",
    "submit": "Pi `finish_and_exit` tool",
}

INTENT_DESCRIPTIONS: dict[str, str] = {
    **_SWE_INTENT_DESCRIPTIONS,
    **_PI_INTENT_DESCRIPTION_OVERRIDES,
    # Ensure every Pi-extra intent has at least a default description.
    **{
        intent: _PI_INTENT_DESCRIPTION_OVERRIDES.get(intent, intent)
        for intent in _PI_EXTRA_INTENT_TO_HIGH_LEVEL
    },
}

INTENT_DISPLAY_NAMES: dict[str, str] = {
    **_SWE_INTENT_DISPLAY_NAMES,
    "read-file-failed": "read (failed)",
    "edit-source(failed)": "edit source (failed)",
    "edit-test-or-repro(failed)": "edit test/repro (failed)",
    "web-search": "web search",
    "tmux-session": "tmux",
    "fetch-url": "curl / http",
    "git-github-context": "GitHub context",
    "git-repo-inspect": "repo inspect",
    "git-diff-review": "diff review",
    "git-sync-integrate": "sync / integrate",
    "git-local-state-change": "local state change",
    "git-publish": "publish",
}

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
