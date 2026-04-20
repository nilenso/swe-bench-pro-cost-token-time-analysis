"""
Pi transcript classifier.

This adapts the original SWE-Agent intent taxonomy to Pi session transcripts.
The taxonomy stays the same; only the mapping from Pi tool calls to intent
labels changes.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import classify_intent as ci  # noqa: E402

from .models import HIGH_LEVEL_LETTER, INTENT_TO_HIGH_LEVEL, infer_repo_name, normalize_model_name

SOURCE_EDIT_INTENTS = ci.SOURCE_EDIT_INTENTS
SEQUENCE_VERIFY_INTENTS = ci.SEQUENCE_VERIFY_INTENTS

_DOC_HINTS = ("summary", "readme", "changes", "implementation", "notes", "design")


@dataclass
class FileResult:
    model: str
    path: str
    instance_id: str
    repo: str
    session_name: str = ""

    base_intents: list[str] = field(default_factory=list)
    high_intents: list[str] = field(default_factory=list)
    high_letters: list[str] = field(default_factory=list)
    verify_outcomes: list[str] = field(default_factory=list)
    seq_labels: list[str] = field(default_factory=list)

    base_intent_counts: dict[str, int] = field(default_factory=dict)
    high_intent_counts: dict[str, int] = field(default_factory=dict)
    verify_outcome_counts: dict[str, int] = field(default_factory=dict)
    seq_label_counts: dict[str, int] = field(default_factory=dict)

    positions: dict[str, float | None] = field(default_factory=dict)

    steps: int = 0
    exit_status: str = ""
    submitted: bool = False
    resolved: bool = False
    completed: bool = False
    work_done: bool = False

    phase_profile: dict[str, list[float]] = field(default_factory=dict)
    high_seq: str = ""
    bigram_counts: dict[str, int] = field(default_factory=dict)


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _basename(path: str) -> str:
    return Path(path).name.lower()


def _contains_any(text: str, needles: tuple[str, ...] | list[str]) -> bool:
    return any(n in text for n in needles)


def _session_name(events: list[dict]) -> str:
    for event in events:
        if event.get("type") == "session_info":
            return event.get("name", "") or ""
    return ""


def _session_model(events: list[dict]) -> str:
    for event in events:
        if event.get("type") == "message":
            msg = event.get("message", {})
            if msg.get("role") == "assistant" and msg.get("model"):
                return normalize_model_name(msg.get("model"))
    for event in events:
        if event.get("type") == "model_change" and event.get("modelId"):
            return normalize_model_name(event.get("modelId"))
    return "unknown"


def _session_repo(events: list[dict]) -> str:
    for event in events:
        if event.get("type") == "session":
            return infer_repo_name(event.get("cwd"))
    return "unknown"


def _last_assistant_stop_reason(events: list[dict]) -> str:
    stop = ""
    for event in events:
        if event.get("type") == "message":
            msg = event.get("message", {})
            if msg.get("role") == "assistant":
                stop = msg.get("stopReason") or stop
    return stop


def _flatten_tool_result_text(message: dict) -> str:
    content = message.get("content", []) or []
    parts: list[str] = []
    for block in content:
        typ = block.get("type")
        if typ == "text":
            parts.append(block.get("text", ""))
        elif typ == "image":
            parts.append("[image]")
        else:
            parts.append(str(block))
    if message.get("details"):
        details = message["details"]
        if isinstance(details, dict) and details.get("diff"):
            parts.append(details["diff"])
    return "\n".join(p for p in parts if p)


def _tool_result_map(events: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for event in events:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "toolResult":
            continue
        call_id = msg.get("toolCallId")
        if call_id and call_id not in out:
            out[call_id] = msg
    return out


def _normalize_edits(arguments: dict) -> list[dict]:
    edits = arguments.get("edits")
    if isinstance(edits, list) and edits:
        return edits
    old = arguments.get("oldText", arguments.get("old_string"))
    new = arguments.get("newText", arguments.get("new_string"))
    if old is not None or new is not None:
        return [{"oldText": old or "", "newText": new or ""}]
    return []


def _classify_read(arguments: dict, observation: str, is_error: bool) -> str:
    if is_error:
        return "read-file-failed"
    path = arguments.get("path", "") or ""
    base = _basename(path)
    if "offset" in arguments or "limit" in arguments:
        return "read-file-range"
    if ci.TEST_FILE_RE.match(base):
        return "read-test-file"
    if base in ci.CONFIG_FILES or base in {"dockerfile", ".env", ".gitignore"}:
        return "read-config-file"
    obs_lower = observation.lower()
    if "too large" in obs_lower or "truncated" in obs_lower:
        return "read-file-full(truncated)"
    return "read-file-full"


def _classify_write(arguments: dict) -> str:
    path = arguments.get("path", "") or ""
    filename = _basename(path)
    if _contains_any(filename, ("repro", "reproduce", "demo")):
        return "create-repro-script"
    if _contains_any(filename, ("test_", "test.py", "test.js", "test.go", ".test.", ".spec.")):
        return "create-test-script"
    if _contains_any(filename, ("verify", "check", "validate", "edge_case")):
        return "create-verify-script"
    if _contains_any(filename, _DOC_HINTS):
        return "create-documentation"
    return "create-file"


def _classify_edit(arguments: dict, is_error: bool) -> str:
    path = arguments.get("path", "") or ""
    filename = _basename(path)
    is_test_or_repro = _contains_any(
        filename, ("test_", "repro", "verify", "check", ".test.", ".spec.")
    )
    if is_error:
        return "edit-test-or-repro(failed)" if is_test_or_repro else "edit-source(failed)"
    edits = _normalize_edits(arguments)
    if edits and any((e.get("oldText") or e.get("old_string") or "") == "" for e in edits):
        return "insert-source"
    if is_test_or_repro:
        return "edit-test-or-repro"
    return "edit-source"


_WRAPPER_ONLY_RE = re.compile(
    r"^(?:set\s+-[a-zA-Zuoepx\s-]+|[A-Za-z_][A-Za-z0-9_]*=.*|then|fi|do|done)$"
)


def _meaningful_shell_command(cmd: str) -> str:
    cmd = ci._unwrap_command(cmd)
    cmd = ci._strip_leading_env_and_timeout(cmd)
    lines = [line.strip() for line in cmd.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            continue
        if _WRAPPER_ONLY_RE.match(line):
            continue
        kept.append(line)
    if not kept:
        return cmd.strip()
    if len(kept) > 1 and kept[0].startswith("cd ") and "&&" not in kept[0]:
        kept = kept[1:]
    return "\n".join(kept).strip()


def _extract_tsx_inline_snippet(cmd: str) -> str:
    for token in (" --eval ", " -e "):
        idx = cmd.find(token)
        if idx >= 0:
            snippet = cmd[idx + len(token):].strip()
            if len(snippet) >= 2 and snippet[0] == snippet[-1] and snippet[0] in {'"', "'"}:
                snippet = snippet[1:-1]
            return snippet.strip()
    if "tsx <<" in cmd:
        lines = cmd.splitlines()
        if len(lines) <= 1:
            return ""
        body = lines[1:]
        while body and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", body[-1].strip()):
            body.pop()
        return "\n".join(body).strip()
    return ""


def _classify_pi_git_intent(action: str) -> str:
    head = action.splitlines()[0].strip().lower() if action.strip() else ""
    if head.startswith("gh "):
        return "git-github-context"

    sub = ci._get_git_subcommand(action)
    if sub == "diff":
        return "git-diff-review"
    if sub == "push":
        return "git-publish"
    if sub in {"fetch", "pull", "rebase", "merge", "cherry-pick", "am"}:
        return "git-sync-integrate"
    if sub in {
        "add",
        "commit",
        "stash",
        "reset",
        "checkout",
        "switch",
        "restore",
        "apply",
        "rm",
        "mv",
        "clean",
    }:
        return "git-local-state-change"
    if sub in {
        "status",
        "show",
        "log",
        "branch",
        "tag",
        "worktree",
        "remote",
        "rev-parse",
        "describe",
        "config",
        "ls-files",
        "blame",
        "submodule",
    }:
        return "git-repo-inspect"
    return ""


_BRAVE_SEARCH_RE = re.compile(r"(?:^|\s|/)pi-skills/brave-search/search\.js\b")


def _classify_bash_like(command: str, observation: str, is_error: bool) -> tuple[str, str]:
    action = _meaningful_shell_command(command)
    head = action.lower().splitlines()[0].strip() if action.strip() else ""

    # Pi-specific: brave-search skill path comes through the bash tool.
    if _BRAVE_SEARCH_RE.search(action):
        return "web-search", action

    # Pi bash is the only way to start tmux or hit URLs; surface these
    # rather than letting them fall through to bash-other.
    if re.match(r"^tmux\b", head):
        return "tmux-session", action
    if re.match(r"^curl\b", head):
        return "fetch-url", action

    pi_git_intent = _classify_pi_git_intent(action)
    if pi_git_intent:
        return pi_git_intent, action

    if _contains_any(
        head,
        (
            "npm run check",
            "pnpm check",
            "yarn check",
            "biome check",
            "eslint ",
            "tsc --noemit",
            "tsgo --noemit",
            "npm run lint",
            "pnpm lint",
            "yarn lint",
            "npm run generate-models",
        ),
    ):
        return "compile-build", action

    if _contains_any(head, ("vitest", "jest", "mocha", "ava ", "tap ")):
        if _contains_any(head, (" --run ", " test/", ".test.", ".spec.", " -t ", " --grep ")):
            return "run-test-specific", action
        return "run-test-suite", action

    if re.search(r"\b(?:npx\s+)?tsx\b", head):
        snippet = _extract_tsx_inline_snippet(action)
        if snippet:
            return ci._classify_inline_snippet(snippet), action
        return ci._classify_script_name(ci._extract_script_filename(action)), action

    if _contains_any(head, ("docker run", "docker build", "docker exec", "podman run")):
        return "compile-build", action

    if _contains_any(head, ("mktemp", "tmpdir=$(", "pid=$(cat ")):
        return "file-cleanup", action

    if head.startswith("npm ls") or head.startswith("pnpm ls") or head.startswith("yarn list"):
        return "install-deps", action

    base = ci.classify_step(action, observation)
    if base == "bash-other" and is_error:
        return "bash-command(failed)", action
    return base, action


def _synthetic_shell_for_tool(name: str, arguments: dict) -> str:
    if name == "bash":
        return arguments.get("command", "") or ""
    if name == "ls":
        path = arguments.get("path", "") or "."
        return f"ls {path}"
    if name == "find":
        path = arguments.get("path", ".") or "."
        pattern = arguments.get("pattern", "**/*") or "**/*"
        return f"find {path} -name {json.dumps(pattern)}"
    if name == "grep":
        path = arguments.get("path", ".") or "."
        pattern = arguments.get("pattern", "") or ""
        return f"rg -n {json.dumps(pattern)} {json.dumps(path)}"
    if name == "finish_and_exit":
        return "submit"
    if name in {"hello", "greet"}:
        return "echo hello"
    return ""


def _classify_tool_call(name: str, arguments: dict, observation: str, is_error: bool) -> tuple[str, str]:
    if name == "read":
        base = _classify_read(arguments, observation, is_error)
        action = f"str_replace_editor view {arguments.get('path', '')}"
        return base, action
    if name == "write":
        base = _classify_write(arguments)
        action = f"str_replace_editor create {arguments.get('path', '')}"
        return base, action
    if name == "edit":
        base = _classify_edit(arguments, is_error)
        edits = _normalize_edits(arguments)
        if base == "insert-source" or (edits and any((e.get('oldText') or '') == '' for e in edits)):
            action = f"str_replace_editor insert {arguments.get('path', '')}"
        else:
            action = f"str_replace_editor str_replace {arguments.get('path', '')}"
        return base, action
    if name == "todo":
        return "bash-other", "todo"
    if name == "watch_plans_start":
        return "bash-other", "watch_plans_start"
    if name == "":
        return "empty", ""

    shell = _synthetic_shell_for_tool(name, arguments)
    if shell:
        return _classify_bash_like(shell, observation, is_error)

    if is_error:
        return "bash-command(failed)", name
    return "bash-other", name


def classify_file(path: str, phase_bins: int = 20) -> FileResult | None:
    events = _load_jsonl(path)
    if not events:
        return None

    tool_results = _tool_result_map(events)
    trajectory: list[dict] = []
    base_intents: list[str] = []
    finish_called = False

    for event in events:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []) or []:
            if block.get("type") != "toolCall":
                continue
            name = block.get("name") or ""
            arguments = block.get("arguments", {}) or {}
            result = tool_results.get(block.get("id", ""), {})
            observation = _flatten_tool_result_text(result)
            is_error = bool(result.get("isError"))
            if is_error and observation:
                observation = f"ERROR:\n{observation}"
            base_intent, action = _classify_tool_call(name, arguments, observation, is_error)
            base_intents.append(base_intent)
            trajectory.append({"action": action, "observation": observation})
            if base_intent == "submit":
                finish_called = True

    if not trajectory:
        return None

    hierarchical = [f"{INTENT_TO_HIGH_LEVEL.get(intent, 'other')}.{intent}" for intent in base_intents]
    verify_outcomes = [
        ci.classify_verify_outcome(step.get("action", ""), step.get("observation", ""), base)
        for step, base in zip(trajectory, base_intents)
    ]
    seq_labels = ci.classify_sequence_layer(trajectory, base_intents, verify_outcomes)

    highs = [h.split(".", 1)[0] for h in hierarchical]
    high_letters = [HIGH_LEVEL_LETTER.get(h, "?") for h in highs]
    high_seq = "".join(high_letters)
    n = len(base_intents)

    base_intent_counts = dict(Counter(base_intents))
    high_intent_counts = dict(Counter(highs))
    verify_outcome_counts = dict(Counter(verify_outcomes))
    seq_label_counts = dict(Counter(seq_labels))

    bigram_c: Counter = Counter()
    for i in range(len(high_letters) - 1):
        bigram_c[high_letters[i] + high_letters[i + 1]] += 1

    def pct(idx: int) -> float | None:
        return round(idx / max(n - 1, 1) * 100, 1) if idx >= 0 else None

    first_edit = next((i for i, b in enumerate(base_intents) if b in ci.SOURCE_EDIT_INTENTS), -1)
    last_edit = next((i for i in range(n - 1, -1, -1) if base_intents[i] in ci.SOURCE_EDIT_INTENTS), -1)
    first_verify = next((i for i, b in enumerate(base_intents) if b in ci.SEQUENCE_VERIFY_INTENTS), -1)
    first_verify_pass = next((i for i in range(n) if verify_outcomes[i] == "pass"), -1)
    submit_idx = next((i for i, b in enumerate(base_intents) if b == "submit"), -1)

    positions = {
        "first_edit": pct(first_edit),
        "last_edit": pct(last_edit),
        "first_verify": pct(first_verify),
        "first_verify_pass": pct(first_verify_pass),
        "submit": pct(submit_idx),
    }

    phase_profile: dict[str, list[float]] = {}
    if n >= 5:
        for letter in HIGH_LEVEL_LETTER.values():
            counts_in_bin = []
            for b in range(phase_bins):
                start = int(b * n / phase_bins)
                end = int((b + 1) * n / phase_bins)
                segment = high_letters[start:end]
                counts_in_bin.append(segment.count(letter) / len(segment) if segment else 0.0)
            phase_profile[letter] = counts_in_bin

    stop_reason = _last_assistant_stop_reason(events)
    completed = stop_reason == "stop" or finish_called

    return FileResult(
        model=_session_model(events),
        path=path,
        instance_id=Path(path).stem,
        repo=_session_repo(events),
        session_name=_session_name(events),
        base_intents=base_intents,
        high_intents=highs,
        high_letters=high_letters,
        verify_outcomes=verify_outcomes,
        seq_labels=seq_labels,
        base_intent_counts=base_intent_counts,
        high_intent_counts=high_intent_counts,
        verify_outcome_counts=verify_outcome_counts,
        seq_label_counts=seq_label_counts,
        positions=positions,
        steps=n,
        exit_status=stop_reason,
        submitted=finish_called,
        resolved=False,
        completed=completed,
        work_done="seq-first-all-pass" in seq_labels or "seq-work-done" in seq_labels,
        phase_profile=phase_profile,
        high_seq=high_seq,
        bigram_counts=dict(bigram_c),
    )
