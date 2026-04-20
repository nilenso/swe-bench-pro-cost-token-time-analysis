"""
Per-issue task resolution classifier (scheme 3).

Unit of analysis is (model, issue-number): a maintainer's intent to resolve a
specific GitHub issue with a given model. Resolution counts any of:

- git push (success)
- gh issue close / gh pr merge
- gh issue comment with explicit content (post / triage / won't-fix / duplicate)
- user instruction to close/comment-close/triage on the final turn (maintainer's
  terminal decision on the task, regardless of whether the agent followed through)

Issues are joined across sessions by the issue/PR number found either in the
session name or in the first user message. Sessions that share a model AND an
issue number are rolled up: the issue is resolved if any such session resolved.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .models import normalize_model_name
from .session_filter import SessionFilter, collect_filtered_paths


URL_RE = re.compile(r"https://github\.com/[^\s\)\]]+/(issues|pull|pulls)/(\d+)")

# User instructions that represent a terminal maintainer decision on the issue.
# Three categories:
#   (a) close/comment verb + issue/duplicate noun (with typo-tolerant determiner)
#   (b) comment-posting instructions (triage)
#   (c) won't-fix / not-our-problem dismissals
_CLOSE_INSTRUCT_RE = re.compile(
    r"\b("
    # (a) close/closed + optional 2-6 char determiner + issue/duplicate
    # tolerates typos like "closed hte issue"
    r"(?:close|closed)(?:\s+\S{1,6})?\s+(?:issue|duplicate|as\s+duplicate)"
    r"|close it(?:\s+as\s+\S{1,12})?"
    r"|comment and close|close and comment"
    # (b) triage-comment instructions
    r"|post (?:that|this|a comment) (?:on|to|in) (?:the )?issue"
    r"|post (?:that|this) on the issue"
    r"|comment on the issue"
    r"|add a comment to the issue"
    r"|leave a comment on (?:the )?issue"
    r"|(?:mark|flag) (?:as |it as )?(?:duplicate|resolved|won'?t ?fix|wontfix|not a bug)"
    # (c) won't-fix / dismissal patterns
    r"|won'?t ?fix|wontfix"
    r"|by design"
    r"|working as intended"
    r"|not (?:our|my) (?:bug|problem|issue|responsibility|concern)"
    r"|(?:i )?don'?t care"
    r"|(?:it'?s |that'?s )?on (?:the )?user'?s?\s+(?:end|side|problem)"
    r"|(?:user|they) (?:has to|have to|need(?:s)? to) [^.]{1,80}(?:extension|sdk|their (?:code|end|side))"
    r"|cannot reproduce|can'?t reproduce|not reproducible"
    r")\b",
    re.I,
)

# Commit-like regex helpers.
_GIT_COMMIT_RE = re.compile(r"\bgit (?:-c [^\s]+\s+)?commit\b")
_GIT_PUSH_RE = re.compile(r"\bgit (?:-c [^\s]+\s+)?push\b")


@dataclass
class ResolvedSignals:
    path: str
    model: str
    session_name: str = ""
    issue_keys: set[str] = field(default_factory=set)
    n_user_messages: int = 0
    last_user_text: str = ""

    # Ship signals.
    any_git_push_success: bool = False
    any_gh_issue_close: bool = False
    any_gh_pr_merge: bool = False
    any_gh_issue_comment: bool = False

    # Maintainer-decision signal.
    last_user_close_instruct: bool = False
    any_user_close_instruct: bool = False

    resolved: bool = False
    resolution_kind: str = ""  # push | gh_close | gh_merge | gh_comment | user_close | ""


def _first_user_text(events: list[dict]) -> str:
    seen = 0
    for e in events:
        if e.get("type") != "message":
            continue
        msg = e.get("message", {})
        if msg.get("role") != "user":
            continue
        seen += 1
        if seen == 1:
            return " ".join(
                b.get("text", "") for b in msg.get("content", []) or [] if b.get("type") == "text"
            )[:1000]
    return ""


def _extract_issue_keys(session_name: str, first_user: str) -> set[str]:
    keys: set[str] = set()
    for text in (session_name, first_user):
        for m in URL_RE.finditer(text or ""):
            keys.add(m.group(2))
    return keys


def _flatten_tool_result_text(msg: dict) -> str:
    parts: list[str] = []
    for block in msg.get("content", []) or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def classify_resolution(path: str, model: str | None = None) -> ResolvedSignals:
    with open(path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    session_name = ""
    for e in events:
        if e.get("type") == "session_info" and e.get("name"):
            session_name = e["name"]

    tool_results: dict[str, dict] = {}
    for e in events:
        if e.get("type") != "message":
            continue
        msg = e.get("message", {})
        if msg.get("role") == "toolResult":
            cid = msg.get("toolCallId")
            if cid:
                tool_results[cid] = msg

    first_user = _first_user_text(events)
    sig = ResolvedSignals(
        path=path,
        model=model or "unknown",
        session_name=session_name,
        issue_keys=_extract_issue_keys(session_name, first_user),
    )

    user_msgs: list[str] = []

    for e in events:
        if e.get("type") != "message":
            continue
        msg = e.get("message", {})
        role = msg.get("role")
        content = msg.get("content", []) or []
        if role == "user":
            txt = " ".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            user_msgs.append(txt)
            if _CLOSE_INSTRUCT_RE.search(txt):
                sig.any_user_close_instruct = True
            continue
        if role != "assistant":
            continue
        for b in content:
            if b.get("type") != "toolCall":
                continue
            name = b.get("name") or ""
            args = b.get("arguments", {}) or {}
            tr = tool_results.get(b.get("id", ""), {})
            is_error = bool(tr.get("isError"))
            obs = _flatten_tool_result_text(tr)

            if name == "bash":
                cmd = args.get("command", "") or ""
                low = cmd.lower()

                if _GIT_PUSH_RE.search(cmd) and not is_error:
                    obs_low = obs.lower()
                    if "rejected" not in obs_low and "error:" not in obs_low:
                        sig.any_git_push_success = True

                if "gh issue close" in low and not is_error:
                    sig.any_gh_issue_close = True

                if "gh pr merge" in low and not is_error:
                    sig.any_gh_pr_merge = True

                if ("gh issue comment" in low or "gh pr comment" in low) and not is_error:
                    sig.any_gh_issue_comment = True

    sig.n_user_messages = len(user_msgs)
    if user_msgs:
        sig.last_user_text = user_msgs[-1][:600]
        if len(user_msgs) >= 2 and _CLOSE_INSTRUCT_RE.search(user_msgs[-1]):
            sig.last_user_close_instruct = True

    # Decide resolution.
    if sig.any_git_push_success:
        sig.resolved = True
        sig.resolution_kind = "push"
    elif sig.any_gh_pr_merge:
        sig.resolved = True
        sig.resolution_kind = "gh_merge"
    elif sig.any_gh_issue_close:
        sig.resolved = True
        sig.resolution_kind = "gh_close"
    elif sig.any_gh_issue_comment and sig.any_user_close_instruct:
        sig.resolved = True
        sig.resolution_kind = "gh_comment"
    elif sig.last_user_close_instruct:
        sig.resolved = True
        sig.resolution_kind = "user_close"

    return sig


# ---------------------------------------------------------------------------
# Aggregation (scheme 3): per (model, issue#)
# ---------------------------------------------------------------------------


@dataclass
class ModelResolutionStats:
    model: str
    n_sessions: int = 0
    n_issues_attempted: int = 0
    n_issues_resolved: int = 0
    kind_counts: dict[str, int] = field(default_factory=dict)
    # Sessions included in the analysis (for drill-down).
    sessions: list[ResolvedSignals] = field(default_factory=list)
    issues_resolved: set[str] = field(default_factory=set)
    issues_attempted: set[str] = field(default_factory=set)

    @property
    def resolve_rate(self) -> float:
        if not self.n_issues_attempted:
            return 0.0
        return self.n_issues_resolved / self.n_issues_attempted * 100.0


def _scan_all_sessions(data_root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(data_root.glob("*.jsonl")):
        name = ""
        first_user = ""
        models: list[str] = []
        n_user = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("type")
                if t == "session_info" and obj.get("name"):
                    name = obj["name"]
                elif t == "model_change":
                    m = normalize_model_name(obj.get("modelId"))
                    if m and m not in models:
                        models.append(m)
                elif t == "message":
                    msg = obj.get("message", {})
                    if msg.get("role") == "user":
                        n_user += 1
                        if n_user == 1:
                            first_user = " ".join(
                                b.get("text", "")
                                for b in msg.get("content", []) or []
                                if b.get("type") == "text"
                            )[:1000]
                    elif msg.get("role") == "assistant":
                        m = normalize_model_name(msg.get("model"))
                        if m and m not in models:
                            models.append(m)
        rows.append(
            {
                "path": str(p),
                "name": name,
                "keys": _extract_issue_keys(name, first_user),
                "models": models,
                "single_model": models[0] if len(models) == 1 else None,
            }
        )
    return rows


def compute_resolution_by_model(
    data_root: Path,
    session_filter: SessionFilter,
) -> dict[str, ModelResolutionStats]:
    """
    Scheme-3 resolution rates. For each model, aggregate issues the filter picks
    for that model, unioned with any same-model sessions (even those the filter
    excluded) that reference the same issue number — so an issue is counted as
    resolved if the model shipped on it in ANY session.

    Returns {model: ModelResolutionStats}.
    """
    selected, _, _ = collect_filtered_paths(data_root, session_filter)
    all_rows = _scan_all_sessions(data_root)

    # Index every session by issue key.
    by_key_all: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        for k in r["keys"]:
            by_key_all[k].append(r)

    # Classify resolution for every path we might touch (filter paths + peers).
    wanted_paths: set[str] = set()
    for paths in selected.values():
        wanted_paths.update(paths)
    for r in all_rows:
        for k in r["keys"]:
            for peer in by_key_all[k]:
                wanted_paths.add(peer["path"])

    # Limit to sessions whose single-model is in the filter's allowed models
    # (we only need resolution signals for sessions that can credit an issue
    # to one of the tracked models).
    allowed = set(session_filter.allowed_models or [])
    path_to_model: dict[str, str] = {}
    for r in all_rows:
        if r["single_model"] and r["single_model"] in allowed and r["path"] in wanted_paths:
            path_to_model[r["path"]] = r["single_model"]

    resolved_sigs: dict[str, ResolvedSignals] = {}
    for path, model in path_to_model.items():
        resolved_sigs[path] = classify_resolution(path, model)

    # Build per-model stats.
    stats: dict[str, ModelResolutionStats] = {}
    for model, paths in selected.items():
        if model not in allowed:
            continue
        ms = ModelResolutionStats(model=model, n_sessions=len(paths))
        attempted: set[str] = set()
        resolved: set[str] = set()
        kind_counter: dict[str, int] = defaultdict(int)
        included: list[ResolvedSignals] = []
        for path in paths:
            sig = resolved_sigs.get(path)
            if sig is None:
                sig = classify_resolution(path, model)
                resolved_sigs[path] = sig
            included.append(sig)
            attempted.update(sig.issue_keys)
            if sig.resolved:
                resolved.update(sig.issue_keys)
                kind_counter[sig.resolution_kind] += 1

        # Same-model peers (sessions outside the filter that share issue numbers).
        for key in list(attempted):
            for peer in by_key_all.get(key, []):
                if peer["single_model"] != model:
                    continue
                peer_sig = resolved_sigs.get(peer["path"])
                if peer_sig is None:
                    peer_sig = classify_resolution(peer["path"], model)
                    resolved_sigs[peer["path"]] = peer_sig
                if peer_sig.resolved:
                    resolved.add(key)

        ms.issues_attempted = attempted
        ms.issues_resolved = resolved
        ms.n_issues_attempted = len(attempted)
        ms.n_issues_resolved = len(resolved)
        ms.kind_counts = dict(kind_counter)
        ms.sessions = included
        stats[model] = ms

    return stats
