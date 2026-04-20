"""Deterministic user-message classification for Pi issue-session analysis.

This is intentionally tuned to the current Pi maintainer dataset rather than
trying to generalise. The goal is to capture how the maintainer intervenes in
agent trajectories over time, using a small, reproducible taxonomy.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_ORDER = [
    "task_brief",
    "authorize_work",
    "solution_steer",
    "evidence_or_repro",
    "qa_or_critique",
    "validation_request",
    "workflow_closeout",
]

CLASS_DESCRIPTIONS = {
    "task_brief": "Initial issue framing and operating instructions: what to read, how to analyse it, and whether implementation is allowed yet.",
    "authorize_work": "Explicit switch from analysis into implementation work, e.g. implement / fix it / make the change / do it.",
    "solution_steer": "Guidance that constrains or redirects the solution: architecture, scope, constraints, exact files, or preferred implementation shape.",
    "evidence_or_repro": "New evidence injected by the maintainer: repro details, logs, screenshots, pasted terminal output, file paths, alternate implementations, or environment observations.",
    "qa_or_critique": "Evaluation or correction of the model's reasoning/code quality: brittle?, terrible analysis, why did you do this, you missed the instruction, etc.",
    "validation_request": "Requests to test, verify, demonstrate, or confirm behaviour, including specific test commands and how-to-test questions.",
    "workflow_closeout": "Repository workflow and shipping instructions: commit, push, changelog, docs, issue comment, close issue, merge, or wrap it up.",
}


_TASK_BRIEF_RE = re.compile(
    r"^(analyze github issue\(s\):|\"?/is https://github\.com/.+/issues/\d+)|"
    r"for each issue:|read the issue in full|do not implement unless explicitly asked",
    re.I,
)

_WORKFLOW_RE = re.compile(
    r"("
    r"\bcommit and push\b|\bcommit an dpush\b|\bcommit with(?: closes| #closes)?(?:, push)?\b|"
    r"\bcommit the files\b|\bready to commit\b|^commit\b|^push\b|"
    r"\bclose issue\b|\bclose the issue\b|"
    r"\bleave a comment(?: on the issue)?\b|"
    r"\bin my tone\b|"
    r"\badd a changelog entry\b|\bchangelog entry\b|\bCHANGELOG\.md\b|\bchangelog\b|"
    r"\bdocs? in @\b|\bdocs? are in @\b|\bdocs? up-to-date\b|"
    r"\bmerge(?: into main| to main| via gh cli)?\b|"
    r"\bwrap it\b|\bjust update docs\b"
    r")",
    re.I,
)

_EVIDENCE_RE = re.compile(
    r"("
    r"^➜|"
    r"^/var/folders/|^/Users/|"
    r"screenshot\\|"
    r"still ?get the crash|stillget the crash|seeing this|now i get this|"
    r"git:\(|node packages/|pi-test\.sh|"
    r"failed to load extension|throw new Error|ERR_MODULE_NOT_FOUND|SyntaxError:|ENOENT:|"
    r"the other folder is|different impl in|same file you modified in this clone|"
    r"^ffb647c$|"
    r"^https://agentskills\.io/|^https://github\.com/"
    r")",
    re.I,
)

_VALIDATION_RE = re.compile(
    r"\b("
    r"test it|how can i test|run the test|run the specific test|"
    r"execute it|try locall|can i test|"
    r"what are the changes\?|"
    r"looks good|lgtm|works fine|seems to work|test works\?|"
    r"sleep for \d+ seconds"
    r")\b",
    re.I,
)

_QA_RE = re.compile(
    r"("
    r"terrible analysis|"
    r"isn't that brittle|"
    r"dude, what were your instructions|"
    r"i did not ask you|"
    r"you still don't understand|"
    r"noo, we need a smarter way|"
    r"why (?:this|would|did|in teh fuck)|"
    r"what in the fuck|where the fuck|"
    r"don't care\.|don't care,|"
    r"that seems like the simplest option|"
    r"we established that|"
    r"do i understand correctly|"
    r"are you sure this doesn't|"
    r"explain the test to me|"
    r"what were your instructions"
    r")",
    re.I,
)

_AUTHORIZE_RE = re.compile(
    r"\b("
    r"implement|"
    r"make the change|"
    r"do it|"
    r"fix it|fix this|"
    r"go to town|"
    r"apply the fix|"
    r"continue|"
    r"ok, fix|"
    r"ok, please implement|"
    r"ok, implement|"
    r"can you just implement it properly"
    r")\b",
    re.I,
)

_STEER_RE = re.compile(
    r"("
    r"\bwe need\b|\bwe should\b|\bi think we should\b|\bcan we\b|\bwhy not\b|"
    r"\bno backward compatibility\b|\bonly add\b|\bremove the\b|\bremove that file\b|"
    r"\bwhat about\b|\bwhat baout\b|\bany reason\b|"
    r"\bproblem:\b|\bnot ideal\b|\bok, then lets\b|\bok, then let's\b|"
    r"\bshould be\b|\bmust also\b|\bmust be\b|\bshould rename\b|\bshould use\b|"
    r"\badd a test\b|\bset up a minimal test\b|\bcheck if\b|\buse dynamic import\b|"
    r"\bjust use\b|\bjust implement\b|\bswitch to that\b|\bkill that\b|"
    r"\ball we need\b|\bthat too\b|\bmodify the\b|\bupdate the\b|"
    r"@packages/|<skill |baseUrl\.|model\.id\.includes|XHIGH_MODELS|ENV_AGENT_DIR|"
    r"^const |^// |^\+\d+|^f afterEach|^skills\?:|^declare module"
    r")",
    re.I,
)


@dataclass
class UserMessageRecord:
    model: str
    path: str
    session_name: str
    message_index: int
    text: str
    label: str
    progress_pct: float
    progress_bin: int
    tool_steps_before: int
    total_tool_steps: int


@dataclass
class SessionUserSummary:
    model: str
    path: str
    session_name: str
    total_tool_steps: int
    user_messages: list[UserMessageRecord]


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def classify_user_message(text: str, message_index: int) -> str:
    s = _norm(text)
    low = s.lower()

    if message_index == 1 and _TASK_BRIEF_RE.search(low):
        return "task_brief"
    if _EVIDENCE_RE.search(s):
        return "evidence_or_repro"
    if _VALIDATION_RE.search(s) and not _WORKFLOW_RE.search(s):
        return "validation_request"
    if _WORKFLOW_RE.search(s):
        return "workflow_closeout"
    if _QA_RE.search(s):
        return "qa_or_critique"
    if _AUTHORIZE_RE.search(s):
        # Long messages with explicit implementation plus a detailed preferred
        # design tend to function more as steering than as simple approval.
        if (_STEER_RE.search(s) or len(s) > 220) and any(
            token in low for token in ("should", "instead", "@packages/", "model selector", "wraps the", "use ")
        ):
            return "solution_steer"
        return "authorize_work"
    if _STEER_RE.search(s):
        return "solution_steer"
    if message_index == 1:
        return "task_brief"
    return "solution_steer"


def _extract_session(path: str | Path, model: str | None = None, session_name: str | None = None) -> SessionUserSummary:
    user_events: list[tuple[int, str]] = []
    step_count = 0
    total_steps = 0
    final_name = session_name or ""

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            typ = obj.get("type")
            if typ == "session_info":
                name = obj.get("name", "") or ""
                if name:
                    final_name = name
            if typ != "message":
                continue

            msg = obj.get("message", {})
            role = msg.get("role")
            if role == "user":
                text = _norm(" ".join(block.get("text", "") for block in msg.get("content", []) if block.get("type") == "text"))
                user_events.append((step_count, text))
            elif role == "assistant":
                for block in msg.get("content", []) or []:
                    if block.get("type") == "toolCall":
                        step_count += 1
                        total_steps += 1

    records: list[UserMessageRecord] = []
    inferred_model = model or "unknown"
    total = total_steps
    for idx, (steps_before, text) in enumerate(user_events, start=1):
        progress = (steps_before / total * 100.0) if total else 0.0
        progress = max(0.0, min(100.0, progress))
        bin_idx = min(19, max(0, int(progress // 5)))
        label = classify_user_message(text, idx)
        records.append(
            UserMessageRecord(
                model=inferred_model,
                path=str(path),
                session_name=final_name,
                message_index=idx,
                text=text,
                label=label,
                progress_pct=round(progress, 1),
                progress_bin=bin_idx,
                tool_steps_before=steps_before,
                total_tool_steps=total,
            )
        )

    return SessionUserSummary(
        model=inferred_model,
        path=str(path),
        session_name=final_name,
        total_tool_steps=total,
        user_messages=records,
    )


def analyze_user_messages(allowed_paths: dict[str, set[str]]) -> dict:
    sessions: list[SessionUserSummary] = []
    messages: list[UserMessageRecord] = []

    for model, paths in allowed_paths.items():
        for path in sorted(paths):
            summary = _extract_session(path, model=model)
            sessions.append(summary)
            messages.extend(summary.user_messages)

    per_model = {
        model: {
            "num_sessions": len(paths),
            "num_messages": 0,
            "classes": {},
        }
        for model, paths in allowed_paths.items()
    }

    overall_classes = {}
    messages_by_class = {label: [] for label in CLASS_ORDER}
    for rec in messages:
        messages_by_class[rec.label].append(rec)
        per_model[rec.model]["num_messages"] += 1

    def _class_stats(label: str, recs: list[UserMessageRecord], session_subset: list[SessionUserSummary]) -> dict:
        by_path: dict[str, list[UserMessageRecord]] = defaultdict(list)
        for rec in recs:
            by_path[rec.path].append(rec)
        num_sessions = len(session_subset)
        session_count = len(by_path)
        bin_presence = [0] * 20
        first_vals: list[float] = []
        first_turns: list[int] = []
        for sess in session_subset:
            sess_recs = by_path.get(sess.path, [])
            if not sess_recs:
                continue
            first = min(sess_recs, key=lambda r: r.message_index)
            first_vals.append(first.progress_pct)
            first_turns.append(first.message_index)
            seen_bins = {rec.progress_bin for rec in sess_recs}
            for b in seen_bins:
                bin_presence[b] += 1
        bin_pct = [round(v / num_sessions * 100, 1) if num_sessions else 0.0 for v in bin_presence]
        sample = [
            {
                "model": rec.model,
                "path": rec.path,
                "session_name": rec.session_name,
                "message_index": rec.message_index,
                "progress_pct": rec.progress_pct,
                "text": rec.text,
            }
            for rec in sorted(recs, key=lambda r: (r.model, r.path, r.message_index))
        ]
        return {
            "message_count": len(recs),
            "session_count": session_count,
            "session_pct": round(session_count / num_sessions * 100, 1) if num_sessions else 0.0,
            "message_pct": 0.0,  # filled by caller
            "first_progress_median": round(statistics.median(first_vals), 1) if first_vals else None,
            "first_progress_p25": round(statistics.quantiles(first_vals, n=4)[0], 1) if len(first_vals) >= 2 else (round(first_vals[0], 1) if first_vals else None),
            "first_progress_p75": round(statistics.quantiles(first_vals, n=4)[2], 1) if len(first_vals) >= 2 else (round(first_vals[0], 1) if first_vals else None),
            "first_turn_median": round(statistics.median(first_turns), 1) if first_turns else None,
            "bin_session_pct": bin_pct,
            "messages": sample,
        }

    total_messages = len(messages)
    for label in CLASS_ORDER:
        stats = _class_stats(label, messages_by_class[label], sessions)
        stats["message_pct"] = round(stats["message_count"] / total_messages * 100, 1) if total_messages else 0.0
        overall_classes[label] = stats

    for model in per_model:
        model_sessions = [s for s in sessions if s.model == model]
        model_messages = [m for m in messages if m.model == model]
        total = len(model_messages)
        by_label = defaultdict(list)
        for rec in model_messages:
            by_label[rec.label].append(rec)
        for label in CLASS_ORDER:
            stats = _class_stats(label, by_label[label], model_sessions)
            stats["message_pct"] = round(stats["message_count"] / total * 100, 1) if total else 0.0
            per_model[model]["classes"][label] = stats

    return {
        "class_order": CLASS_ORDER,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "total_sessions": len(sessions),
        "total_messages": total_messages,
        "overall": overall_classes,
        "per_model": per_model,
    }
