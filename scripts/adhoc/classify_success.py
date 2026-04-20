"""
Ad-hoc classifier: label each of the 163 single-model issue sessions as
success / failure based on closeout signals in the transcript.

Not wired into the main analysis. Dumps a JSON + prints a summary so we can
inspect the signals and decide whether the heuristic is trustworthy.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis_pi.session_filter import (
    DEFAULT_EXACT_MODELS,
    SessionFilter,
    collect_filtered_paths,
)


CRITIQUE_RE = re.compile(
    r"\b(brittle|terrible|shit|fuck|what were your instructions|you missed|"
    r"you still don't understand|dude,? what|don't care|no,? we need|"
    r"why (this|would|did)|did you actually|we established that|"
    r"are you sure this doesn't)\b",
    re.I,
)

WORKFLOW_CLOSEOUT_RE = re.compile(
    r"\b(commit|push|close (the )?issue|merge|ship|wrap it|"
    r"add a? changelog|leave a comment|thanks,? (fixed|implemented)|"
    r"looking good|lgtm|looks good|seems to work|jupp|works fine)\b",
    re.I,
)

AGENT_DONE_RE = re.compile(
    r"^(done\.?|wrapped\.?|shipped\.?|complete\.?|✅|all done|finished)",
    re.I | re.M,
)


@dataclass
class SessionSignals:
    path: str
    model: str
    session_name: str = ""
    n_user_messages: int = 0
    first_user: str = ""
    last_user: str = ""
    final_assistant_text: str = ""

    any_source_edit: bool = False
    any_git_commit: bool = False
    any_git_commit_success: bool = False
    any_git_push: bool = False
    any_git_push_success: bool = False
    any_gh_issue_close: bool = False
    any_gh_pr_merge: bool = False
    any_gh_issue_comment_fixed: bool = False
    last_user_has_closeout: bool = False
    last_user_is_critique: bool = False
    agent_says_done: bool = False

    # Derived
    label: str = ""  # success / failure / ambiguous
    reasons: list[str] = field(default_factory=list)


def _flatten_tool_result(msg: dict) -> str:
    parts = []
    for block in msg.get("content", []) or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def analyze(path: str, model: str) -> SessionSignals:
    sig = SessionSignals(path=path, model=model)
    events = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]

    # Tool results lookup
    tool_results: dict[str, dict] = {}
    for e in events:
        if e.get("type") != "message":
            continue
        msg = e["message"]
        if msg.get("role") == "toolResult":
            cid = msg.get("toolCallId")
            if cid:
                tool_results[cid] = msg

    user_msgs: list[str] = []
    final_assistant_text = ""

    for e in events:
        if e.get("type") == "session_info" and e.get("name"):
            sig.session_name = e["name"]
        if e.get("type") != "message":
            continue
        msg = e["message"]
        role = msg.get("role")
        content = msg.get("content", []) or []
        if role == "user":
            txt = " ".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            ).strip()
            user_msgs.append(txt)
        elif role == "assistant":
            text_parts = []
            for b in content:
                if b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif b.get("type") == "toolCall":
                    name = b.get("name") or ""
                    args = b.get("arguments", {}) or {}
                    tr = tool_results.get(b.get("id", ""), {})
                    is_error = bool(tr.get("isError"))
                    obs = _flatten_tool_result(tr)

                    if name in ("edit", "write"):
                        p = (args.get("path") or "").lower()
                        # Heuristic: anything that looks like a source file
                        if p and not any(
                            k in p for k in ("test", "repro", ".md", "changelog")
                        ):
                            sig.any_source_edit = True

                    if name == "bash":
                        cmd = args.get("command", "") or ""
                        head = cmd.strip().splitlines()[0].lower() if cmd.strip() else ""

                        # git commit
                        if re.search(r"\bgit (?:-c [^\s]+\s+)?commit\b", cmd):
                            sig.any_git_commit = True
                            if not is_error:
                                sig.any_git_commit_success = True

                        # git push
                        if re.search(r"\bgit (?:-c [^\s]+\s+)?push\b", cmd):
                            sig.any_git_push = True
                            # Determine success — git push prints "To <url>" on success
                            # and "! [rejected]" / "error:" on failure
                            if not is_error:
                                low = obs.lower()
                                if ("rejected" not in low) and ("error:" not in low):
                                    sig.any_git_push_success = True

                        if "gh issue close" in cmd:
                            sig.any_gh_issue_close = True
                        if "gh pr merge" in cmd:
                            sig.any_gh_pr_merge = True
                        if "gh issue comment" in cmd or "gh pr comment" in cmd:
                            low = cmd.lower()
                            if (
                                "fixed" in low
                                or "implemented" in low
                                or "shipped" in low
                                or "thanks" in low
                            ):
                                sig.any_gh_issue_comment_fixed = True

            if text_parts:
                final_assistant_text = "\n".join(text_parts)

    sig.n_user_messages = len(user_msgs)
    sig.first_user = user_msgs[0] if user_msgs else ""
    sig.last_user = user_msgs[-1] if user_msgs else ""
    sig.final_assistant_text = final_assistant_text

    # Only call "last user closeout" if there's more than the task brief
    if len(user_msgs) >= 2:
        sig.last_user_has_closeout = bool(WORKFLOW_CLOSEOUT_RE.search(sig.last_user))
        sig.last_user_is_critique = bool(CRITIQUE_RE.search(sig.last_user))
    sig.agent_says_done = bool(AGENT_DONE_RE.search(final_assistant_text))

    # Decide label
    label, reasons = decide(sig)
    sig.label = label
    sig.reasons = reasons
    return sig


def decide(sig: SessionSignals) -> tuple[str, list[str]]:
    reasons: list[str] = []

    # Task-brief-only runs almost always mean the user aborted.
    if sig.n_user_messages <= 1:
        reasons.append("task-brief-only (no follow-up from user)")
        return "failure", reasons

    # No implementation at all
    if not sig.any_source_edit and not sig.any_git_commit:
        reasons.append("no source edit and no git commit")
        return "failure", reasons

    # Actual shipping signal
    shipped = (
        sig.any_git_push_success
        or sig.any_gh_issue_close
        or sig.any_gh_pr_merge
    )

    if shipped:
        if sig.last_user_is_critique:
            reasons.append("shipped but last user message is critique — ambiguous")
            return "ambiguous", reasons
        if sig.last_user_has_closeout or sig.agent_says_done or sig.any_gh_issue_comment_fixed:
            reasons.append("shipped + positive closeout")
            return "success", reasons
        reasons.append("shipped, no explicit closeout line")
        return "success", reasons

    # Not shipped
    if sig.any_git_commit_success and sig.last_user_has_closeout:
        # User said "commit/push" but push didn't run or errored
        reasons.append("committed but not pushed; user asked to ship")
        return "ambiguous", reasons

    reasons.append("never pushed")
    return "failure", reasons


def main() -> None:
    sf = SessionFilter(
        allowed_models=list(DEFAULT_EXACT_MODELS),
        require_single_model=True,
        session_name_prefixes=["Issue:"],
    )
    selected, counts, _ = collect_filtered_paths(Path("data/pi-mono"), sf)
    all_sigs: list[SessionSignals] = []
    for model, paths in selected.items():
        for path in sorted(paths):
            sig = analyze(path, model)
            all_sigs.append(sig)

    # Print per-model table
    from collections import defaultdict

    per_model = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0, "ambiguous": 0})
    for s in all_sigs:
        per_model[s.model]["total"] += 1
        per_model[s.model][s.label] += 1

    print(f"\n{'model':<24} {'total':>6} {'succ':>6} {'fail':>6} {'amb':>6} {'succ%':>7}")
    for m in DEFAULT_EXACT_MODELS:
        if m not in per_model:
            continue
        d = per_model[m]
        rate = d["success"] / d["total"] * 100 if d["total"] else 0
        print(f"{m:<24} {d['total']:>6} {d['success']:>6} {d['failure']:>6} {d['ambiguous']:>6} {rate:>6.1f}%")

    total = sum(d["total"] for d in per_model.values())
    succ = sum(d["success"] for d in per_model.values())
    fail = sum(d["failure"] for d in per_model.values())
    amb = sum(d["ambiguous"] for d in per_model.values())
    print(f"{'TOTAL':<24} {total:>6} {succ:>6} {fail:>6} {amb:>6} {succ / total * 100:>6.1f}%")

    # Write JSON for inspection
    out_path = Path("scripts/adhoc/classify_success_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for s in all_sigs:
        d = s.__dict__.copy()
        # Truncate text fields for readability
        d["first_user"] = s.first_user[:400]
        d["last_user"] = s.last_user[:400]
        d["final_assistant_text"] = s.final_assistant_text[:800]
        data.append(d)
    out_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
