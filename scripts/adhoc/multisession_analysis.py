"""
Ad-hoc: analyze cross-session linking by GitHub issue/PR number.
Tests 3 unit-of-analysis schemes against the 163 single-model Issue: sessions.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_pi.models import normalize_model_name
from classify_success import analyze as _analyze

URL_RE = re.compile(r"https://github\.com/[^\s\)\]]+/(issues|pull|pulls)/(\d+)")
ALLOWED = ["gpt-5.4", "claude-opus-4-5", "gpt-5.2-codex", "gpt-5.3-codex", "claude-opus-4-6"]
ALLOWED_SET = set(ALLOWED)


def scan(path: str) -> dict:
    name = ""
    first_user = ""
    n_user = 0
    models: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type")
            if t == "session_info" and obj.get("name"):
                name = obj["name"]
            elif t == "model_change":
                m = obj.get("modelId")
                if m:
                    n = normalize_model_name(m)
                    if n and n not in models:
                        models.append(n)
            elif t == "message":
                msg = obj.get("message", {})
                if msg.get("role") == "user":
                    n_user += 1
                    if n_user == 1:
                        first_user = " ".join(
                            b.get("text", "")
                            for b in msg.get("content", []) or []
                            if b.get("type") == "text"
                        )[:500]
                elif msg.get("role") == "assistant":
                    m = msg.get("model")
                    if m:
                        n = normalize_model_name(m)
                        if n and n not in models:
                            models.append(n)
    keys: set[str] = set()
    m = URL_RE.search(name)
    if m:
        keys.add(m.group(2))
    for m in URL_RE.finditer(first_user):
        keys.add(m.group(2))
    prefix = (
        "issue" if name.lower().startswith("issue:")
        else "pr" if name.lower().startswith("pr:")
        else "other"
    )
    return {
        "path": path,
        "name": name,
        "first_user": first_user,
        "target_keys": keys,
        "models": models,
        "single_model": models[0] if len(models) == 1 else None,
        "prefix": prefix,
    }


def classify_one(path_model: tuple[str, str | None]) -> tuple[str, dict]:
    path, model = path_model
    try:
        sig = _analyze(path, model or "unknown")
        return path, {
            "shipped": bool(sig.any_git_push_success or sig.any_gh_issue_close or sig.any_gh_pr_merge),
            "label": sig.label,
            "n_user_messages": sig.n_user_messages,
            "any_source_edit": sig.any_source_edit,
            "any_git_commit_success": sig.any_git_commit_success,
        }
    except Exception as e:
        return path, {"shipped": False, "label": "error", "error": str(e), "n_user_messages": 0}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    paths = sorted(str(p) for p in Path("data/pi-mono").glob("*.jsonl"))
    print(f"Scanning {len(paths)} sessions...", file=sys.stderr)

    rows = [scan(p) for p in paths]

    print("Classifying...", file=sys.stderr)
    items = [(r["path"], r["single_model"]) for r in rows]
    with ProcessPoolExecutor(max_workers=8) as ex:
        cls_map = dict(ex.map(classify_one, items, chunksize=16))
    for r in rows:
        r.update(cls_map.get(r["path"], {}))

    # Index by key
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for k in r["target_keys"]:
            by_key[k].append(r)

    issue_163 = [r for r in rows if r["prefix"] == "issue" and r["single_model"] in ALLOWED_SET]
    print(f"\n163 set = {len(issue_163)}")

    # --- Scheme 1: per-session current ---
    print("\n=== Scheme 1: per-session (current) ===")
    s1 = defaultdict(lambda: {"t": 0, "s": 0})
    for r in issue_163:
        m = r["single_model"]
        s1[m]["t"] += 1
        if r["shipped"]:
            s1[m]["s"] += 1
    print(f"{'model':<20} {'N':>4} {'shipped':>8} {'rate':>7}")
    for m in ALLOWED:
        d = s1[m]
        rate = d["s"] / d["t"] * 100 if d["t"] else 0
        print(f"{m:<20} {d['t']:>4d} {d['s']:>8d} {rate:>6.1f}%")

    # --- Scheme 2: per-session + same-model peer rescue ---
    print("\n=== Scheme 2: per-session + same-model peer rescue ===")
    s2 = defaultdict(lambda: {"t": 0, "s": 0})
    rescues = []
    for r in issue_163:
        m = r["single_model"]
        s2[m]["t"] += 1
        is_ship = r["shipped"]
        rescued_by = None
        if not is_ship:
            for k in r["target_keys"]:
                for p in by_key.get(k, []):
                    if p["path"] == r["path"]:
                        continue
                    if p["single_model"] != m:
                        continue
                    if p["shipped"]:
                        is_ship = True
                        rescued_by = p
                        break
                if is_ship:
                    break
        if is_ship:
            s2[m]["s"] += 1
        if rescued_by:
            rescues.append((r, rescued_by))
    print(f"{'model':<20} {'N':>4} {'shipped':>8} {'rate':>7}")
    for m in ALLOWED:
        d = s2[m]
        rate = d["s"] / d["t"] * 100 if d["t"] else 0
        print(f"{m:<20} {d['t']:>4d} {d['s']:>8d} {rate:>6.1f}%")

    # --- Scheme 3: per-(model, issue) dedup ---
    print("\n=== Scheme 3: per-(model, issue) — dedup same-model sessions on same issue ===")
    attempts: dict[str, set[str]] = defaultdict(set)
    shipped_issues: dict[str, set[str]] = defaultdict(set)
    for r in issue_163:
        m = r["single_model"]
        for k in r["target_keys"]:
            attempts[m].add(k)
            for p in by_key[k]:
                if p["single_model"] == m and p["shipped"]:
                    shipped_issues[m].add(k)
                    break
    print(f"{'model':<20} {'N-issues':>9} {'shipped':>8} {'rate':>7}")
    for m in ALLOWED:
        t = len(attempts[m])
        s = len(shipped_issues[m])
        rate = s / t * 100 if t else 0
        print(f"{m:<20} {t:>9d} {s:>8d} {rate:>6.1f}%")

    # --- Rescue events detail ---
    print(f"\n=== Rescue events: {len(rescues)} ===")
    for aborted, saver in rescues:
        print(
            f"  {aborted['single_model']:<18} n_user={aborted['n_user_messages']:>2} "
            f"→ peer n_user={saver['n_user_messages']:>2} prefix={saver['prefix']:6} "
            f"| {aborted['name'][:55]}"
        )

    # --- Peer landscape table: for each 163 session, what peers exist? ---
    print("\n=== Peer landscape (163 sessions with peers) ===")
    same_m = Counter()
    diff_m = Counter()
    pr_peer = Counter()
    other_peer = Counter()
    no_peer = Counter()
    for r in issue_163:
        m = r["single_model"]
        peers = []
        for k in r["target_keys"]:
            for p in by_key.get(k, []):
                if p["path"] != r["path"] and p not in peers:
                    peers.append(p)
        if not peers:
            no_peer[m] += 1
            continue
        for p in peers:
            if p["prefix"] == "pr":
                pr_peer[m] += 1
            elif p["prefix"] == "other":
                other_peer[m] += 1
            if p["single_model"] == m:
                same_m[m] += 1
            elif p["single_model"] and p["single_model"] != m:
                diff_m[m] += 1

    print(f"{'model':<20} {'no-peer':>8} {'same-m-peer':>13} {'diff-m-peer':>13} {'PR-peer':>9} {'other-peer':>12}")
    for m in ALLOWED:
        print(
            f"{m:<20} {no_peer[m]:>8d} {same_m[m]:>13d} {diff_m[m]:>13d} "
            f"{pr_peer[m]:>9d} {other_peer[m]:>12d}"
        )

    # --- What do 'other' (non-Issue/non-PR) peer sessions look like? ---
    print("\n=== Sample 'other'-prefix peers (these are the interesting 'continuation' candidates) ===")
    seen = set()
    shown = 0
    for r in issue_163:
        for k in r["target_keys"]:
            for p in by_key.get(k, []):
                if p["prefix"] != "other":
                    continue
                if p["path"] in seen:
                    continue
                seen.add(p["path"])
                pm = ",".join(p["models"]) if p["models"] else "(none)"
                print(f"  key=#{k}  models={pm:30s}  n_user={p['n_user_messages']:>2}  shipped={p['shipped']}")
                print(f"     first_user: {p['first_user'][:120]}")
                shown += 1
                if shown >= 10:
                    return


if __name__ == "__main__":
    main()
