"""
Fast parallel extraction of per-trajectory stats from SWE-Bench Pro trajectory files.

Uses orjson for ~4x faster JSON parsing and multiprocessing for CPU parallelism.
The GIL blocks stdlib json in threads, so processes are necessary.

Usage:
    python extract_stats_fast.py data/gpt5/traj data/claude45/traj \
        --eval-results eval_results_gpt5.json eval_results_claude45.json \
        -o full_stats.json

    # Control parallelism:
    python extract_stats_fast.py data/gpt5/traj -w 4 -o stats.json
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import orjson
import tiktoken


# ── Dataclasses (identical to extract_stats.py) ─────────────────────────

@dataclass
class TokenStats:
    content: int = 0
    thought: int = 0
    tool_call_args: int = 0
    total: int = 0


@dataclass
class ModelStats:
    instance_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0


@dataclass
class ToolTimeStats:
    total_seconds: float = 0.0
    mean_seconds: float = 0.0
    median_seconds: float = 0.0
    max_seconds: float = 0.0
    steps_over_10s: int = 0
    steps_over_60s: int = 0


@dataclass
class ActionBreakdown:
    bash: int = 0
    view: int = 0
    edit: int = 0
    create: int = 0
    search_find: int = 0
    submit: int = 0
    other: int = 0


@dataclass
class ContentVolume:
    observation_chars: int = 0
    action_chars: int = 0
    thought_chars: int = 0
    response_chars: int = 0


@dataclass
class Config:
    model_name: str = ""
    temperature: float = 0.0
    per_instance_call_limit: int = 0
    total_cost_limit: float = 0.0
    completion_kwargs: dict = field(default_factory=dict)


@dataclass
class TrajectoryStats:
    instance_id: str = ""
    repo: str = ""
    model_dir: str = ""
    exit_status: str = ""
    submitted: bool = False
    patch_chars: int = 0
    patch_tokens: int = 0
    resolved: bool | None = None
    steps: int = 0
    history_entries: int = 0
    model_stats: ModelStats = field(default_factory=ModelStats)
    tokens_per_call: float = 0.0
    output_tokens: TokenStats = field(default_factory=TokenStats)
    tool_time: ToolTimeStats = field(default_factory=ToolTimeStats)
    actions: ActionBreakdown = field(default_factory=ActionBreakdown)
    content: ContentVolume = field(default_factory=ContentVolume)
    config: Config = field(default_factory=Config)


# ── Pure functions (no global state) ────────────────────────────────────

def classify_action(action: str) -> str:
    if not action or not action.strip():
        return 'other'
    prefix = action[:80].lower()
    if action.startswith('str_replace_editor'):
        if ' view' in prefix:
            return 'view'
        elif ' create' in prefix:
            return 'create'
        elif ' str_replace' in prefix:
            return 'edit'
        elif ' insert' in prefix:
            return 'edit'
        else:
            return 'other'
    elif action.startswith('submit'):
        return 'submit'
    elif action.startswith(('find ', 'find/')):
        return 'search_find'
    elif action.startswith(('grep ', 'grep\t')):
        return 'search_find'

    # bash -lc "grep ..." is also a search
    if action.startswith('bash'):
        inner = action
        for delim in ('"', "'"):
            idx = action.find(delim)
            if idx != -1:
                inner = action[idx+1:].lstrip()
                break
        if inner.startswith(('grep ', 'grep\t', 'find ', 'find/')):
            return 'search_find'

    shell_prefixes = (
        'bash', 'cd ', 'python', 'node ', 'npm ', 'go ', 'pytest', 'cat ',
        'sed ', 'head ', 'tail ', 'ls ', 'wc ', 'nl ', 'rm ',
        'mkdir ', 'cp ', 'mv ', 'chmod ', 'echo ', 'export ', 'source ',
        'pip ', 'apt ', 'make ', 'docker ', 'git ', 'curl ', 'wget ',
        'touch ', 'sort ', 'uniq ', 'awk ', 'cut ', 'diff ', 'patch ',
        'tar ', 'which ', 'env ', 'timeout ', 'applypatch',
        'redis-', 'stat ', 'pkill ', 'pgrep ', 'sleep ', 'kill ',
        'yarn ', 'md5sum ', 'test ', 'pwd', 'whoami', 'locale ',
        'command ', 'ripgrep',
    )
    if action.startswith(shell_prefixes):
        return 'bash'
    return 'other'


def extract_config(traj: dict) -> Config:
    rc = traj.get('replay_config', '{}')
    if isinstance(rc, str):
        try:
            rc = json.loads(rc)
        except json.JSONDecodeError:
            return Config()
    agent = rc.get('agent', {})
    model_cfg = agent.get('model', {})
    return Config(
        model_name=model_cfg.get('name', ''),
        temperature=model_cfg.get('temperature', 0.0),
        per_instance_call_limit=model_cfg.get('per_instance_call_limit', 0),
        total_cost_limit=model_cfg.get('total_cost_limit', 0.0),
        completion_kwargs=model_cfg.get('completion_kwargs', {}),
    )


def extract_one(traj_path: str, enc: tiktoken.Encoding) -> dict:
    """Extract stats from a single .traj file. Returns a plain dict (for pickling)."""
    with open(traj_path, 'rb') as f:
        traj = orjson.loads(f.read())

    info = traj.get('info', {})
    trajectory = traj.get('trajectory', [])
    history = traj.get('history', [])

    # Instance ID
    instance_id = traj.get('environment', '')
    if not instance_id:
        instance_id = Path(traj_path).stem

    # Repo
    repo = ""
    name = instance_id.replace("instance_", "")
    parts = name.split("__")
    if len(parts) >= 2:
        org = parts[0]
        repo_name = parts[1].split("-")[0]
        repo = f"{org}/{repo_name}"

    # Model stats
    ms = info.get('model_stats', {})
    model_stats = ModelStats(
        instance_cost=ms.get('instance_cost', 0.0),
        tokens_sent=ms.get('tokens_sent', 0),
        tokens_received=ms.get('tokens_received', 0),
        api_calls=ms.get('api_calls', 0),
    )

    # Output tokens via tiktoken
    output_tokens = TokenStats()
    for h in history:
        if h.get('role') != 'assistant':
            continue
        content = h.get('content', '') or ''
        thought = h.get('thought', '') or ''
        if content:
            output_tokens.content += len(enc.encode(content))
        if thought and thought != content:
            output_tokens.thought += len(enc.encode(thought))
        for tc in (h.get('tool_calls', None) or []):
            args = tc.get('function', {}).get('arguments', '') or ''
            if args:
                output_tokens.tool_call_args += len(enc.encode(args))
    output_tokens.total = output_tokens.content + output_tokens.thought + output_tokens.tool_call_args

    # Tool execution time
    exec_times = [s.get('execution_time', 0) or 0 for s in trajectory]
    tool_time = ToolTimeStats()
    if exec_times:
        sorted_times = sorted(exec_times)
        tool_time.total_seconds = sum(exec_times)
        tool_time.mean_seconds = tool_time.total_seconds / len(exec_times)
        tool_time.median_seconds = sorted_times[len(sorted_times) // 2]
        tool_time.max_seconds = sorted_times[-1]
        tool_time.steps_over_10s = sum(1 for t in exec_times if t > 10)
        tool_time.steps_over_60s = sum(1 for t in exec_times if t > 60)

    # Action breakdown
    actions = ActionBreakdown()
    for s in trajectory:
        action = s.get('action', '') or ''
        category = classify_action(action)
        setattr(actions, category, getattr(actions, category) + 1)

    # Content volume
    content_vol = ContentVolume()
    for s in trajectory:
        content_vol.observation_chars += len(s.get('observation', '') or '')
        content_vol.action_chars += len(s.get('action', '') or '')
        content_vol.thought_chars += len(s.get('thought', '') or '')
        content_vol.response_chars += len(s.get('response', '') or '')

    # Outcome
    submission = info.get('submission', '') or ''

    # Config
    config = extract_config(traj)

    # Derived
    api_calls = model_stats.api_calls or 1
    tokens_per_call = model_stats.tokens_sent / api_calls

    stats = TrajectoryStats(
        instance_id=instance_id,
        repo=repo,
        model_dir=str(Path(traj_path).parent.parent.name),
        exit_status=info.get('exit_status', ''),
        submitted=bool(submission),
        patch_chars=len(submission),
        patch_tokens=len(enc.encode(submission)) if submission else 0,
        steps=len(trajectory),
        history_entries=len(history),
        model_stats=model_stats,
        tokens_per_call=tokens_per_call,
        output_tokens=output_tokens,
        tool_time=tool_time,
        actions=actions,
        content=content_vol,
        config=config,
    )
    result = asdict(stats)
    result['_traj_path'] = traj_path
    return result


# ── Worker process ──────────────────────────────────────────────────────

# Each worker initializes its own tiktoken encoder (not picklable across processes).
_worker_enc = None

def _worker_init():
    global _worker_enc
    _worker_enc = tiktoken.get_encoding("cl100k_base")


def _worker_extract(traj_path: str) -> dict:
    """Called in worker process. Returns dict or error dict."""
    try:
        return extract_one(traj_path, _worker_enc)
    except Exception as e:
        return {"_error": str(e), "_path": traj_path}


# ── File discovery ──────────────────────────────────────────────────────

def find_trajectories(directory: str) -> list[str]:
    """Find all .traj files in instance subdirectories (one level deep)."""
    traj_files = []
    dirpath = Path(directory)
    if not dirpath.is_dir():
        print(f"WARNING: {directory} is not a directory, skipping", file=sys.stderr)
        return []
    for inst_dir in sorted(dirpath.iterdir()):
        if not inst_dir.is_dir():
            continue
        for f in inst_dir.iterdir():
            if f.suffix == '.traj':
                traj_files.append(str(f))
    return traj_files


def load_eval_results(pairs: list[str]) -> dict[str, dict[str, bool]]:
    """Load eval results keyed by (directory, instance_id).

    Args:
        pairs: list of 'directory:eval_file.json' strings
    Returns:
        dict mapping directory -> {instance_id: resolved}
    """
    results = {}
    for pair in pairs:
        if ':' in pair:
            directory, path = pair.split(':', 1)
        else:
            # Legacy: bare path, no directory scoping
            directory = '__all__'
            path = pair
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
        results[directory] = data
    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast parallel stats extraction from SWE-Bench Pro trajectories")
    parser.add_argument("directories", nargs="+", help="Directories containing instance subdirs with .traj files")
    parser.add_argument("--eval-results", nargs="*", default=[],
                        help="dir:eval_results.json pairs (e.g. data/gpt5/traj:eval_gpt5.json)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Worker processes (default: cpu_count - 2)")
    args = parser.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    # ── Discover files ──────────────────────────────────────────────
    all_traj_files = []
    for directory in args.directories:
        files = find_trajectories(directory)
        print(f"  {directory}: {len(files)} trajectories", file=sys.stderr)
        all_traj_files.extend(files)

    total = len(all_traj_files)
    if total == 0:
        print("No .traj files found.", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(os.path.getsize(f) for f in all_traj_files)
    print(f"  Total: {total} files, {total_bytes / 1e9:.2f} GB", file=sys.stderr)
    print(f"  Workers: {workers}", file=sys.stderr)

    # ── Load eval results ───────────────────────────────────────────
    eval_by_dir = {}
    if args.eval_results:
        eval_by_dir = load_eval_results(args.eval_results)
        total_entries = sum(len(v) for v in eval_by_dir.values())
        print(f"  Eval results: {total_entries} entries across {len(eval_by_dir)} files", file=sys.stderr)

    # ── Parallel extraction ─────────────────────────────────────────
    t0 = time.time()
    done = 0
    errors = 0
    all_stats = []

    print(f"\n  [{'·' * 50}] 0/{total}", end='', file=sys.stderr)

    with multiprocessing.Pool(workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_worker_extract, all_traj_files, chunksize=4):
            done += 1
            if '_error' in result:
                errors += 1
                print(f"\n  ERROR: {result['_path']}: {result['_error']}", file=sys.stderr)
            else:
                # Cross-reference eval results — match by traj directory
                if eval_by_dir:
                    resolved = None
                    traj_path = result.get('_traj_path', '')
                    for directory, evals in eval_by_dir.items():
                        if directory == '__all__' or directory in traj_path:
                            resolved = evals.get(result['instance_id'], None)
                            if resolved is not None:
                                break
                    result['resolved'] = resolved
                result.pop('_traj_path', None)
                all_stats.append(result)

            # Progress bar
            filled = int(50 * done / total)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"\r  [{'█' * filled}{'·' * (50 - filled)}] {done}/{total}  "
                  f"{elapsed:.1f}s  {rate:.0f}/s  ETA {eta:.0f}s   ",
                  end='', file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n\n  Done: {len(all_stats)} trajectories in {elapsed:.1f}s "
          f"({elapsed/max(len(all_stats),1)*1000:.0f}ms each, {errors} errors)",
          file=sys.stderr)

    # ── Sort for stable output ──────────────────────────────────────
    all_stats.sort(key=lambda s: (s['model_dir'], s['instance_id']))

    # ── Write output ────────────────────────────────────────────────
    json_bytes = orjson.dumps(all_stats, option=orjson.OPT_INDENT_2)

    if args.output:
        with open(args.output, 'wb') as f:
            f.write(json_bytes)
        print(f"  Wrote {args.output} ({len(json_bytes) / 1e6:.1f} MB)", file=sys.stderr)
    else:
        sys.stdout.buffer.write(json_bytes)
        sys.stdout.buffer.write(b'\n')


if __name__ == "__main__":
    main()
