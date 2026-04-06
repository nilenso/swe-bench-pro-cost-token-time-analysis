"""
Extract per-trajectory stats from SWE-Bench Pro trajectory files.

Usage:
    python extract_stats.py sample/gpt5 sample/claude45
    python extract_stats.py --eval-results eval_results_gpt5.json eval_results_claude45.json sample/gpt5 sample/claude45

Each positional arg is a directory containing instance subdirectories, each with a .traj file.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import tiktoken


@dataclass
class TokenStats:
    """Output token counts computed via tiktoken."""
    content: int = 0       # message.content tokens (what tokens_received reports)
    thought: int = 0       # thought tokens (Claude only, often == content)
    tool_call_args: int = 0  # tool_calls[].function.arguments tokens (missed by tokens_received)
    total: int = 0         # content + tool_call_args (thought excluded when == content)


@dataclass
class ModelStats:
    """Raw stats from info.model_stats in the trajectory file."""
    instance_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0  # undercounts — see TokenStats for corrected version
    api_calls: int = 0


@dataclass
class ToolTimeStats:
    """Tool/command execution time stats from trajectory[].execution_time."""
    total_seconds: float = 0.0
    mean_seconds: float = 0.0
    median_seconds: float = 0.0
    max_seconds: float = 0.0
    steps_over_10s: int = 0
    steps_over_60s: int = 0


@dataclass
class ActionBreakdown:
    """Counts of action types."""
    bash: int = 0
    view: int = 0
    edit: int = 0
    create: int = 0
    search_find: int = 0
    submit: int = 0
    other: int = 0


@dataclass
class ContentVolume:
    """Character counts of content flowing through the trajectory."""
    observation_chars: int = 0   # tool output the agent consumed
    action_chars: int = 0        # agent's action strings
    thought_chars: int = 0       # agent's reasoning text
    response_chars: int = 0      # agent's response text


@dataclass
class Config:
    """Model configuration from replay_config."""
    model_name: str = ""
    temperature: float = 0.0
    per_instance_call_limit: int = 0
    total_cost_limit: float = 0.0
    completion_kwargs: dict = field(default_factory=dict)


@dataclass
class TrajectoryStats:
    """All stats for a single trajectory."""
    instance_id: str = ""
    repo: str = ""
    model_dir: str = ""        # which directory it came from

    # outcome
    exit_status: str = ""
    submitted: bool = False
    patch_chars: int = 0
    resolved: bool | None = None  # None if eval results not provided

    # counts
    steps: int = 0
    history_entries: int = 0

    # from info.model_stats
    model_stats: ModelStats = field(default_factory=ModelStats)

    # derived
    tokens_per_call: float = 0.0  # tokens_sent / api_calls

    # tiktoken-counted output tokens
    output_tokens: TokenStats = field(default_factory=TokenStats)

    # tool execution time
    tool_time: ToolTimeStats = field(default_factory=ToolTimeStats)

    # action breakdown
    actions: ActionBreakdown = field(default_factory=ActionBreakdown)

    # content volume
    content: ContentVolume = field(default_factory=ContentVolume)

    # config
    config: Config = field(default_factory=Config)


def classify_action(action: str) -> str:
    """Classify an action string into a category."""
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

    # Shell commands — either wrapped in "bash -lc ..." or bare
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
    """Extract model configuration from replay_config."""
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


def extract_stats(traj_path: str, enc: tiktoken.Encoding) -> TrajectoryStats:
    """Extract all stats from a single trajectory file."""
    with open(traj_path) as f:
        traj = json.load(f)

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
    content = ContentVolume()
    for s in trajectory:
        content.observation_chars += len(s.get('observation', '') or '')
        content.action_chars += len(s.get('action', '') or '')
        content.thought_chars += len(s.get('thought', '') or '')
        content.response_chars += len(s.get('response', '') or '')

    # Outcome
    submission = info.get('submission', '') or ''

    # Config
    config = extract_config(traj)

    # Derived
    api_calls = model_stats.api_calls or 1
    tokens_per_call = model_stats.tokens_sent / api_calls

    return TrajectoryStats(
        instance_id=instance_id,
        repo=repo,
        model_dir=str(Path(traj_path).parent.parent.name),
        exit_status=info.get('exit_status', ''),
        submitted=bool(submission),
        patch_chars=len(submission),
        steps=len(trajectory),
        history_entries=len(history),
        model_stats=model_stats,
        tokens_per_call=tokens_per_call,
        output_tokens=output_tokens,
        tool_time=tool_time,
        actions=actions,
        content=content,
        config=config,
    )


def find_trajectories(directory: str) -> list[str]:
    """Find all .traj files in instance subdirectories."""
    traj_files = []
    dirpath = Path(directory)
    for inst_dir in sorted(dirpath.iterdir()):
        if not inst_dir.is_dir():
            continue
        for f in inst_dir.iterdir():
            if f.suffix == '.traj':
                traj_files.append(str(f))
    return traj_files


def load_eval_results(paths: list[str]) -> dict[str, bool]:
    """Load eval results from one or more JSON files. Returns instance_id -> resolved."""
    results = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        results.update(data)
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract stats from SWE-Bench Pro trajectories")
    parser.add_argument("directories", nargs="+", help="Directories containing instance subdirs with .traj files")
    parser.add_argument("--eval-results", nargs="*", default=[], help="eval_results.json files for resolved status")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    # Load eval results if provided
    eval_results = {}
    if args.eval_results:
        eval_results = load_eval_results(args.eval_results)

    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")

    # Find and process all trajectories
    all_stats = []
    t0 = time.time()

    for directory in args.directories:
        traj_files = find_trajectories(directory)
        print(f"Found {len(traj_files)} trajectories in {directory}", file=sys.stderr)

        for traj_path in traj_files:
            stats = extract_stats(traj_path, enc)

            # Cross-reference eval results
            if eval_results:
                stats.resolved = eval_results.get(stats.instance_id, None)

            all_stats.append(stats)

    elapsed = time.time() - t0
    print(f"Processed {len(all_stats)} trajectories in {elapsed:.1f}s ({elapsed/len(all_stats)*1000:.0f}ms each)", file=sys.stderr)

    # Output
    output = [asdict(s) for s in all_stats]
    json_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
