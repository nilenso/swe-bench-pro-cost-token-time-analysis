"""
Generate a tabular summary from extracted trajectory stats JSON.

Usage:
    python summarize_stats.py partial_stats.json
    python summarize_stats.py partial_stats.json -o partial_summary.txt
"""

import argparse
import json
import sys
from statistics import mean, median


def split_by_model(data):
    """Split stats into GPT-5 and Claude groups by config.model_name."""
    gpt, claude = [], []
    for s in data:
        name = s.get('config', {}).get('model_name', '')
        if 'gpt' in name.lower():
            gpt.append(s)
        elif 'claude' in name.lower():
            claude.append(s)
    return gpt, claude


def safe_mean(vals):
    return mean(vals) if vals else 0


def safe_median(vals):
    return median(vals) if vals else 0


def fmt_int(n):
    return f"{n:>,d}" if isinstance(n, int) else f"{n:>,.0f}"


def summarize(data):
    gpt, claude = split_by_model(data)
    n_gpt, n_claude = len(gpt), len(claude)

    if n_gpt == 0 and n_claude == 0:
        return "No data found.\n"

    # Repo breakdown
    gpt_repos = sorted(set(s['repo'] for s in gpt))
    claude_repos = sorted(set(s['repo'] for s in claude))
    all_repos = sorted(set(gpt_repos + claude_repos))

    # Outcomes
    gpt_resolved = sum(1 for s in gpt if s.get('resolved') is True)
    gpt_submitted = sum(1 for s in gpt if s.get('submitted'))
    claude_resolved = sum(1 for s in claude if s.get('resolved') is True)
    claude_submitted = sum(1 for s in claude if s.get('submitted'))

    # Cost
    gpt_costs = [s['model_stats']['instance_cost'] for s in gpt]
    claude_costs = [s['model_stats']['instance_cost'] for s in claude]

    # Tokens
    gpt_input = [s['model_stats']['tokens_sent'] for s in gpt]
    claude_input = [s['model_stats']['tokens_sent'] for s in claude]
    gpt_calls = [s['model_stats']['api_calls'] for s in gpt]
    claude_calls = [s['model_stats']['api_calls'] for s in claude]
    gpt_tpc = [s['tokens_per_call'] for s in gpt]
    claude_tpc = [s['tokens_per_call'] for s in claude]
    gpt_out = [s['output_tokens']['total'] for s in gpt]
    claude_out = [s['output_tokens']['total'] for s in claude]
    gpt_out_content = [s['output_tokens']['content'] for s in gpt]
    claude_out_content = [s['output_tokens']['content'] for s in claude]
    gpt_out_tc = [s['output_tokens']['tool_call_args'] for s in gpt]
    claude_out_tc = [s['output_tokens']['tool_call_args'] for s in claude]
    gpt_recv = [s['model_stats']['tokens_received'] for s in gpt]
    claude_recv = [s['model_stats']['tokens_received'] for s in claude]

    # Tool time
    gpt_tt = [s['tool_time']['total_seconds'] for s in gpt]
    claude_tt = [s['tool_time']['total_seconds'] for s in claude]
    gpt_max_step = [s['tool_time']['max_seconds'] for s in gpt]
    claude_max_step = [s['tool_time']['max_seconds'] for s in claude]
    gpt_over10 = sum(s['tool_time']['steps_over_10s'] for s in gpt)
    claude_over10 = sum(s['tool_time']['steps_over_10s'] for s in claude)

    # Steps & actions
    gpt_steps = [s['steps'] for s in gpt]
    claude_steps = [s['steps'] for s in claude]

    action_keys = ['bash', 'view', 'edit', 'create', 'search_find', 'submit', 'other']

    # Content volume
    gpt_obs = [s['content']['observation_chars'] for s in gpt]
    claude_obs = [s['content']['observation_chars'] for s in claude]
    gpt_act = [s['content']['action_chars'] for s in gpt]
    claude_act = [s['content']['action_chars'] for s in claude]
    gpt_tht = [s['content']['thought_chars'] for s in gpt]
    claude_tht = [s['content']['thought_chars'] for s in claude]
    gpt_patch = [s['patch_chars'] for s in gpt]
    claude_patch = [s['patch_chars'] for s in claude]

    # $/1M input
    gpt_per_1m = (sum(gpt_costs) / sum(gpt_input) * 1e6) if sum(gpt_input) > 0 else 0
    claude_per_1m = (sum(claude_costs) / sum(claude_input) * 1e6) if sum(claude_input) > 0 else 0

    # Cost per resolve
    gpt_cpr = sum(gpt_costs) / gpt_resolved if gpt_resolved > 0 else float('inf')
    claude_cpr = sum(claude_costs) / claude_resolved if claude_resolved > 0 else float('inf')

    W = 75
    lines = []
    def line(s=""):
        lines.append(s)
    def sep():
        lines.append("-" * W)
    def header(s):
        lines.append("")
        lines.append(s)
        sep()

    line(f"SWE-Bench Pro Trajectory Analysis — {n_gpt} + {n_claude} instances ({n_gpt + n_claude} total)")
    line("=" * W)

    # Repos covered
    header("REPOS")
    for repo in all_repos:
        g = sum(1 for s in gpt if s['repo'] == repo)
        c = sum(1 for s in claude if s['repo'] == repo)
        line(f"  {repo:<40s}  GPT-5: {g:>3d}   Claude: {c:>3d}")

    header("OUTCOME")
    line(f"  GPT-5                resolved: {gpt_resolved}/{n_gpt}  submitted: {gpt_submitted}/{n_gpt}")
    line(f"  Claude 4.5 Sonnet    resolved: {claude_resolved}/{n_claude}  submitted: {claude_submitted}/{n_claude}")

    header("COST")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s}")
    line(f"  {'Mean':30s} $ {safe_mean(gpt_costs):>10.2f} $ {safe_mean(claude_costs):>10.2f}")
    line(f"  {'Median':30s} $ {safe_median(gpt_costs):>10.2f} $ {safe_median(claude_costs):>10.2f}")
    line(f"  {'Min':30s} $ {min(gpt_costs) if gpt_costs else 0:>10.2f} $ {min(claude_costs) if claude_costs else 0:>10.2f}")
    line(f"  {'Max':30s} $ {max(gpt_costs) if gpt_costs else 0:>10.2f} $ {max(claude_costs) if claude_costs else 0:>10.2f}")
    line(f"  {'Total':30s} $ {sum(gpt_costs):>10.2f} $ {sum(claude_costs):>10.2f}")
    line(f"  {'Cost per resolve':30s} $ {gpt_cpr:>10.2f} $ {claude_cpr:>10.2f}")

    header("TOKENS")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s}")
    line(f"  {'Input tokens (mean)':30s} {safe_mean(gpt_input):>12,.0f} {safe_mean(claude_input):>12,.0f}")
    line(f"  {'API calls (mean)':30s} {safe_mean(gpt_calls):>12,.0f} {safe_mean(claude_calls):>12,.0f}")
    line(f"  {'Tokens/call (mean)':30s} {safe_mean(gpt_tpc):>12,.0f} {safe_mean(claude_tpc):>12,.0f}")
    line(f"  {'Output tokens (mean)':30s} {safe_mean(gpt_out):>12,.0f} {safe_mean(claude_out):>12,.0f}")
    line(f"  {'  content':30s} {safe_mean(gpt_out_content):>12,.0f} {safe_mean(claude_out_content):>12,.0f}")
    line(f"  {'  tool_call args':30s} {safe_mean(gpt_out_tc):>12,.0f} {safe_mean(claude_out_tc):>12,.0f}")
    line(f"  {'tokens_received *':30s} {safe_mean(gpt_recv):>12,.0f} {safe_mean(claude_recv):>12,.0f}")
    line(f"  {'$/1M input tokens':30s} $ {gpt_per_1m:>10.2f} $ {claude_per_1m:>10.2f}")

    header("TOOL EXECUTION TIME")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s}")
    line(f"  {'Mean total (s)':30s} {safe_mean(gpt_tt):>12.1f} {safe_mean(claude_tt):>12.1f}")
    line(f"  {'Mean max step (s)':30s} {safe_mean(gpt_max_step):>12.1f} {safe_mean(claude_max_step):>12.1f}")
    line(f"  {'Steps >10s (total)':30s} {gpt_over10:>12d} {claude_over10:>12d}")

    header("STEPS & ACTIONS")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s}")
    line(f"  {'Mean steps':30s} {safe_mean(gpt_steps):>12.0f} {safe_mean(claude_steps):>12.0f}")
    for ak in action_keys:
        gv = [s['actions'][ak] for s in gpt]
        cv = [s['actions'][ak] for s in claude]
        label = f"  {ak}"
        line(f"  {label:30s} {safe_mean(gv):>12.1f} {safe_mean(cv):>12.1f}")

    header("CONTENT VOLUME (mean chars/instance)")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s}")
    line(f"  {'Observations (in)':30s} {safe_mean(gpt_obs):>12,.0f} {safe_mean(claude_obs):>12,.0f}")
    line(f"  {'Actions (out)':30s} {safe_mean(gpt_act):>12,.0f} {safe_mean(claude_act):>12,.0f}")
    line(f"  {'Thoughts (out)':30s} {safe_mean(gpt_tht):>12,.0f} {safe_mean(claude_tht):>12,.0f}")
    line(f"  {'Patch size':30s} {safe_mean(gpt_patch):>12,.0f} {safe_mean(claude_patch):>12,.0f}")

    # By-outcome breakdown
    header("BY OUTCOME (mean cost / mean steps / mean input tokens)")
    line(f"  {'':30s} {'GPT-5':>30s}   {'Claude 4.5':>30s}")
    for label, pred in [
        ("Resolved", lambda s: s.get('resolved') is True),
        ("Unresolved", lambda s: s.get('resolved') is False),
        ("Not submitted", lambda s: not s.get('submitted')),
    ]:
        gsub = [s for s in gpt if pred(s)]
        csub = [s for s in claude if pred(s)]
        if not gsub and not csub:
            continue
        def fmt_group(sub):
            if not sub:
                return f"{'—':>30s}"
            c = safe_mean([s['model_stats']['instance_cost'] for s in sub])
            st = safe_mean([s['steps'] for s in sub])
            tk = safe_mean([s['model_stats']['tokens_sent'] for s in sub])
            return f"${c:.2f} / {st:.0f}st / {tk/1e6:.1f}M"
        gf = fmt_group(gsub)
        cf = fmt_group(csub)
        n_g = len(gsub)
        n_c = len(csub)
        line(f"  {label + f' (n={n_g},{n_c})':30s} {gf:>30s}   {cf:>30s}")

    line("")
    line("=" * W)
    line("* tokens_received undercounts: only counts message.content, misses")
    line("  tool_call arguments. See output_tokens for tiktoken-corrected count.")
    line("")
    line(f"Data: {n_gpt + n_claude} trajectories (partial download)")
    line(f"Repos: {len(all_repos)} ({', '.join(all_repos)})")
    line("Config: 250 turn limit, no cost limit, SWE-Agent scaffold")
    line("  GPT-5: reasoning_effort=high")
    line("  Claude 4.5 Sonnet: default")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", help="JSON stats file from extract_stats_fast.py")
    parser.add_argument("-o", "--output", default=None, help="Output text file (default: stdout)")
    args = parser.parse_args()

    with open(args.stats_file) as f:
        data = json.load(f)

    text = summarize(data)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
