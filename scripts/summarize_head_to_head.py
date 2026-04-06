"""
Head-to-head summary: only instances where BOTH models resolved the task.
Shows per-instance deltas so you can see which model was more efficient on
each shared success.

Usage:
    python summarize_head_to_head.py partial_stats.json
    python summarize_head_to_head.py partial_stats.json -o h2h_summary.txt
"""

import argparse
import json
import sys
from collections import defaultdict
from statistics import mean, median


def build_pairs(data):
    """Return list of (instance_id, gpt_stats, claude_stats) for both-resolved instances."""
    by_inst = defaultdict(dict)
    for s in data:
        name = s['config']['model_name']
        model = 'gpt5' if 'gpt' in name.lower() else 'claude'
        by_inst[s['instance_id']][model] = s

    pairs = []
    for iid in sorted(by_inst):
        models = by_inst[iid]
        if ('gpt5' in models and 'claude' in models
                and models['gpt5'].get('resolved') is True
                and models['claude'].get('resolved') is True):
            pairs.append((iid, models['gpt5'], models['claude']))
    return pairs


def safe_mean(vals):
    return mean(vals) if vals else 0

def safe_median(vals):
    return median(vals) if vals else 0


def summarize(data):
    pairs = build_pairs(data)
    n = len(pairs)

    if n == 0:
        return "No instances where both models resolved.\n"

    gpt_all = [p[1] for p in pairs]
    claude_all = [p[2] for p in pairs]

    repos = sorted(set(s['repo'] for s in gpt_all))

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

    line(f"Head-to-Head: Both Models Resolved — {n} instances")
    line("=" * W)
    line("Only tasks where GPT-5 AND Claude 4.5 Sonnet both produced a")
    line("correct patch. Apples-to-apples comparison of efficiency on")
    line("identical successes.")

    header("REPOS")
    for repo in repos:
        count = sum(1 for s in gpt_all if s['repo'] == repo)
        line(f"  {repo:<40s} {count:>3d}")

    # --- COST ---
    gc = [s['model_stats']['instance_cost'] for s in gpt_all]
    cc = [s['model_stats']['instance_cost'] for s in claude_all]
    deltas_cost = [c - g for g, c in zip(gc, cc)]

    header("COST")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s} {'Ratio C/G':>10s}")
    line(f"  {'Mean':30s} $ {mean(gc):>10.2f} $ {mean(cc):>10.2f} {mean(cc)/mean(gc):>10.1f}×")
    line(f"  {'Median':30s} $ {median(gc):>10.2f} $ {median(cc):>10.2f} {median(cc)/median(gc):>10.1f}×")
    line(f"  {'Min':30s} $ {min(gc):>10.2f} $ {min(cc):>10.2f}")
    line(f"  {'Max':30s} $ {max(gc):>10.2f} $ {max(cc):>10.2f}")
    line(f"  {'Total':30s} $ {sum(gc):>10.2f} $ {sum(cc):>10.2f} {sum(cc)/sum(gc):>10.1f}×")
    line(f"")
    line(f"  Claude more expensive on {sum(1 for d in deltas_cost if d > 0)}/{n} instances")

    # --- TOKENS ---
    gi = [s['model_stats']['tokens_sent'] for s in gpt_all]
    ci = [s['model_stats']['tokens_sent'] for s in claude_all]
    ga = [s['model_stats']['api_calls'] for s in gpt_all]
    ca = [s['model_stats']['api_calls'] for s in claude_all]
    gtpc = [s['tokens_per_call'] for s in gpt_all]
    ctpc = [s['tokens_per_call'] for s in claude_all]
    go = [s['output_tokens']['total'] for s in gpt_all]
    co = [s['output_tokens']['total'] for s in claude_all]
    goc = [s['output_tokens']['content'] for s in gpt_all]
    coc = [s['output_tokens']['content'] for s in claude_all]
    got = [s['output_tokens']['tool_call_args'] for s in gpt_all]
    cot = [s['output_tokens']['tool_call_args'] for s in claude_all]

    header("TOKENS")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s} {'Ratio C/G':>10s}")
    line(f"  {'Input tokens (mean)':30s} {mean(gi):>12,.0f} {mean(ci):>12,.0f} {mean(ci)/mean(gi):>10.1f}×")
    line(f"  {'API calls (mean)':30s} {mean(ga):>12,.0f} {mean(ca):>12,.0f} {mean(ca)/mean(ga):>10.1f}×")
    line(f"  {'Tokens/call (mean)':30s} {mean(gtpc):>12,.0f} {mean(ctpc):>12,.0f} {mean(ctpc)/mean(gtpc):>10.1f}×")
    line(f"  {'Output tokens (mean)':30s} {mean(go):>12,.0f} {mean(co):>12,.0f} {mean(co)/mean(go):>10.1f}×")
    line(f"  {'  content':30s} {mean(goc):>12,.0f} {mean(coc):>12,.0f}")
    line(f"  {'  tool_call args':30s} {mean(got):>12,.0f} {mean(cot):>12,.0f}")
    line(f"")
    line(f"  GPT-5 sends more input on {sum(1 for g, c in zip(gi, ci) if g > c)}/{n} instances")

    # --- STEPS & ACTIONS ---
    gs = [s['steps'] for s in gpt_all]
    cs = [s['steps'] for s in claude_all]

    action_keys = ['bash', 'view', 'edit', 'create', 'search_find', 'submit', 'other']

    header("STEPS & ACTIONS")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s} {'Ratio C/G':>10s}")
    line(f"  {'Mean steps':30s} {mean(gs):>12.0f} {mean(cs):>12.0f} {mean(cs)/mean(gs):>10.1f}×")
    for ak in action_keys:
        gv = [s['actions'][ak] for s in gpt_all]
        cv = [s['actions'][ak] for s in claude_all]
        mg, mc = mean(gv), mean(cv)
        ratio = f"{mc/mg:.1f}×" if mg > 0.5 else "—"
        line(f"    {ak:28s} {mg:>12.1f} {mc:>12.1f} {ratio:>10s}")

    # --- TOOL TIME ---
    gtt = [s['tool_time']['total_seconds'] for s in gpt_all]
    ctt = [s['tool_time']['total_seconds'] for s in claude_all]
    gms = [s['tool_time']['max_seconds'] for s in gpt_all]
    cms = [s['tool_time']['max_seconds'] for s in claude_all]

    header("TOOL EXECUTION TIME")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s} {'Ratio C/G':>10s}")
    line(f"  {'Mean total (s)':30s} {mean(gtt):>12.1f} {mean(ctt):>12.1f} {mean(ctt)/mean(gtt):>10.1f}×")
    line(f"  {'Median total (s)':30s} {median(gtt):>12.1f} {median(ctt):>12.1f}")
    line(f"  {'Mean max step (s)':30s} {mean(gms):>12.1f} {mean(cms):>12.1f}")

    # --- CONTENT VOLUME ---
    gobs = [s['content']['observation_chars'] for s in gpt_all]
    cobs = [s['content']['observation_chars'] for s in claude_all]
    gact = [s['content']['action_chars'] for s in gpt_all]
    cact = [s['content']['action_chars'] for s in claude_all]
    gtht = [s['content']['thought_chars'] for s in gpt_all]
    ctht = [s['content']['thought_chars'] for s in claude_all]
    gpat = [s['patch_chars'] for s in gpt_all]
    cpat = [s['patch_chars'] for s in claude_all]

    header("CONTENT VOLUME (mean chars/instance)")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s} {'Ratio C/G':>10s}")
    line(f"  {'Observations (in)':30s} {mean(gobs):>12,.0f} {mean(cobs):>12,.0f} {mean(cobs)/mean(gobs):>10.1f}×")
    line(f"  {'Actions (out)':30s} {mean(gact):>12,.0f} {mean(cact):>12,.0f} {mean(cact)/mean(gact):>10.1f}×")
    line(f"  {'Thoughts (out)':30s} {mean(gtht):>12,.0f} {mean(ctht):>12,.0f} {mean(ctht)/mean(gtht):>10.1f}×")
    line(f"  {'Patch size':30s} {mean(gpat):>12,.0f} {mean(cpat):>12,.0f} {mean(cpat)/mean(gpat):>10.1f}×")

    # --- EFFICIENCY ---
    # tokens per patch char (how many input tokens to produce each char of correct patch)
    g_eff = [g/p if p > 0 else 0 for g, p in zip(gi, gpat)]
    c_eff = [c/p if p > 0 else 0 for c, p in zip(ci, cpat)]

    header("EFFICIENCY (on these shared successes)")
    line(f"  {'':30s} {'GPT-5':>12s} {'Claude 4.5':>12s}")
    line(f"  {'Input tokens / patch char':30s} {mean(g_eff):>12.0f} {mean(c_eff):>12.0f}")
    line(f"  {'$ / patch char':30s} $ {sum(gc)/sum(gpat)*100:>9.2f}¢ $ {sum(cc)/sum(cpat)*100:>9.2f}¢")
    line(f"  {'Mean steps to solve':30s} {mean(gs):>12.0f} {mean(cs):>12.0f}")

    # --- PER-INSTANCE TABLE ---
    header("PER-INSTANCE BREAKDOWN (sorted by GPT-5 cost)")
    line(f"  {'Instance':50s} {'$GPT':>6s} {'$CLD':>6s} {'StG':>4s} {'StC':>4s} {'PchG':>6s} {'PchC':>6s}")
    sep()
    for iid, g, c in sorted(pairs, key=lambda p: p[1]['model_stats']['instance_cost']):
        short_id = iid.replace('instance_', '')
        if len(short_id) > 48:
            short_id = short_id[:48] + '…'
        line(f"  {short_id:50s} "
             f"{g['model_stats']['instance_cost']:>6.2f} "
             f"{c['model_stats']['instance_cost']:>6.2f} "
             f"{g['steps']:>4d} "
             f"{c['steps']:>4d} "
             f"{g['patch_chars']:>6d} "
             f"{c['patch_chars']:>6d}")

    line("")
    line("=" * W)
    line(f"Data: {n} instances where both models resolved (partial download)")
    line(f"Repos: {len(repos)} ({', '.join(repos)})")
    line("Config: 250 turn limit, no cost limit, SWE-Agent scaffold")
    line("  GPT-5: reasoning_effort=high")
    line("  Claude 4.5 Sonnet: default")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", help="JSON stats file from extract_stats_fast.py")
    parser.add_argument("-o", "--output", default=None)
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
