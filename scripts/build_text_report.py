"""
Generate a plain-text report from extracted trajectory stats.
Designed to be consumed by AI agents — no formatting tricks, just
structured data with context.

Usage:
    python scripts/build_text_report.py stats.json -o report.txt
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from statistics import mean, median


def build_pairs(data):
    by_inst = defaultdict(dict)
    for s in data:
        name = s.get('config', {}).get('model_name', '')
        if 'gpt' in name.lower():
            model = 'gpt5'
        elif 'claude' in name.lower():
            model = 'claude'
        else:
            continue
        by_inst[s['instance_id']][model] = s

    paired, unpaired_count, unsub_count = [], 0, 0
    for iid in sorted(by_inst):
        m = by_inst[iid]
        if 'gpt5' not in m or 'claude' not in m:
            unpaired_count += 1
            continue
        if not m['gpt5']['submitted'] or not m['claude']['submitted']:
            unsub_count += 1
            continue
        paired.append(m)
    return paired, unpaired_count, unsub_count


def r(a, b):
    """Ratio string."""
    if not b or b == 0:
        return "n/a"
    return f"{a/b:.1f}x"


def report(data):
    pairs, unpaired, unsub = build_pairs(data)
    n = len(pairs)
    g = [p['gpt5'] for p in pairs]
    c = [p['claude'] for p in pairs]

    both = sum(1 for p in pairs if p['gpt5']['resolved'] and p['claude']['resolved'])
    g_only = sum(1 for p in pairs if p['gpt5']['resolved'] and not p['claude']['resolved'])
    c_only = sum(1 for p in pairs if not p['gpt5']['resolved'] and p['claude']['resolved'])
    neither = n - both - g_only - c_only
    g_resolved = both + g_only
    c_resolved = both + c_only
    repos = sorted(set(s['repo'] for s in g))

    lines = []
    def w(s=""):
        lines.append(s)

    w("SWE-BENCH PRO: GPT-5 vs CLAUDE SONNET 4.5")
    w("=" * 70)
    w()
    w("WHAT THIS IS")
    w("-" * 70)
    w("Paired comparison of GPT-5 and Claude Sonnet 4.5 on SWE-Bench Pro,")
    w("an enterprise software engineering benchmark (731 tasks across 11 repos).")
    w("Both models ran under identical conditions: SWE-Agent scaffold v1.1.0,")
    w("250 turn limit, no cost limit, same tools and prompt.")
    w("Only instances where BOTH models submitted a patch are included.")
    w()
    w(f"Instances: {n} paired (skipped {unpaired} unpaired, {unsub} unsubmitted)")
    w(f"Repos in this data: {', '.join(repos)}")
    w(f"Note: data is from a partial download. Full dataset is ~1460 trajectories.")
    w()

    w("OUTCOME")
    w("-" * 70)
    w(f"Both resolved:                {both:>6d}  ({both/n*100:.1f}%)")
    w(f"Only GPT-5 resolved:          {g_only:>6d}  ({g_only/n*100:.1f}%)")
    w(f"Only Claude Sonnet 4.5:       {c_only:>6d}  ({c_only/n*100:.1f}%)")
    w(f"Neither resolved:             {neither:>6d}  ({neither/n*100:.1f}%)")
    w(f"GPT-5 resolve rate:           {g_resolved}/{n} = {g_resolved/n*100:.1f}%")
    w(f"Claude Sonnet 4.5 resolve rate: {c_resolved}/{n} = {c_resolved/n*100:.1f}%")
    w()

    w("REPOS")
    w("-" * 70)
    for repo in repos:
        rg = sum(1 for p in pairs if p['gpt5']['repo'] == repo and p['gpt5']['resolved'])
        rc = sum(1 for p in pairs if p['claude']['repo'] == repo and p['claude']['resolved'])
        rn = sum(1 for p in pairs if p['gpt5']['repo'] == repo)
        w(f"  {repo:<35s}  n={rn:>3d}  GPT-5 resolved: {rg:>3d}  Claude: {rc:>3d}")
    w()

    # Helper
    def section(title, rows):
        """rows: list of (label, description, gpt5_val, claude_val, ratio_str)"""
        w(title)
        w("-" * 70)
        w(f"{'Metric':<35s} {'GPT-5':>12s} {'Claude S4.5':>12s} {'C/G ratio':>10s}")
        w(f"{'':35s} {'':>12s} {'':>12s} {'':>10s}")
        for label, desc, gv, cv, rat in rows:
            w(f"{label:<35s} {gv:>12s} {cv:>12s} {rat:>10s}")
            if desc:
                w(f"  ^ {desc}")
        w()

    # Cost
    gc = [s['model_stats']['instance_cost'] for s in g]
    cc = [s['model_stats']['instance_cost'] for s in c]
    g_cpr = sum(gc) / g_resolved if g_resolved else float('inf')
    c_cpr = sum(cc) / c_resolved if c_resolved else float('inf')
    g_more = sum(1 for a, b in zip(gc, cc) if a > b)

    section("COST", [
        ("Mean", "Average API cost per task.", f"${mean(gc):.2f}", f"${mean(cc):.2f}", r(mean(cc), mean(gc))),
        ("Median", "Middle value, less affected by outliers.", f"${median(gc):.2f}", f"${median(cc):.2f}", r(median(cc), median(gc))),
        ("Total", "Sum across all filtered tasks.", f"${sum(gc):.2f}", f"${sum(cc):.2f}", r(sum(cc), sum(gc))),
        ("Cost per resolve", "Total spend / tasks fixed.", f"${g_cpr:.2f}", f"${c_cpr:.2f}", r(c_cpr, g_cpr)),
    ])
    w(f"  GPT-5 more expensive on {g_more}/{n} instances.")
    w(f"  Costs reflect Scale AI's internal litellm pricing, not public list prices.")
    w()

    # Tokens
    gi = [s['model_stats']['tokens_sent'] for s in g]
    ci = [s['model_stats']['tokens_sent'] for s in c]
    ga = [s['model_stats']['api_calls'] for s in g]
    ca = [s['model_stats']['api_calls'] for s in c]
    gtpc = [s['tokens_per_call'] for s in g]
    ctpc = [s['tokens_per_call'] for s in c]
    go = [s['output_tokens']['total'] for s in g]
    co = [s['output_tokens']['total'] for s in c]
    goc = [s['output_tokens']['content'] for s in g]
    coc = [s['output_tokens']['content'] for s in c]
    got = [s['output_tokens']['tool_call_args'] for s in g]
    cot = [s['output_tokens']['tool_call_args'] for s in c]
    g1m = sum(gc) / sum(gi) * 1e6 if sum(gi) else 0
    c1m = sum(cc) / sum(ci) * 1e6 if sum(ci) else 0

    section("TOKENS", [
        ("Input tokens (mean)", "Total tokens sent across all API calls. Grows each step as history accumulates.", f"{mean(gi):,.0f}", f"{mean(ci):,.0f}", r(mean(ci), mean(gi))),
        ("API calls (mean)", "Number of model roundtrips.", f"{mean(ga):,.0f}", f"{mean(ca):,.0f}", r(mean(ca), mean(ga))),
        ("Tokens/call (mean)", "Average context size per call.", f"{mean(gtpc):,.0f}", f"{mean(ctpc):,.0f}", r(mean(ctpc), mean(gtpc))),
        ("Output tokens (mean)", "Response text + tool call args, via tiktoken. GPT-5 hidden reasoning not included.", f"{mean(go):,.0f}", f"{mean(co):,.0f}", r(mean(co), mean(go))),
        ("  content", "Text portion of the response.", f"{mean(goc):,.0f}", f"{mean(coc):,.0f}", r(mean(coc), mean(goc)) if mean(goc) > 0.5 else "n/a"),
        ("  tool_call args", "Arguments to tools — edits, commands, queries.", f"{mean(got):,.0f}", f"{mean(cot):,.0f}", r(mean(cot), mean(got))),
        ("$/1M input tokens", "Effective rate paid per million input tokens.", f"${g1m:.2f}", f"${c1m:.2f}", r(c1m, g1m)),
    ])

    # Tool time
    gtt = [s['tool_time']['total_seconds'] for s in g]
    ctt = [s['tool_time']['total_seconds'] for s in c]
    gms = [s['tool_time']['max_seconds'] for s in g]
    cms = [s['tool_time']['max_seconds'] for s in c]
    go10 = sum(s['tool_time']['steps_over_10s'] for s in g)
    co10 = sum(s['tool_time']['steps_over_10s'] for s in c)

    section("TOOL EXECUTION TIME", [
        ("Mean total (s)", "Seconds waiting for tools per task. Does not include model thinking time.", f"{mean(gtt):.1f}", f"{mean(ctt):.1f}", r(mean(ctt), mean(gtt))),
        ("Median total (s)", "Less skewed by tasks with long test suites.", f"{median(gtt):.1f}", f"{median(ctt):.1f}", r(median(ctt), median(gtt))),
        ("Mean max step (s)", "Longest single tool call per task, averaged.", f"{mean(gms):.1f}", f"{mean(cms):.1f}", r(mean(cms), mean(gms))),
        ("Steps >10s (total)", "Tool calls taking over 10 seconds.", f"{go10}", f"{co10}", r(co10, go10)),
    ])

    # Steps & actions
    gs = [s['steps'] for s in g]
    cs = [s['steps'] for s in c]
    action_keys = ['bash', 'view', 'edit', 'create', 'search_find', 'submit', 'other']
    action_descs = {
        'bash': 'Shell commands — tests, deps, output.',
        'view': 'Reading file contents.',
        'edit': 'Modifying existing files.',
        'create': 'Creating new files.',
        'search_find': 'find/grep to locate files.',
        'submit': 'Final patch submission.',
        'other': 'Unclassified.',
    }
    action_rows = [("Mean steps", "Total model turns per task.", f"{mean(gs):.0f}", f"{mean(cs):.0f}", r(mean(cs), mean(gs)))]
    for ak in action_keys:
        gv = [s['actions'][ak] for s in g]
        cv = [s['actions'][ak] for s in c]
        mg, mc = mean(gv), mean(cv)
        action_rows.append((f"  {ak}", action_descs[ak], f"{mg:.1f}", f"{mc:.1f}", r(mc, mg) if mg > 0.5 else "n/a"))
    section("STEPS & ACTIONS", action_rows)

    # Content volume
    gobs = [s['content']['observation_chars'] for s in g]
    cobs = [s['content']['observation_chars'] for s in c]
    gact = [s['content']['action_chars'] for s in g]
    cact = [s['content']['action_chars'] for s in c]
    gtht = [s['content']['thought_chars'] for s in g]
    ctht = [s['content']['thought_chars'] for s in c]
    gpat = [s['patch_chars'] for s in g]
    cpat = [s['patch_chars'] for s in c]
    gptok = [s['patch_tokens'] for s in g]
    cptok = [s['patch_tokens'] for s in c]

    section("CONTENT VOLUME (mean chars/instance)", [
        ("Observations (in)", "Tool output consumed — file contents, stdout, test results.", f"{mean(gobs):,.0f}", f"{mean(cobs):,.0f}", r(mean(cobs), mean(gobs))),
        ("Actions (out)", "What the model asked tools to do.", f"{mean(gact):,.0f}", f"{mean(cact):,.0f}", r(mean(cact), mean(gact))),
        ("Thoughts (out)", "Visible reasoning. GPT-5 uses hidden chain-of-thought.", f"{mean(gtht):,.0f}", f"{mean(ctht):,.0f}", r(mean(ctht), mean(gtht))),
        ("Patch size (chars)", "Final submitted diff.", f"{mean(gpat):,.0f}", f"{mean(cpat):,.0f}", r(mean(cpat), mean(gpat))),
        ("Patch size (tokens)", "Tiktoken count of the submitted diff.", f"{mean(gptok):,.0f}", f"{mean(cptok):,.0f}", r(mean(cptok), mean(gptok))),
    ])

    # Methodology
    w("METHODOLOGY & CAVEATS")
    w("-" * 70)
    w("Source: SWE-Bench Pro leaderboard trajectories from Scale AI S3 bucket.")
    w("Both runs dated October 13, 2025. SWE-Agent v1.1.0, same prompt/tools.")
    w("GPT-5: reasoning_effort=high. Claude Sonnet 4.5: default settings.")
    w()
    w("Token counting:")
    w("- Input tokens: from litellm.utils.token_counter (reliable).")
    w("- Output tokens: counted via tiktoken cl100k_base across message.content")
    w("  + message.thought + tool_calls[].function.arguments.")
    w("- GPT-5 hidden reasoning tokens are billed but invisible — not counted")
    w("  in output tokens. This means GPT-5's true output is higher than shown.")
    w("- The upstream tokens_received field is broken (only counts content,")
    w("  misses tool_call args). We don't use it.")
    w()
    w("Cost:")
    w("- instance_cost from litellm.cost_calculator.completion_cost (reliable).")
    w("- Reflects Scale AI's internal proxy pricing, NOT public list prices.")
    w("- At public list prices the cost ratio would be different.")
    w()
    w("Action classification:")
    w("- Classified by action string prefix. grep reclassified from bash to")
    w("  search_find. 'create' correctly separated from 'edit' (earlier bug")
    w("  where str_replace_editor matched str_replace before create was fixed).")
    w()
    w("What's NOT measured:")
    w("- Wall-clock time per instance (not recorded anywhere).")
    w("- LLM inference latency (not recorded).")
    w("- GPT-5 hidden reasoning token count.")
    w("- Input/output cost breakdown (only blended total).")
    w()
    w("=" * 70)
    w(f"Generated from {n} paired instances (partial download).")
    w(f"Full dataset: ~1460 trajectories across 11 repos, 4 languages.")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    with open(args.stats_file) as f:
        data = json.load(f)

    text = report(data)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
