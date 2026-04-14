#!/usr/bin/env python3
"""
Build an HTML analytics page comparing GPT-5 vs Claude 4.5 trajectories.

Charts:
  1. High-level letter frequencies (bar chart, side-by-side)
  2. Low-level intent frequencies (bar chart, side-by-side)
  3. Top 2-letter transitions / bigrams (bar chart, side-by-side)
  4. Typical trajectory shape visualisations

Usage:
  python scripts/build_analytics.py --data-root data --output docs/analytics.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import classify_intent as ci

# Single-letter codes for transition matrices and sequence analysis only
HIGH_LEVEL_LETTER = {
    "read": "R",
    "search": "S",
    "reproduce": "P",
    "edit": "E",
    "verify": "V",
    "git": "G",
    "housekeeping": "H",
    "failed": "X",
    "other": "O",
}
# Reverse: letter → short display name (used in transition matrices)
LETTER_TO_NAME = {v: k for k, v in HIGH_LEVEL_LETTER.items()}

# Colors keyed by the short name (not letter)
HIGH_LEVEL_COLORS = {
    "read": "#6cb6ff",
    "search": "#f0883e",
    "reproduce": "#bc8cff",
    "edit": "#3fb950",
    "verify": "#f778ba",
    "git": "#e0c745",
    "housekeeping": "#39c5cf",
    "failed": "#f85149",
    "other": "#545d68",
}
# Also keyed by letter for transition matrix rendering
LETTER_COLORS = {HIGH_LEVEL_LETTER[k]: v for k, v in HIGH_LEVEL_COLORS.items()}


def parse_repo(instance_id: str) -> str:
    core = instance_id
    if core.startswith("instance_"):
        core = core[len("instance_"):]
    if "__" not in core:
        return "unknown/unknown"
    org, rest = core.split("__", 1)
    m = re.search(r"-(?:[0-9a-f]{7,40})(?:-v.*)?$", rest)
    repo = rest[: m.start()] if m else rest
    return f"{org}/{repo}"


def collect_files(data_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for model in ("gpt5", "claude45"):
        base = data_root / model / "traj"
        if not base.exists():
            continue
        for p in sorted(base.glob("*/*.traj")):
            out.append((model, p))
    return out


def _process_one_file(args: tuple[str, str]) -> dict:
    """Process a single .traj file. Top-level function for pickling."""
    model, path_str = args
    data = ci._load_json(path_str)
    trajectory = data.get("trajectory", [])
    if not trajectory:
        return {"model": model, "empty": True}

    base_intents = ci.classify_trajectory(trajectory)
    hierarchical = ci.classify_hierarchical_layer(base_intents)

    highs = [h.split(".", 1)[0] for h in hierarchical]
    high_letters = [HIGH_LEVEL_LETTER.get(h, "?") for h in highs]
    high_seq = "".join(high_letters)

    high_c = Counter(high_letters)
    low_c = Counter(base_intents)
    bigram_c = Counter()
    for i in range(len(high_letters) - 1):
        bigram_c[high_letters[i] + high_letters[i + 1]] += 1

    # Phase profile: 20 bins
    phase: dict[str, list[float]] = {}
    n = len(high_letters)
    if n >= 5:
        bins = 20
        for letter in HIGH_LEVEL_LETTER.values():
            counts_in_bin = []
            for b in range(bins):
                start = int(b * n / bins)
                end = int((b + 1) * n / bins)
                segment = high_letters[start:end]
                counts_in_bin.append(segment.count(letter) / len(segment) if segment else 0.0)
            phase[letter] = counts_in_bin

    return {
        "model": model,
        "empty": False,
        "high_c": dict(high_c),
        "low_c": dict(low_c),
        "bigram_c": dict(bigram_c),
        "high_seq": high_seq,
        "steps": len(base_intents),
        "phase": phase,
    }


def _load_intent_descriptions() -> dict[str, str]:
    desc_path = Path(__file__).parent / "intent_descriptions.json"
    with open(desc_path) as f:
        return json.load(f)


def build_payload(data_root: Path) -> dict:
    files = collect_files(data_root)

    # Per-model accumulators
    high_counts: dict[str, Counter] = {"gpt5": Counter(), "claude45": Counter()}
    low_counts: dict[str, Counter] = {"gpt5": Counter(), "claude45": Counter()}
    bigram_counts: dict[str, Counter] = {"gpt5": Counter(), "claude45": Counter()}

    all_high_seqs: dict[str, list[str]] = {"gpt5": [], "claude45": []}
    step_counts: dict[str, list[int]] = {"gpt5": [], "claude45": []}
    phase_profiles: dict[str, dict[str, list[list[float]]]] = {
        "gpt5": {l: [] for l in HIGH_LEVEL_LETTER.values()},
        "claude45": {l: [] for l in HIGH_LEVEL_LETTER.values()},
    }

    tasks = [(model, str(path)) for model, path in files]
    max_workers = min(8, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_process_one_file, tasks, chunksize=32):
            if result["empty"]:
                continue
            model = result["model"]
            high_counts[model].update(result["high_c"])
            low_counts[model].update(result["low_c"])
            bigram_counts[model].update(result["bigram_c"])
            all_high_seqs[model].append(result["high_seq"])
            step_counts[model].append(result["steps"])
            for letter, bins in result["phase"].items():
                phase_profiles[model][letter].append(bins)

    # Aggregate phase profiles: average across trajectories
    avg_phase: dict[str, dict[str, list[float]]] = {}
    for model in ("gpt5", "claude45"):
        avg_phase[model] = {}
        for letter in HIGH_LEVEL_LETTER.values():
            profiles = phase_profiles[model][letter]
            if not profiles:
                avg_phase[model][letter] = [0.0] * 20
                continue
            bins = len(profiles[0])
            avg_phase[model][letter] = [
                sum(p[b] for p in profiles) / len(profiles) for b in range(bins)
            ]

    # Normalize counts to proportions for fair comparison
    def to_proportions(counter: Counter) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    # Build transition matrices (row=from, col=to) as proportions
    ordered_letters = ["R", "S", "P", "E", "V", "G", "H", "X", "O"]
    bigram_matrix: dict[str, list[list[float]]] = {}
    for model in ("gpt5", "claude45"):
        total_bg = sum(bigram_counts[model].values())
        if total_bg == 0:
            bigram_matrix[model] = [[0.0] * len(ordered_letters) for _ in ordered_letters]
            continue
        matrix = []
        for fr in ordered_letters:
            row = []
            for to in ordered_letters:
                row.append(bigram_counts[model].get(fr + to, 0) / total_bg)
            matrix.append(row)
        bigram_matrix[model] = matrix

    # Also keep top bigrams list for reference
    top_bigrams_set = set()
    for model in ("gpt5", "claude45"):
        for bg, _ in bigram_counts[model].most_common(20):
            top_bigrams_set.add(bg)
    top_bigrams = sorted(top_bigrams_set, key=lambda b: -(bigram_counts["gpt5"].get(b, 0) + bigram_counts["claude45"].get(b, 0)))

    # Get all intents from displayed categories (exclude failed/X)
    displayed_categories = {"read", "search", "reproduce", "edit", "verify", "git", "housekeeping", "other"}
    top_low_set = set()
    for intent, high in ci.INTENT_TO_HIGH_LEVEL.items():
        if high in displayed_categories:
            if low_counts["gpt5"].get(intent, 0) + low_counts["claude45"].get(intent, 0) > 0:
                top_low_set.add(intent)
    top_low = sorted(top_low_set, key=lambda i: -(low_counts["gpt5"].get(i, 0) + low_counts["claude45"].get(i, 0)))

    # Step count distributions (binned)
    def bin_steps(counts, bin_size=5):
        if not counts:
            return {}
        bins = Counter()
        for c in counts:
            b = (c // bin_size) * bin_size
            bins[b] += 1
        return dict(sorted(bins.items()))

    return {
        "high_counts": {m: dict(high_counts[m].most_common()) for m in ("gpt5", "claude45")},
        "high_proportions": {
            m: {LETTER_TO_NAME.get(k, k): v for k, v in to_proportions(high_counts[m]).items()}
            for m in ("gpt5", "claude45")
        },
        "low_proportions": {m: to_proportions(low_counts[m]) for m in ("gpt5", "claude45")},
        "top_low_intents": top_low,
        "bigram_matrix": bigram_matrix,
        "bigram_letters": ordered_letters,
        "avg_phase": avg_phase,
        "step_dist": {m: bin_steps(step_counts[m]) for m in ("gpt5", "claude45")},
        "num_trajs": {m: len(all_high_seqs[m]) for m in ("gpt5", "claude45")},
        "high_level_letter": HIGH_LEVEL_LETTER,
        "letter_to_name": LETTER_TO_NAME,
        "name_to_letter": HIGH_LEVEL_LETTER,
        "letter_colors": LETTER_COLORS,
        "name_colors": HIGH_LEVEL_COLORS,
        "intent_to_category": dict(ci.INTENT_TO_HIGH_LEVEL),
        "intent_descriptions": _load_intent_descriptions(),
    }


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>GPT-5 vs Claude 4.5 — Trajectory Analytics</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #121922;
      --muted: #9fb0c0;
      --text: #e8eef5;
      --accent: #6cb6ff;
      --border: #2a3645;
      --gpt: #f0883e;
      --claude: #6cb6ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Inter, system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 28px; margin-bottom: 6px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 30px; }}
    h2 {{
      font-size: 18px;
      margin: 40px 0 6px 0;
      padding-top: 20px;
      border-top: 1px solid var(--border);
    }}
    .chart-desc {{ color: var(--muted); font-size: 13px; margin-bottom: 14px; }}
    .legend {{
      display: flex; gap: 20px; margin-bottom: 16px; font-size: 13px;
    }}
    .legend-item {{
      display: flex; align-items: center; gap: 6px;
    }}
    .legend-swatch {{
      width: 14px; height: 14px; border-radius: 3px;
    }}
    .chart-wrapper {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 10px;
      overflow-x: auto;
    }}
    canvas {{ display: block; }}

    /* Phase heatmap */
    .heatmap-grid {{
      display: grid;
      grid-template-columns: 80px repeat(10, 1fr);
      gap: 2px;
      font-size: 12px;
      margin-bottom: 10px;
    }}
    .heatmap-label {{
      display: flex; align-items: center; color: var(--muted);
      font-family: ui-monospace, monospace;
    }}
    .heatmap-cell {{
      height: 28px;
      border-radius: 3px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      color: rgba(255,255,255,0.7);
    }}
    .heatmap-header {{
      font-size: 11px; color: var(--muted); text-align: center;
    }}
    .model-tag {{
      display: inline-block;
      padding: 1px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .model-tag.gpt {{ background: rgba(240,136,62,0.2); color: var(--gpt); }}
    .model-tag.claude {{ background: rgba(108,182,255,0.2); color: var(--claude); }}

    .side-by-side {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 800px) {{
      .side-by-side {{ grid-template-columns: 1fr; }}
    }}
    .side-by-side .chart-wrapper {{ margin-bottom: 0; }}
    .side-label {{
      font-size: 14px; font-weight: 600; margin-bottom: 10px;
    }}

    /* Dumbbell chart */
    .dumbbell-row {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px;
      align-items: center;
      padding: 4px 0;
      border-bottom: 1px solid #1a2230;
    }}
    .dumbbell-row:last-child {{ border-bottom: none; }}
    .dumbbell-label {{
      text-align: right;
      font-family: ui-monospace, monospace;
      font-size: 12px;
      color: var(--text);
    }}
    .dumbbell-track {{
      position: relative;
      height: 22px;
    }}
    .dumbbell-line {{
      position: absolute;
      top: 10px;
      height: 2px;
      border-radius: 1px;
    }}
    .dumbbell-dot {{
      position: absolute;
      top: 3px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .dumbbell-val {{
      position: absolute;
      top: 2px;
      font-family: ui-monospace, monospace;
      font-size: 11px;
      white-space: nowrap;
    }}
    .dumbbell-header {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px;
      font-size: 11px;
      color: var(--muted);
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 4px;
    }}
    .dumbbell-scale {{
      position: relative;
      height: 16px;
    }}
    .dumbbell-scale-tick {{
      position: absolute;
      top: 0;
      font-size: 10px;
      color: var(--muted);
      font-family: ui-monospace, monospace;
      transform: translateX(-50%);
    }}

    /* Heat table */
    .heat-table {{
      width: 100%;
      border-collapse: collapse;
      font-family: ui-monospace, monospace;
      font-size: 12px;
    }}
    .heat-table th {{
      text-align: right;
      padding: 4px 10px;
      font-size: 11px;
      color: var(--muted);
      font-weight: 400;
      border-bottom: 1px solid var(--border);
    }}
    .heat-table th.col-gpt {{ color: var(--gpt); font-weight: 600; }}
    .heat-table th.col-claude {{ color: var(--claude); font-weight: 600; }}
    .heat-table td {{
      padding: 3px 0;
      border-bottom: 1px solid #1a2230;
    }}
    .heat-table td.intent-name {{
      text-align: right;
      padding-right: 12px;
      color: var(--text);
      width: 200px;
    }}
    .heat-table td.bar-cell {{
      width: 25%;
      padding: 3px 4px;
    }}
    .heat-table .bar-wrap {{
      position: relative;
      height: 20px;
      display: flex;
      align-items: center;
    }}
    .heat-table .bar-bg {{
      position: absolute;
      top: 1px;
      height: 18px;
      border-radius: 3px;
      opacity: 0.7;
    }}
    .heat-table .bar-val {{
      position: relative;
      z-index: 1;
      padding: 0 6px;
      font-size: 11px;
      color: var(--text);
    }}
    .heat-table .bar-cell.gpt .bar-wrap {{
      justify-content: flex-end;
    }}
    .heat-table .bar-cell.gpt .bar-bg {{
      right: 0;
    }}
    .heat-table .bar-cell.claude .bar-bg {{
      left: 0;
    }}
    .heat-table .intent-desc {{
      font-family: Inter, system-ui, sans-serif;
      font-size: 11px;
      color: var(--muted);
      font-weight: 400;
      margin-top: 1px;
    }}
    .heat-table .cat-header td {{
      padding: 10px 0 4px 0;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
      font-size: 12px;
    }}

    /* Stacked area legend */
    .stacked-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin-top: 10px;
      font-size: 12px;
      font-family: ui-monospace, monospace;
      color: var(--text);
    }}
    .stacked-legend .item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .stacked-legend .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
      flex-shrink: 0;
    }}

    /* Transition matrix */
    .tmatrix {{
      display: inline-grid;
      gap: 2px;
      font-size: 11px;
      font-family: ui-monospace, monospace;
    }}
    .tmatrix .corner {{ }}
    .tmatrix .col-hdr {{
      text-align: center;
      color: var(--text);
      font-weight: 700;
      padding: 2px 0;
    }}
    .tmatrix .row-hdr {{
      display: flex; align-items: center; justify-content: flex-end;
      color: var(--text);
      font-weight: 700;
      padding-right: 6px;
    }}
    .tmatrix .cell {{
      width: 48px; height: 32px;
      border-radius: 3px;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px;
      color: rgba(255,255,255,0.85);
      transition: transform 0.1s;
    }}
    .tmatrix .cell:hover {{
      transform: scale(1.15);
      z-index: 2;
      outline: 1px solid var(--text);
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>GPT-5 vs Claude 4.5</h1>
  <div class="subtitle">Trajectory behaviour analytics &mdash; SWE-Bench Pro</div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--gpt)"></div>
      <span>GPT-5</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--claude)"></div>
      <span>Claude 4.5</span>
    </div>
  </div>

  <h2>1. High-Level Action Frequencies</h2>
  <p class="chart-desc">Proportion of steps in each high-level category. Normalised so models are comparable despite different step counts.</p>
  <div class="chart-wrapper">
    <canvas id="highChart" height="300"></canvas>
  </div>

  <h2>2. Low-Level Intent Differences</h2>
  <p class="chart-desc">Per 100 steps, how many times does each model perform this action? <span style="color:var(--gpt)">Orange dot</span> = GPT-5, <span style="color:var(--claude)">blue dot</span> = Claude 4.5. The bar between dots is the gap. Sorted by largest gap within each category.</p>
  <div class="chart-wrapper">
    <div id="lowDumbbell"></div>
  </div>

  <h2>2b. Intent Comparison Table</h2>
  <p class="chart-desc">Bar width = frequency per 100 steps. Compare bar lengths within each row to spot differences; compare across rows to see which intents dominate overall.</p>
  <div class="chart-wrapper">
    <div id="heatTable"></div>
  </div>

  <h2>3. Transition Matrices</h2>
  <p class="chart-desc">Each cell shows how often one action follows another (as % of all transitions). Diagonal = self-repeats. Read row &rarr; column as "from &rarr; to".</p>
  <div class="side-by-side">
    <div class="chart-wrapper">
      <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
      <div id="matrixGpt"></div>
    </div>
    <div class="chart-wrapper">
      <div class="side-label"><span class="model-tag claude">Claude 4.5</span></div>
      <div id="matrixClaude"></div>
    </div>
  </div>
  <div class="chart-wrapper" style="margin-top:16px">
    <div class="side-label" style="margin-bottom:12px">Difference (GPT &minus; Claude)</div>
    <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:12px;margin-bottom:14px;color:var(--muted)">
      <div style="display:flex;align-items:center;gap:6px">
        <div style="width:32px;height:16px;border-radius:3px;background:#f0883e"></div>
        <span>GPT does this transition more</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <div style="width:32px;height:16px;border-radius:3px;background:#6cb6ff"></div>
        <span>Claude does this transition more</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <div style="width:32px;height:16px;border-radius:3px;background:#333"></div>
        <span>~equal</span>
      </div>
      <div><span style="color:var(--text)">Brighter</span> = larger gap &nbsp; <span style="color:var(--text)">Values</span> = percentage point difference</div>
    </div>
    <div id="matrixDiff"></div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:11px;margin-top:14px;color:var(--muted);font-family:ui-monospace,monospace">
      <span><strong style="color:var(--text)">R</strong>=read</span>
      <span><strong style="color:var(--text)">S</strong>=search</span>
      <span><strong style="color:var(--text)">P</strong>=reproduce</span>
      <span><strong style="color:var(--text)">E</strong>=edit</span>
      <span><strong style="color:var(--text)">V</strong>=verify</span>
      <span><strong style="color:var(--text)">G</strong>=git</span>
      <span><strong style="color:var(--text)">H</strong>=housekeeping</span>
      <span><strong style="color:var(--text)">X</strong>=failed</span>
      <span><strong style="color:var(--text)">O</strong>=other</span>
    </div>
  </div>

  <h2>4. Trajectory Length Distribution</h2>
  <p class="chart-desc">How many steps each model typically takes per task.</p>
  <div class="chart-wrapper">
    <canvas id="stepDistChart" height="280"></canvas>
  </div>

  <h2>5. Phase Profile — When Does Each Action Happen?</h2>
  <p class="chart-desc">Each trajectory is divided into 20 equal time-slices (0%–100%). The heatmap shows what proportion of steps in each slice belong to each category. Read left-to-right as "beginning → end of trajectory".</p>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
    <div id="heatmapGpt"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag claude">Claude 4.5</span></div>
    <div id="heatmapClaude"></div>
  </div>

  <h2>5b. Phase Dominance — What Dominates Each Slice?</h2>
  <p class="chart-desc">Same data, but normalised per column: within each time-slice, which category takes the largest share? Brighter = dominates that phase.</p>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
    <div id="heatmapColGpt"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag claude">Claude 4.5</span></div>
    <div id="heatmapColClaude"></div>
  </div>

  <h2>6. Typical Trajectory Shape</h2>
  <p class="chart-desc">Stacked area chart: how the mix of actions evolves from start to end of the average trajectory.</p>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
    <canvas id="stackedGpt" height="200"></canvas>
    <div class="stacked-legend" id="legendGpt"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag claude">Claude 4.5</span></div>
    <canvas id="stackedClaude" height="200"></canvas>
    <div class="stacked-legend" id="legendClaude"></div>
  </div>

</div>

<script>
const D = {payload_json};

// ── Helpers ──────────────────────────────────────────────
function getCtx(id) {{
  const c = document.getElementById(id);
  c.width = c.parentElement.clientWidth - 40;
  return {{ canvas: c, ctx: c.getContext('2d'), w: c.width, h: c.height }};
}}

const GPT_COLOR = '#f0883e';
const CLAUDE_COLOR = '#6cb6ff';
const MUTED = '#546170';
const TEXT = '#c8d4df';

// Names for display (everywhere except transition matrices)
const CATEGORY_NAMES = ['read','search','reproduce','edit','verify','git','housekeeping'];
const NAME_COLORS = D.name_colors;

// Letters for transition matrices only
const LETTERS = Object.values(D.name_to_letter);
const LETTER_COLORS = D.letter_colors;
const LETTER_TO_NAME = D.letter_to_name;

// ── 1. High-Level Bar Chart ──────────────────────────────
function drawGroupedBar(canvasId, labels, gptVals, claudeVals, labelMap) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 50, right = 20, top = 20, bot = 60;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = labels.length;
  const groupW = plotW / n;
  const barW = groupW * 0.35;
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.01);

  // Y axis
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = top + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(1) + '%', left - 6, y + 4);
  }}

  for (let i = 0; i < n; i++) {{
    const x = left + i * groupW + groupW * 0.1;
    const gH = (gptVals[i] / maxVal) * plotH;
    const cH = (claudeVals[i] / maxVal) * plotH;

    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(x, top + plotH - gH, barW, gH);

    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(x + barW + 2, top + plotH - cH, barW, cH);

    // Label
    ctx.fillStyle = TEXT;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    const lx = x + barW + 1;
    const label = labels[i];
    ctx.fillText(label, lx, top + plotH + 16);
    if (labelMap && labelMap[label]) {{
      ctx.fillStyle = MUTED;
      ctx.font = '10px sans-serif';
      ctx.fillText(labelMap[label], lx, top + plotH + 30);
    }}
  }}
}}

// ── 2. Horizontal grouped bar (for long labels) ─────────
function drawHorizontalGroupedBar(canvasId, labels, gptVals, claudeVals) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 160, right = 30, top = 10, bot = 30;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = labels.length;
  const rowH = plotH / n;
  const barH = rowH * 0.35;
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.001);

  // Vertical grid
  ctx.strokeStyle = MUTED; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const x = left + (i / 4) * plotW;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(1) + '%', x, top + plotH + 16);
  }}

  for (let i = 0; i < n; i++) {{
    const y = top + i * rowH + rowH * 0.15;
    const gW = (gptVals[i] / maxVal) * plotW;
    const cW = (claudeVals[i] / maxVal) * plotW;

    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(left, y, gW, barH);

    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(left, y + barH + 1, cW, barH);

    ctx.fillStyle = TEXT; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(labels[i], left - 8, y + rowH * 0.45);
  }}
}}

// ── Draw charts ──────────────────────────────────────────

// 1. High-level
const highLabels = CATEGORY_NAMES;
const gptHigh = highLabels.map(l => D.high_proportions.gpt5[l] || 0);
const claudeHigh = highLabels.map(l => D.high_proportions.claude45[l] || 0);
drawGroupedBar('highChart', highLabels, gptHigh, claudeHigh, null);

// 2. Low-level dumbbell chart, grouped by category
(function() {{
  const el = document.getElementById('lowDumbbell');
  const intents = D.top_low_intents;
  const catMap = D.intent_to_category || {{}};
  const catOrder = ['read','search','reproduce','edit','verify','git','housekeeping','other'];

  // Convert to "per 100 steps"
  const rows = intents.map(intent => {{
    const g = (D.low_proportions.gpt5[intent] || 0) * 100;
    const c = (D.low_proportions.claude45[intent] || 0) * 100;
    const cat = catMap[intent] || '?';
    return {{ intent, g, c, gap: Math.abs(g - c), cat }};
  }});

  // Group by category, sort by gap within each
  const grouped = {{}};
  for (const r of rows) {{
    if (!grouped[r.cat]) grouped[r.cat] = [];
    grouped[r.cat].push(r);
  }}
  for (const cat of Object.keys(grouped)) {{
    grouped[cat].sort((a, b) => b.gap - a.gap);
  }}
  const sorted = [];
  for (const cat of catOrder) {{
    if (grouped[cat]) sorted.push(...grouped[cat]);
  }}

  // Scale: 0 to max value, mapped to 8%–92% of track width for label room
  const maxVal = Math.max(...sorted.map(r => Math.max(r.g, r.c)), 1);
  const PAD_L = 8;  // % left padding
  const PAD_R = 92; // % right limit
  const RANGE = PAD_R - PAD_L;

  function pos(v) {{ return (PAD_L + (v / maxVal) * RANGE).toFixed(2); }}

  // Scale ticks
  let html = `<div class="dumbbell-header">
    <div style="text-align:right;font-size:12px">per 100 steps</div>
    <div class="dumbbell-scale">`;
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {{
    const v = (maxVal * i / nTicks);
    html += `<span class="dumbbell-scale-tick" style="left:${{pos(v)}}%">${{v.toFixed(0)}}</span>`;
  }}
  html += `</div></div>`;

  let prevCat = '';
  for (const r of sorted) {{
    if (r.cat !== prevCat) {{
      const catName = r.cat;
      const catColor = NAME_COLORS[r.cat] || MUTED;
      html += `<div style="padding:8px 0 4px 0;font-size:12px;font-weight:600;color:${{catColor}};border-top:1px solid var(--border);margin-top:4px">
        ${{r.cat}} ${{catName}}
      </div>`;
      prevCat = r.cat;
    }}

    const lo = Math.min(r.g, r.c);
    const hi = Math.max(r.g, r.c);
    const loPos = pos(lo);
    const hiPos = pos(hi);
    const gPos = pos(r.g);
    const cPos = pos(r.c);

    // Bar color: who has more?
    const barColor = r.g > r.c ? GPT_COLOR : CLAUDE_COLOR;

    // Lower value label to the LEFT of its dot, higher value label to the RIGHT
    const loModel = r.g <= r.c ? 'g' : 'c';
    const loColor = loModel === 'g' ? GPT_COLOR : CLAUDE_COLOR;
    const hiColor = loModel === 'g' ? CLAUDE_COLOR : GPT_COLOR;

    // Hide lower label when dots nearly overlap
    const showLo = r.gap >= 0.3;

    html += `<div class="dumbbell-row">
      <div class="dumbbell-label">${{r.intent}}</div>
      <div class="dumbbell-track">
        <div class="dumbbell-line" style="left:${{loPos}}%;width:${{(parseFloat(hiPos)-parseFloat(loPos)).toFixed(2)}}%;background:${{barColor}};opacity:0.5"></div>
        <div class="dumbbell-dot" style="left:calc(${{gPos}}% - 8px);background:${{GPT_COLOR}}"></div>
        <div class="dumbbell-dot" style="left:calc(${{cPos}}% - 8px);background:${{CLAUDE_COLOR}}"></div>
        <div class="dumbbell-val" style="right:${{(100 - parseFloat(loPos)).toFixed(2)}}%;text-align:right;padding-right:20px;color:${{loColor}};opacity:${{showLo ? 1 : 0}}">${{lo.toFixed(1)}}</div>
        <div class="dumbbell-val" style="left:${{hiPos}}%;padding-left:20px;color:${{hiColor}}">${{hi.toFixed(1)}}</div>
      </div>
    </div>`;
  }}

  el.innerHTML = html;
}})();

// 2b. Heat table
(function() {{
  const el = document.getElementById('heatTable');
  const intents = D.top_low_intents;
  const catMap = D.intent_to_category || {{}};
  const catOrder = ['read','search','reproduce','edit','verify','git','housekeeping','other'];

  const rows = intents.map(intent => {{
    const g = (D.low_proportions.gpt5[intent] || 0) * 100;
    const c = (D.low_proportions.claude45[intent] || 0) * 100;
    const cat = catMap[intent] || '?';
    return {{ intent, g, c, gap: Math.abs(g - c), cat }};
  }});

  const grouped = {{}};
  for (const r of rows) {{
    if (!grouped[r.cat]) grouped[r.cat] = [];
    grouped[r.cat].push(r);
  }}
  for (const cat of Object.keys(grouped)) {{
    grouped[cat].sort((a, b) => b.gap - a.gap);
  }}
  const sorted = [];
  for (const cat of catOrder) {{
    if (grouped[cat]) sorted.push(...grouped[cat]);
  }}

  const maxVal = Math.max(...sorted.map(r => Math.max(r.g, r.c)), 1);

  function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

  let html = `<table class="heat-table">
    <thead><tr>
      <th></th>
      <th class="col-gpt" style="text-align:center">GPT-5</th>
      <th class="col-claude" style="text-align:center">Claude 4.5</th>
    </tr></thead><tbody>`;

  let prevCat = '';
  for (const r of sorted) {{
    if (r.cat !== prevCat) {{
      const catName = r.cat;
      const catColor = NAME_COLORS[r.cat] || MUTED;
      html += `<tr class="cat-header"><td colspan="3" style="color:${{catColor}}">${{r.cat}} ${{catName}}</td></tr>`;
      prevCat = r.cat;
    }}

    const desc = (D.intent_descriptions || {{}})[r.intent] || '';
    const gBold = r.g > r.c && r.gap >= 0.3;
    const cBold = r.c > r.g && r.gap >= 0.3;

    html += `<tr>
      <td class="intent-name">${{r.intent}}${{desc ? `<div class="intent-desc">${{desc}}</div>` : ''}}</td>
      <td class="bar-cell gpt">
        <div class="bar-wrap">
          <div class="bar-bg" style="width:${{barPct(r.g)}}%;background:${{GPT_COLOR}}"></div>
          <span class="bar-val" style="${{gBold ? 'font-weight:700' : ''}}">${{r.g.toFixed(1)}}</span>
        </div>
      </td>
      <td class="bar-cell claude">
        <div class="bar-wrap">
          <div class="bar-bg" style="width:${{barPct(r.c)}}%;background:${{CLAUDE_COLOR}}"></div>
          <span class="bar-val" style="${{cBold ? 'font-weight:700' : ''}}">${{r.c.toFixed(1)}}</span>
        </div>
      </td>
    </tr>`;
  }}

  html += '</tbody></table>';
  el.innerHTML = html;
}})();

// 3. Transition matrices
function drawTransitionMatrix(containerId, matrix, mode) {{
  const el = document.getElementById(containerId);
  const letters = D.bigram_letters;
  const n = letters.length;
  const cols = n + 1; // +1 for row header
  el.innerHTML = '';

  const grid = document.createElement('div');
  grid.className = 'tmatrix';
  grid.style.gridTemplateColumns = `40px repeat(${{n}}, 48px)`;

  // Corner
  const corner = document.createElement('div');
  corner.className = 'corner';
  corner.innerHTML = '<span style="color:#546170;font-size:9px">from\\to</span>';
  grid.appendChild(corner);

  // Column headers
  for (const l of letters) {{
    const hdr = document.createElement('div');
    hdr.className = 'col-hdr';
    hdr.textContent = l;
    grid.appendChild(hdr);
  }}

  // Find max for color scaling
  let maxVal = 0;
  if (mode === 'diff') {{
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++)
        maxVal = Math.max(maxVal, Math.abs(matrix[r][c]));
  }} else {{
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++)
        maxVal = Math.max(maxVal, matrix[r][c]);
  }}
  if (maxVal === 0) maxVal = 0.01;

  for (let r = 0; r < n; r++) {{
    // Row header
    const rh = document.createElement('div');
    rh.className = 'row-hdr';
    rh.textContent = letters[r];
    grid.appendChild(rh);

    for (let c = 0; c < n; c++) {{
      const cell = document.createElement('div');
      cell.className = 'cell';
      const v = matrix[r][c];

      if (mode === 'diff') {{
        const intensity = Math.sqrt(Math.abs(v) / maxVal);
        const alpha = 0.15 + intensity * 0.85;
        const color = v > 0.0001 ? GPT_COLOR : v < -0.0001 ? CLAUDE_COLOR : '#333';
        cell.style.background = color;
        cell.style.opacity = alpha.toFixed(2);
        if (Math.abs(v) > 0.001) {{
          cell.textContent = (v > 0 ? '+' : '') + (v * 100).toFixed(1);
        }}
      }} else {{
        const intensity = Math.sqrt(v / maxVal);
        const alpha = 0.2 + intensity * 0.8;
        const color = LETTER_COLORS[letters[r]] || '#6cb6ff';
        cell.style.background = color;
        cell.style.opacity = alpha.toFixed(2);
        if (v > 0.001) {{
          cell.textContent = (v * 100).toFixed(1);
        }}
      }}

      cell.title = `${{letters[r]}} → ${{letters[c]}}: ${{(v * 100).toFixed(2)}}%`;
      grid.appendChild(cell);
    }}
  }}

  el.appendChild(grid);
}}

drawTransitionMatrix('matrixGpt', D.bigram_matrix.gpt5, 'single');
drawTransitionMatrix('matrixClaude', D.bigram_matrix.claude45, 'single');

// Diff matrix
const diffMatrix = D.bigram_matrix.gpt5.map((row, r) =>
  row.map((v, c) => v - D.bigram_matrix.claude45[r][c])
);
drawTransitionMatrix('matrixDiff', diffMatrix, 'diff');

// 4. Step distribution
(function() {{
  const {{ ctx, w, h }} = getCtx('stepDistChart');
  const left = 50, right = 20, top = 20, bot = 50;
  const allBins = new Set([
    ...Object.keys(D.step_dist.gpt5).map(Number),
    ...Object.keys(D.step_dist.claude45).map(Number),
  ]);
  const bins = [...allBins].sort((a,b) => a - b);
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = bins.length;
  const groupW = plotW / n;
  const barW = groupW * 0.38;

  const gptTotal = Object.values(D.step_dist.gpt5).reduce((a,b) => a+b, 0);
  const claudeTotal = Object.values(D.step_dist.claude45).reduce((a,b) => a+b, 0);

  const gptVals = bins.map(b => (D.step_dist.gpt5[b] || 0) / gptTotal);
  const claudeVals = bins.map(b => (D.step_dist.claude45[b] || 0) / claudeTotal);
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.01);

  ctx.strokeStyle = MUTED; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = top + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(0) + '%', left - 6, y + 4);
  }}

  for (let i = 0; i < n; i++) {{
    const x = left + i * groupW + groupW * 0.08;
    const gH = (gptVals[i] / maxVal) * plotH;
    const cH = (claudeVals[i] / maxVal) * plotH;
    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(x, top + plotH - gH, barW, gH);
    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(x + barW + 1, top + plotH - cH, barW, cH);

    if (i % 2 === 0 || n < 20) {{
      ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
      ctx.fillText(bins[i], x + barW, top + plotH + 14);
    }}
  }}
  ctx.fillStyle = MUTED; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('steps per trajectory', w / 2, h - 6);
}})();

// Color interpolation helper: blend between bg (#0f151d) and target color
// t=0 → background, t=1 → full color
function lerpColor(hex, t) {{
  const bg = [15, 21, 29]; // #0f151d
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  const mr = Math.round(bg[0] + (r - bg[0]) * t);
  const mg = Math.round(bg[1] + (g - bg[1]) * t);
  const mb = Math.round(bg[2] + (b - bg[2]) * t);
  return `rgb(${{mr}},${{mg}},${{mb}})`;
}}

// Render a heatmap. normalize='row' or 'col'.
function drawHeatmap(containerId, model, normalize) {{
  const el = document.getElementById(containerId);
  const letters = ['R','S','P','E','V','G','H'];
  const bins = 20;

  // Renormalize: compute per-bin sum across shown letters
  const binSums = new Array(bins).fill(0);
  for (const l of letters) {{
    const v = D.avg_phase[model][l];
    for (let b = 0; b < bins; b++) binSums[b] += v[b];
  }}

  // Compute renormalized values (proportion of shown letters only)
  const renormed = {{}};
  for (const l of letters) {{
    const raw = D.avg_phase[model][l];
    renormed[l] = raw.map((v, b) => binSums[b] > 0 ? v / binSums[b] : 0);
  }}

  // Compute max per row or per column for scaling
  const maxPerRow = {{}};
  const maxPerCol = new Array(bins).fill(0);
  for (const l of letters) {{
    maxPerRow[l] = Math.max(...renormed[l]);
    for (let b = 0; b < bins; b++) {{
      if (renormed[l][b] > maxPerCol[b]) maxPerCol[b] = renormed[l][b];
    }}
  }}

  let html = '<div class="heatmap-grid" style="grid-template-columns: 80px repeat(20, 1fr)">';
  html += '<div></div>';
  for (let b = 0; b < bins; b++) {{
    html += `<div class="heatmap-header">${{b % 5 === 0 ? (b*5)+'%' : ''}}</div>`;
  }}

  for (const letter of letters) {{
    const name = LETTER_TO_NAME[letter] || letter;
    const color = LETTER_COLORS[letter];
    html += `<div class="heatmap-label">${{name}}</div>`;
    const vals = renormed[letter];
    for (let b = 0; b < bins; b++) {{
      const maxV = normalize === 'col' ? (maxPerCol[b] || 0.001) : (maxPerRow[letter] || 0.001);
      const ratio = vals[b] / maxV;
      // Use sqrt for better spread, but map 0 → 0 exactly
      const t = ratio > 0 ? 0.15 + Math.sqrt(ratio) * 0.85 : 0;
      const bg = lerpColor(color, t);
      const textColor = t > 0.4 ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.35)';
      html += `<div class="heatmap-cell" style="background:${{bg}};color:${{textColor}}">${{(vals[b]*100).toFixed(0)}}%</div>`;
    }}
  }}
  html += '</div>';
  el.innerHTML = html;
}}

// 5. Row-normalised
drawHeatmap('heatmapGpt', 'gpt5', 'row');
drawHeatmap('heatmapClaude', 'claude45', 'row');

// 5b. Column-normalised
drawHeatmapCol = drawHeatmap; // reuse
drawHeatmap('heatmapColGpt', 'gpt5', 'col');
drawHeatmap('heatmapColClaude', 'claude45', 'col');

// 6. Stacked area charts
function drawStackedArea(canvasId, model) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 40, right = 10, top = 10, bot = 20;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const bins = 20;
  const letters = ['R','S','P','E','V','G','H'];

  // Build stacked values
  const stacked = [];
  let cumulative = new Array(bins).fill(0);
  for (const letter of letters) {{
    const vals = D.avg_phase[model][letter];
    const layer = vals.map((v, i) => cumulative[i] + v);
    stacked.push({{ letter, bottom: [...cumulative], top: layer }});
    cumulative = layer;
  }}

  // Normalize so each bin sums to ~1
  const maxes = cumulative;

  function xAt(i) {{ return left + (i / (bins - 1)) * plotW; }}
  function yAt(v, binIdx) {{
    const norm = maxes[binIdx] > 0 ? v / maxes[binIdx] : 0;
    return top + plotH - norm * plotH;
  }}

  // Draw from bottom layer up
  for (let s = stacked.length - 1; s >= 0; s--) {{
    const layer = stacked[s];
    ctx.fillStyle = LETTER_COLORS[layer.letter] || '#555';
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    // Top edge left to right
    for (let i = 0; i < bins; i++) {{
      const x = xAt(i), y = yAt(layer.top[i], i);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    // Bottom edge right to left
    for (let i = bins - 1; i >= 0; i--) {{
      ctx.lineTo(xAt(i), yAt(layer.bottom[i], i));
    }}
    ctx.closePath();
    ctx.fill();
  }}
  ctx.globalAlpha = 1;

  // X labels
  ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  for (let i = 0; i < bins; i++) {{
    if (i % 5 === 0) ctx.fillText(i * 5 + '%', xAt(i), top + plotH + 16);
  }}

}}

function fillStackedLegend(legendId) {{
  const letters = ['R','S','P','E','V','G','H'];
  const el = document.getElementById(legendId);
  el.innerHTML = letters.map(l => {{
    const name = LETTER_TO_NAME[l] || l;
    return `<div class="item"><div class="swatch" style="background:${{LETTER_COLORS[l]}}"></div>${{name}}</div>`;
  }}).join('');
}}

drawStackedArea('stackedGpt', 'gpt5');
fillStackedLegend('legendGpt');
drawStackedArea('stackedClaude', 'claude45');
fillStackedLegend('legendClaude');

</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", "-o", default="docs/analytics.html")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    payload = build_payload(data_root)
    html = render_html(payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")
    print(f"gpt5={payload['num_trajs']['gpt5']} claude45={payload['num_trajs']['claude45']}")


if __name__ == "__main__":
    main()
