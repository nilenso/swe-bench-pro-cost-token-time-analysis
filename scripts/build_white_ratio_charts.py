"""Build a white-background editorial page with per-task ratio charts.

Creates 5 main ratio-distribution histograms
(cost, input/output/total tokens, time)
and one compact summary dot-whisker chart.

Usage:
    python scripts/build_white_ratio_charts.py stats.json -o docs/ratio-charts-white.html
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def pair_instances(data: list[dict]) -> list[dict]:
    by_inst: dict[str, dict] = defaultdict(dict)
    for s in data:
        name = s.get("config", {}).get("model_name", "").lower()
        if "gpt" in name:
            by_inst[s["instance_id"]]["gpt5"] = s
        elif "claude" in name:
            by_inst[s["instance_id"]]["sonnet"] = s

    pairs: list[dict] = []
    for iid, models in by_inst.items():
        if "gpt5" not in models or "sonnet" not in models:
            continue
        g = models["gpt5"]
        s = models["sonnet"]
        if not g.get("submitted") or not s.get("submitted"):
            continue
        pairs.append({"instance_id": iid, "gpt5": g, "sonnet": s})
    return pairs


def extract_ratios(pairs: list[dict], g_fn, s_fn):
    ratios = []
    excluded = 0
    for p in pairs:
        gv = g_fn(p["gpt5"])
        sv = s_fn(p["sonnet"])
        if gv is None or sv is None or gv <= 0 or sv <= 0:
            excluded += 1
            continue
        ratios.append(sv / gv)
    return ratios, excluded


def fmtx(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}×"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats_file")
    ap.add_argument("-o", "--output", default="docs/ratio-charts-white.html")
    args = ap.parse_args()

    data = json.loads(Path(args.stats_file).read_text())
    pairs = pair_instances(data)
    n_total = len(pairs)

    metrics = {
        "cost": {
            "title": "Per-task cost ratio",
            "subtitle": "",
            "ratio_def": "Ratio = Sonnet 4.5 / GPT-5; 1× = equal",
            "x_label": "Cost ratio",
            "left_label": "Sonnet cheaper",
            "right_label": "Sonnet more expensive",
            "takeaway_phrase": "Sonnet was more expensive on",
            "median_label": "Median {x}",
            "g_fn": lambda s: s["model_stats"]["instance_cost"],
            "s_fn": lambda s: s["model_stats"]["instance_cost"],
        },
        "tokens_input": {
            "title": "Per-task input token ratio",
            "subtitle": "",
            "ratio_def": "Ratio = Sonnet 4.5 / GPT-5; 1× = equal",
            "x_label": "Input token ratio",
            "left_label": "Sonnet fewer input tokens",
            "right_label": "Sonnet more input tokens",
            "takeaway_phrase": "Sonnet used more input tokens on",
            "median_label": "Median {x}",
            "g_fn": lambda s: s["model_stats"]["tokens_sent"],
            "s_fn": lambda s: s["model_stats"]["tokens_sent"],
        },
        "tokens_output": {
            "title": "Per-task output token ratio",
            "subtitle": "",
            "ratio_def": "Ratio = Sonnet 4.5 / GPT-5; 1× = equal",
            "x_label": "Output token ratio",
            "left_label": "Sonnet fewer output tokens",
            "right_label": "Sonnet more output tokens",
            "takeaway_phrase": "Sonnet used more output tokens on",
            "median_label": "Median {x}",
            "g_fn": lambda s: s["model_stats"]["tokens_received"],
            "s_fn": lambda s: s["model_stats"]["tokens_received"],
        },
        "tokens_total": {
            "title": "Per-task total token ratio",
            "subtitle": "",
            "ratio_def": "Ratio = Sonnet 4.5 / GPT-5; 1× = equal",
            "x_label": "Total token ratio",
            "left_label": "Sonnet fewer total tokens",
            "right_label": "Sonnet more total tokens",
            "takeaway_phrase": "Sonnet used more total tokens on",
            "median_label": "Median {x}",
            "g_fn": lambda s: s["model_stats"]["tokens_sent"] + s["model_stats"]["tokens_received"],
            "s_fn": lambda s: s["model_stats"]["tokens_sent"] + s["model_stats"]["tokens_received"],
        },
        "time": {
            "title": "Per-task time ratio",
            "subtitle": "",
            "ratio_def": "Ratio = Sonnet 4.5 / GPT-5; 1× = equal",
            "x_label": "Time ratio",
            "left_label": "Sonnet faster",
            "right_label": "Sonnet slower",
            "takeaway_phrase": "Sonnet was slower on",
            "median_label": "Median {x}",
            # Tool execution time available from trajectories.
            "g_fn": lambda s: s["tool_time"]["total_seconds"],
            "s_fn": lambda s: s["tool_time"]["total_seconds"],
        },
    }

    payload = {"n_total": n_total, "metrics": {}}
    footnotes = []

    for key, m in metrics.items():
        ratios, excluded = extract_ratios(pairs, m["g_fn"], m["s_fn"])
        n = len(ratios)
        med = quantile(ratios, 0.5)
        q1 = quantile(ratios, 0.25)
        q3 = quantile(ratios, 0.75)
        lower = sum(1 for r in ratios if r < 1.0)
        lower_pct = (100.0 * lower / n) if n else 0.0
        higher = sum(1 for r in ratios if r > 1.0)
        higher_pct = (100.0 * higher / n) if n else 0.0

        stat_line = (
            f"IQR {fmtx(q1)}–{fmtx(q3)}; "
            f"{m['takeaway_phrase']} {higher} of {n} tasks ({higher_pct:.1f}%)."
        )
        if excluded:
            stat_line += f" ({excluded} tasks excluded due to non-positive values.)"

        payload["metrics"][key] = {
            "title": m["title"],
            "subtitle": m["subtitle"],
            "xLabel": m["x_label"],
            "yLabel": "Tasks",
            "ratioDef": m["ratio_def"],
            "leftLabel": m["left_label"],
            "rightLabel": m["right_label"],
            "medianLabel": m["median_label"].format(x=fmtx(med)),
            "statLine": stat_line,
            "ratios": ratios,
            "median": med,
            "q1": q1,
            "q3": q3,
            "lowerCount": lower,
            "lowerPct": lower_pct,
            "higherCount": higher,
            "higherPct": higher_pct,
            "included": n,
            "excluded": excluded,
        }

        if excluded:
            footnotes.append(
                f"{m['title']}: excluded {excluded} tasks with non-positive values before ratio computation."
            )

    payload["summary"] = {
        "title": "Typical per-task difference across metrics",
        "subtitle": "Sonnet 4.5 / GPT-5; 1× = equal; left of 1× means Sonnet is lower, right means Sonnet is higher.",
        "xLabel": "Ratio",
        "yLabel": "Metric",
        "rows": [
            {
                "name": "Cost",
                "median": payload["metrics"]["cost"]["median"],
                "q1": payload["metrics"]["cost"]["q1"],
                "q3": payload["metrics"]["cost"]["q3"],
            },
            {
                "name": "Input tokens",
                "median": payload["metrics"]["tokens_input"]["median"],
                "q1": payload["metrics"]["tokens_input"]["q1"],
                "q3": payload["metrics"]["tokens_input"]["q3"],
            },
            {
                "name": "Time",
                "median": payload["metrics"]["time"]["median"],
                "q1": payload["metrics"]["time"]["q1"],
                "q3": payload["metrics"]["time"]["q3"],
            },
        ],
        "note": "For cost and tokens, lower is better. For time, lower means faster.",
    }
    payload["footnotes"] = footnotes

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Per-task ratio charts</title>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #111111;
      --muted: #555555;
      --axis: #e6e6e6;
      --grid: #f2f2f2;
      --bar: #202020;
      --line: #111111;
      --line-soft: #bbbbbb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 960px;
      margin: 28px auto 48px;
      padding: 0 20px;
    }}
    .chart {{
      margin: 34px 0 48px;
    }}
    .title {{
      font-size: 24px;
      font-weight: 500;
      margin: 0;
      letter-spacing: -0.01em;
    }}
    .subtitle {{
      margin: 4px 0 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .stat {{
      margin-top: 8px;
      color: #222;
      font-size: 14px;
    }}
    .note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .footnote {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .summary-title {{ margin-top: 6px; }}
  </style>
</head>
<body>
  <div class=\"wrap\" id=\"app\"></div>

  <script>
  const PAYLOAD = {json.dumps(payload)};

  function fmtRatio(v) {{
    if (!isFinite(v)) return '—';
    if (v >= 10) return v.toFixed(1) + '×';
    if (v >= 1) return v.toFixed(2) + '×';
    if (v >= 0.1) return v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '') + '×';
    return v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') + '×';
  }}

  function niceMax(n) {{
    if (n <= 5) return 5;
    const p = Math.pow(10, Math.floor(Math.log10(n)));
    const m = n / p;
    if (m <= 1) return 1 * p;
    if (m <= 2) return 2 * p;
    if (m <= 5) return 5 * p;
    return 10 * p;
  }}

  function ratioTicks() {{
    // Fixed ratio axis for all charts: 0.063× to 32×
    return [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
  }}

  function renderHistogram(metric) {{
    const ratios = metric.ratios;
    const logs = ratios.map(r => Math.log2(r));
    // Fixed axis across all histograms: [0.063×, 32×]
    const lo = -4, hi = 5;

    const binsN = 34;
    const binW = (hi - lo) / binsN;
    const bins = new Array(binsN).fill(0);
    logs.forEach(v => {{
      let i = Math.floor((v - lo) / binW);
      if (i < 0) i = 0;
      if (i >= binsN) i = binsN - 1;
      bins[i] += 1;
    }});

    const W = 900, H = 320;
    const m = {{l: 62, r: 18, t: 26, b: 58}};
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;

    const yMax = niceMax(Math.max(...bins));
    const x = (v) => m.l + ((v - lo) / (hi - lo)) * pw;
    const y = (v) => m.t + ph - (v / yMax) * ph;

    const yTicks = [0, yMax / 2, yMax];
    const tickKs = ratioTicks();

    let svg = '';

    // Very light horizontal gridlines
    yTicks.forEach(t => {{
      svg += `<line x1="${{m.l}}" y1="${{y(t)}}" x2="${{m.l + pw}}" y2="${{y(t)}}" stroke="var(--grid)" stroke-width="1" />`;
    }});

    // Bars
    for (let i = 0; i < binsN; i++) {{
      const bx0 = m.l + (i / binsN) * pw;
      const bx1 = m.l + ((i + 1) / binsN) * pw;
      const bh = (bins[i] / yMax) * ph;
      svg += `<rect x="${{bx0 + 0.3}}" y="${{m.t + ph - bh}}" width="${{Math.max(0, bx1 - bx0 - 0.8)}}" height="${{bh}}" fill="var(--bar)" />`;
    }}

    // 1× reference
    svg += `<line x1="${{x(0)}}" y1="${{m.t}}" x2="${{x(0)}}" y2="${{m.t + ph}}" stroke="var(--line-soft)" stroke-width="1" stroke-dasharray="4,4" />`;

    // Median line
    const medX = x(Math.log2(metric.median));
    svg += `<line x1="${{medX}}" y1="${{m.t}}" x2="${{medX}}" y2="${{m.t + ph}}" stroke="var(--line)" stroke-width="1.4" />`;

    // Left/right direct labels
    svg += `<text x="${{m.l}}" y="14" fill="var(--muted)" font-size="12" text-anchor="start">${{metric.leftLabel}}</text>`;
    svg += `<text x="${{m.l + pw}}" y="14" fill="var(--muted)" font-size="12" text-anchor="end">${{metric.rightLabel}}</text>`;

    // Median label
    const medLabelX = Math.max(m.l + 24, Math.min(m.l + pw - 24, medX));
    svg += `<text x="${{medLabelX}}" y="${{m.t - 8}}" fill="var(--text)" font-size="12" text-anchor="middle">${{metric.medianLabel}}</text>`;

    // Axes
    svg += `<line x1="${{m.l}}" y1="${{m.t + ph}}" x2="${{m.l + pw}}" y2="${{m.t + ph}}" stroke="var(--axis)" stroke-width="1" />`;
    svg += `<line x1="${{m.l}}" y1="${{m.t}}" x2="${{m.l}}" y2="${{m.t + ph}}" stroke="var(--axis)" stroke-width="1" />`;

    // X ticks in ratio labels
    tickKs.forEach(k => {{
      const xv = x(k);
      const rv = Math.pow(2, k);
      const label = (rv >= 1 ? (Number.isInteger(rv) ? rv.toFixed(0) : rv.toFixed(2)) : rv.toFixed(3).replace(/0+$/,'').replace(/\.$/,'')) + '×';
      svg += `<line x1="${{xv}}" y1="${{m.t + ph}}" x2="${{xv}}" y2="${{m.t + ph + 5}}" stroke="var(--axis)" stroke-width="1" />`;
      svg += `<text x="${{xv}}" y="${{H - 28}}" fill="var(--muted)" font-size="11" text-anchor="middle">${{label}}</text>`;
    }});

    // Y ticks
    yTicks.forEach(t => {{
      svg += `<text x="${{m.l - 8}}" y="${{y(t) + 4}}" fill="var(--muted)" font-size="11" text-anchor="end">${{Math.round(t)}}</text>`;
    }});

    // Axis labels
    svg += `<text x="${{m.l + pw / 2}}" y="${{H - 6}}" fill="var(--text)" font-size="12" text-anchor="middle">${{metric.xLabel}}</text>`;
    svg += `<text x="16" y="${{m.t + ph / 2}}" fill="var(--text)" font-size="12" text-anchor="middle" transform="rotate(-90 16 ${{m.t + ph / 2}})">${{metric.yLabel}}</text>`;

    return `
      <section class="chart">
        <h2 class="title">${{metric.title}}</h2>
        <svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{metric.title}}">${{svg}}</svg>
        <p class="note">${{metric.ratioDef}}</p>
        <p class="stat">${{metric.statLine}}</p>
      </section>
    `;
  }}

  function renderSummary(summary) {{
    const rows = summary.rows;

    // Tight summary domain based on whiskers (Q1..Q3), with padding.
    const whiskerLogs = rows.flatMap(r => [Math.log2(r.q1), Math.log2(r.q3)]).filter(v => isFinite(v));
    let lo = Math.min(...whiskerLogs) - 0.6;
    let hi = Math.max(...whiskerLogs) + 0.6;
    // Always include 1× reference and keep some breathing room around it.
    lo = Math.min(lo, -0.5);
    hi = Math.max(hi, 0.5);
    // Snap to half-steps for cleaner ticks.
    lo = Math.floor(lo * 2) / 2;
    hi = Math.ceil(hi * 2) / 2;

    const W = 900, H = 230;
    const m = {{l: 95, r: 18, t: 20, b: 52}};
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;

    const x = (lv) => m.l + ((lv - lo) / (hi - lo)) * pw;
    const rowY = (i) => m.t + ((i + 0.5) / rows.length) * ph;

    const allTickKs = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
    const tickKs = allTickKs.filter(k => k >= lo && k <= hi);

    let svg = '';

    // Vertical ticks/grid
    tickKs.forEach(k => {{
      const xv = x(k);
      svg += `<line x1="${{xv}}" y1="${{m.t}}" x2="${{xv}}" y2="${{m.t + ph}}" stroke="var(--grid)" stroke-width="1" />`;
      const rv = Math.pow(2, k);
      const label = (rv >= 1 ? (Number.isInteger(rv) ? rv.toFixed(0) : rv.toFixed(2)) : rv.toFixed(3).replace(/0+$/,'').replace(/\.$/,'')) + '×';
      svg += `<text x="${{xv}}" y="${{H - 24}}" fill="var(--muted)" font-size="11" text-anchor="middle">${{label}}</text>`;
    }});

    // 1× line
    svg += `<line x1="${{x(0)}}" y1="${{m.t}}" x2="${{x(0)}}" y2="${{m.t + ph}}" stroke="var(--line-soft)" stroke-width="1" stroke-dasharray="4,4" />`;

    // Rows
    rows.forEach((r, i) => {{
      const y = rowY(i);
      const x1 = x(Math.log2(r.q1));
      const x3 = x(Math.log2(r.q3));
      const xm = x(Math.log2(r.median));
      svg += `<line x1="${{x1}}" y1="${{y}}" x2="${{x3}}" y2="${{y}}" stroke="var(--line)" stroke-width="1.6" />`;
      svg += `<circle cx="${{xm}}" cy="${{y}}" r="3.2" fill="var(--line)" />`;
      svg += `<text x="${{m.l - 12}}" y="${{y + 4}}" fill="var(--text)" font-size="12" text-anchor="end">${{r.name}}</text>`;

      // Direct median value label next to each point.
      const medLabel = (r.median >= 10 ? r.median.toFixed(1) : r.median.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')) + '×';
      const rightEdge = m.l + pw - 8;
      const useLeft = xm > rightEdge - 44;
      const tx = useLeft ? (xm - 8) : (xm + 8);
      const anchor = useLeft ? 'end' : 'start';
      svg += `<text x="${{tx}}" y="${{y - 6}}" fill="var(--text)" font-size="11" text-anchor="${{anchor}}">${{medLabel}}</text>`;
    }});

    // Frame axes (minimal)
    svg += `<line x1="${{m.l}}" y1="${{m.t + ph}}" x2="${{m.l + pw}}" y2="${{m.t + ph}}" stroke="var(--axis)" stroke-width="1" />`;

    // Axis label
    svg += `<text x="${{m.l + pw / 2}}" y="${{H - 6}}" fill="var(--text)" font-size="12" text-anchor="middle">${{summary.xLabel}}</text>`;

    return `
      <section class="chart">
        <h2 class="title summary-title">${{summary.title}}</h2>
        <p class="subtitle">${{summary.subtitle}}</p>
        <svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{summary.title}}">${{svg}}</svg>
        <p class="note">${{summary.note}}</p>
      </section>
    `;
  }}

  const app = document.getElementById('app');
  app.innerHTML =
      renderHistogram(PAYLOAD.metrics.cost) +
      renderHistogram(PAYLOAD.metrics.tokens_input) +
      renderHistogram(PAYLOAD.metrics.tokens_output) +
      renderHistogram(PAYLOAD.metrics.tokens_total) +
      renderHistogram(PAYLOAD.metrics.time) +
      renderSummary(PAYLOAD.summary) +
      (PAYLOAD.footnotes.length
        ? `<div>${{PAYLOAD.footnotes.map(f => `<p class=\"footnote\">${{f}}</p>`).join('')}}</div>`
        : '');
  </script>
</body>
</html>
"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out} with {n_total} paired tasks")


if __name__ == "__main__":
    main()
