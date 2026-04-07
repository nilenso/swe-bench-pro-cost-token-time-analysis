"""
Build a self-contained HTML report from extracted trajectory stats.

Pairs instances (both models must have submitted), embeds as JSON,
all filtering/aggregation is client-side JS for instant interactivity.

Usage:
    python build_report.py stats.json -o report.html
"""

import argparse
import json
import sys
from collections import defaultdict

import orjson


def pair_instances(data):
    """Group by instance_id, keep only pairs where both models submitted."""
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

    pairs = []
    skipped_unpaired = 0
    skipped_unsubmitted = 0
    for iid in sorted(by_inst):
        models = by_inst[iid]
        if 'gpt5' not in models or 'claude' not in models:
            skipped_unpaired += 1
            continue
        g, c = models['gpt5'], models['claude']
        if not g.get('submitted') or not c.get('submitted'):
            skipped_unsubmitted += 1
            continue
        pairs.append({
            'instance_id': iid,
            'repo': g['repo'],
            'gpt5': g,
            'claude': c,
        })

    print(f"  Paired: {len(pairs)}, skipped unpaired: {skipped_unpaired}, "
          f"skipped unsubmitted: {skipped_unsubmitted}", file=sys.stderr)
    return pairs


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SWE-Bench Pro: GPT-5 vs Sonnet 4.5</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 15px; background: #191919; color: #d4d4d4; padding: 32px 24px 64px; line-height: 1.7; }
.wrap { max-width: 720px; margin: 0 auto; }
h1 { font-size: 28px; font-weight: 600; color: #f5f5f5; margin-bottom: 6px; letter-spacing: -0.3px; }
h2 { font-size: 16px; color: #999; font-weight: 400; margin-bottom: 32px; line-height: 1.5; }
h3 { font-size: 18px; font-weight: 600; color: #f5f5f5; margin: 48px 0 12px; letter-spacing: -0.2px; }
p { margin: 12px 0; color: #bbb; line-height: 1.7; }
.prose { color: #999; font-size: 15px; line-height: 1.7; margin: 8px 0 20px; }

.filters { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; padding: 14px 16px; background: #222; border: 1px solid #333; border-radius: 8px; align-items: center; }
.filter-group { display: flex; align-items: center; gap: 8px; }
.filter-group label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
select { background: #191919; color: #d4d4d4; border: 1px solid #444; border-radius: 6px; padding: 6px 10px; font-family: inherit; font-size: 14px; }
.count-badge { background: #333; color: #999; padding: 3px 10px; border-radius: 12px; font-size: 12px; }

table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; }
th, td { text-align: right; padding: 6px 14px; border-bottom: 1px solid #2a2a2a; white-space: nowrap; }
td:first-child { white-space: normal; max-width: 340px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px; }
th { color: #666; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
td:first-child, th:first-child { text-align: left; }
tr:hover td { background: #1e1e1e; }
.section { margin-bottom: 48px; }
.ratio { color: #555; font-size: 12px; }
.highlight-g { color: #4ade80; }
.highlight-c { color: #c084fc; }
.muted { color: #555; }
.metric-desc { color: #666; font-size: 12px; font-weight: normal; font-style: normal; display: block; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin-top: 1px; }

.cost-layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; margin-top: 16px; }
.cost-layout > table { flex: 1; min-width: 300px; }
.cost-layout > .fig-box { flex: 1; min-width: 280px; }

.fig { margin: 20px 0; }
.fig-row { display: flex; gap: 24px; flex-wrap: wrap; }
.fig-box { flex: 1; min-width: 280px; overflow: hidden; }
.fig-box h4 { font-size: 12px; color: #777; font-weight: 500; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.3px; }
.fig-caption { color: #666; font-size: 13px; margin-top: 8px; line-height: 1.6; }
.fig-caption b { color: #999; font-weight: 500; }

.per-instance-table { max-height: 600px; overflow-y: auto; margin-top: 12px; border: 1px solid #2a2a2a; border-radius: 8px; }
.per-instance-table table { font-size: 12px; margin: 0; }
.per-instance-table th { position: sticky; top: 0; background: #222; z-index: 1; }
.per-instance-table td.inst-id { max-width: 320px; overflow: hidden; text-overflow: ellipsis; font-size: 11px; }
.sort-btn { cursor: pointer; user-select: none; }
.sort-btn:hover { color: #f5f5f5; }
.sort-btn.active { color: #93c5fd; }
a { color: #93c5fd; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">

<h1>SWE-Bench Pro: GPT-5 vs Sonnet 4.5</h1>
<h2>Cost, tokens, and execution time from the publicly available trajectory data</h2>

<div class="section">
<p>My intuitions about GPT being slow and token hungry were wrong. I had a feel for how these models worked from daily use, but that feel was built from my own tasks and my own repos. Benchmarks exist because vibes can mislead, but most benchmarks report accuracy without reporting how long things take, how many tokens get burned, or how much code gets written along the way.</p>

<p>The <a href="https://labs.scale.com/leaderboard/swe_bench_pro_public">SWE-Bench Pro leaderboard</a> reports resolve rates but not cost, tokens, or time. The raw trajectory data is <a href="https://github.com/scaleapi/SWE-bench_Pro-os">publicly available</a>, and I wanted to check my assumptions against it. These are October 2025 models (GPT-5 and Sonnet 4.5), run under identical conditions on the <a href="https://github.com/SWE-agent/SWE-agent">SWE-Agent</a> scaffold. The coding agent landscape has <a href="https://simonwillison.net/2026/Jan/4/inflection/">changed significantly</a> since then. This data is a snapshot, not the current state of things.</p>

<p>I downloaded all 1,460 trajectory files (~23 GB), extracted per-instance cost, token counts, tool execution time, and action breakdowns, then paired them for direct comparison. Both runs use the same scaffold (SWE-Agent v1.1.0), same tools, same prompt, 250-turn limit, no cost limit. GPT-5 runs with <code>reasoning_effort: high</code>. Sonnet 4.5 uses default settings.</p>
</div>

<p style="color:#999;font-size:14px;margin-bottom:16px;">The overall patterns hold across repos, but the magnitudes vary. Filter by repo to see how.</p>

<div class="filters">
  <div class="filter-group">
    <label>Outcome</label>
    <select id="f-outcome">
      <option value="all">All submitted</option>
      <option value="both">Both resolved</option>
      <option value="gpt5-only">Only GPT-5 resolved</option>
      <option value="claude-only">Only Sonnet 4.5 resolved</option>
      <option value="neither">Neither resolved</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Repo</label>
    <select id="f-repo"><option value="all">All repos</option></select>
  </div>
  <div class="filter-group">
    <span class="count-badge" id="count-badge">0 instances</span>
  </div>
</div>

<div id="report"></div>

</div><!-- .wrap -->

<script>
const DATA = __DATA_PLACEHOLDER__;

const repos = [...new Set(DATA.map(p => p.repo))].sort();
const repoSel = document.getElementById('f-repo');
repos.forEach(r => { const o = document.createElement('option'); o.value = r; o.textContent = r; repoSel.appendChild(o); });

function outcomeOf(p) {
  const gr = p.gpt5.resolved === true;
  const cr = p.claude.resolved === true;
  if (gr && cr) return 'both';
  if (gr) return 'gpt5-only';
  if (cr) return 'claude-only';
  return 'neither';
}

function applyFilters() {
  const fo = document.getElementById('f-outcome').value;
  const fr = document.getElementById('f-repo').value;
  let filtered = DATA;
  if (fo !== 'all') filtered = filtered.filter(p => outcomeOf(p) === fo);
  if (fr !== 'all') filtered = filtered.filter(p => p.repo === fr);
  document.getElementById('count-badge').textContent = filtered.length + ' instances';
  render(filtered);
}

function fmt(n, dec) {
  if (dec === undefined) dec = 0;
  if (n === Infinity || n === -Infinity || isNaN(n)) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function fmtD(n) { return '$' + fmt(n, 2); }
function ratio(a, b) {
  if (!b || b === 0) return '—';
  return (a / b).toFixed(1) + '×';
}
function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function sum(arr) { return arr.reduce((a, b) => a + b, 0); }
function pct(n, d) { return d ? (n / d * 100).toFixed(1) + '%' : '—'; }

function aggRow(label, gVals, cVals, opts) {
  opts = opts || {};
  const fn = opts.fn || mean;
  const dec = opts.dec !== undefined ? opts.dec : (opts.dollar ? 2 : 0);
  const g = fn(gVals);
  const c = fn(cVals);
  const gStr = opts.dollar ? fmtD(g) : fmt(g, dec);
  const cStr = opts.dollar ? fmtD(c) : fmt(c, dec);
  const r = (g > 0 && c > 0) ? ratio(c, g) : '—';
  const desc = opts.desc ? `<span class="metric-desc">${opts.desc}</span>` : '';
  return `<tr><td>${label}${desc}</td><td class="highlight-g">${gStr}</td><td class="highlight-c">${cStr}</td><td class="ratio">${r}</td></tr>`;
}

// ── Histogram (SVG, log-scale x-axis) ──────────────────────────────
function histogram(gVals, cVals, opts) {
  opts = opts || {};
  const title = opts.title || '';
  const fmtTick = opts.fmtTick || (v => v.toFixed(0));
  const W = opts.width || 400, H = opts.height || 200;
  const PAD = {l: 55, r: 10, t: 5, b: 28};
  const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
  const nBins = opts.bins || 20;

  const all = gVals.concat(cVals).filter(v => isFinite(v) && v > 0);
  if (!all.length) return `<div class="fig-box"><h4>${title}</h4><span class="muted">No data</span></div>`;

  const logAll = all.map(v => Math.log10(v));
  const logLo = Math.min(...logAll), logHi = Math.max(...logAll);
  const logRange = logHi - logLo || 1;
  const logBinW = logRange / nBins;

  function toBins(vals) {
    const bins = new Array(nBins).fill(0);
    vals.forEach(v => {
      if (v <= 0) return;
      let i = Math.floor((Math.log10(v) - logLo) / logBinW);
      if (i >= nBins) i = nBins - 1;
      if (i < 0) i = 0;
      bins[i]++;
    });
    return bins;
  }

  const gBins = toBins(gVals);
  const cBins = toBins(cVals);
  const maxCount = Math.max(...gBins, ...cBins, 1);
  const barW = pw / nBins;
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="display:block;width:100%;height:auto">`;

  const yTicks = [0, Math.round(maxCount / 2), maxCount];
  yTicks.forEach(t => {
    const y = PAD.t + ph - (t / maxCount) * ph;
    svg += `<line x1="${PAD.l}" y1="${y}" x2="${PAD.l + pw}" y2="${y}" stroke="#2a2a2a" stroke-width="1"/>`;
    svg += `<text x="${PAD.l - 6}" y="${y + 4}" text-anchor="end" fill="#555" font-size="10" font-family="-apple-system, sans-serif">${t}</text>`;
  });

  for (let i = 0; i < nBins; i++) {
    const x = PAD.l + i * barW;
    const gh = (gBins[i] / maxCount) * ph;
    const ch = (cBins[i] / maxCount) * ph;
    if (gBins[i] > 0)
      svg += `<rect x="${x}" y="${PAD.t + ph - gh}" width="${barW - 1}" height="${gh}" fill="#4ade80" opacity="0.45"/>`;
    if (cBins[i] > 0)
      svg += `<rect x="${x}" y="${PAD.t + ph - ch}" width="${barW - 1}" height="${ch}" fill="#c084fc" opacity="0.45"/>`;
  }

  const xTickCount = Math.min(5, nBins);
  for (let i = 0; i <= xTickCount; i++) {
    const logVal = logLo + (logRange * i / xTickCount);
    const val = Math.pow(10, logVal);
    const x = PAD.l + (i / xTickCount) * pw;
    svg += `<text x="${x}" y="${H - 4}" text-anchor="middle" fill="#555" font-size="10" font-family="-apple-system, sans-serif">${fmtTick(val)}</text>`;
  }

  svg += `<line x1="${PAD.l}" y1="${PAD.t + ph}" x2="${PAD.l + pw}" y2="${PAD.t + ph}" stroke="#333" stroke-width="1"/>`;
  svg += `<rect x="${PAD.l + pw - 115}" y="${PAD.t}" width="10" height="10" fill="#4ade80" opacity="0.6"/>`;
  svg += `<text x="${PAD.l + pw - 102}" y="${PAD.t + 9}" fill="#888" font-size="9" font-family="-apple-system, sans-serif">GPT-5</text>`;
  svg += `<rect x="${PAD.l + pw - 60}" y="${PAD.t}" width="10" height="10" fill="#c084fc" opacity="0.6"/>`;
  svg += `<text x="${PAD.l + pw - 47}" y="${PAD.t + 9}" fill="#888" font-size="9" font-family="-apple-system, sans-serif">S4.5</text>`;
  svg += '</svg>';
  return `<div class="fig-box"><h4>${title}</h4>${svg}</div>`;
}

let sortCol = 'gpt5_cost';
let sortDir = 1;
let currentFiltered = [];

function render(filtered) {
  currentFiltered = filtered;
  if (!filtered.length) {
    document.getElementById('report').innerHTML = '<p class="muted">No instances match filters.</p>';
    return;
  }

  const g = filtered.map(p => p.gpt5);
  const c = filtered.map(p => p.claude);
  const n = filtered.length;

  const nBoth = filtered.filter(p => outcomeOf(p) === 'both').length;
  const nGOnly = filtered.filter(p => outcomeOf(p) === 'gpt5-only').length;
  const nCOnly = filtered.filter(p => outcomeOf(p) === 'claude-only').length;
  const nNeither = filtered.filter(p => outcomeOf(p) === 'neither').length;
  const gResolved = nBoth + nGOnly;
  const cResolved = nBoth + nCOnly;
  const repoCounts = {};
  filtered.forEach(p => { repoCounts[p.repo] = (repoCounts[p.repo] || 0) + 1; });
  const hdr = '<tr><th style="text-align:left"></th><th>GPT-5</th><th>Sonnet 4.5</th><th>S/G</th></tr>';

  let html = '';

  // ── OUTCOME ──────────────────────────────────────────────────────
  html += '<div class="section"><h3>Outcome</h3>';
  html += `<p class="prose">Resolve rates are close enough (GPT-5 ${pct(gResolved,n)}, Sonnet 4.5 ${pct(cResolved,n)}) that I wouldn't pick between them on accuracy alone. The more interesting differences are in how they spend resources to get there.</p>`;
  html += `<p class="prose">${n} instances where both models submitted a patch on the same task. Both solved ${nBoth}, only GPT-5 solved ${nGOnly}, only Sonnet 4.5 solved ${nCOnly}, neither solved ${nNeither}.</p>`;
  html += '</div>';

  // ── Shared data arrays ───────────────────────────────────────────
  const gCost = g.map(s => s.model_stats.instance_cost);
  const cCost = c.map(s => s.model_stats.instance_cost);
  const gIn = g.map(s => s.model_stats.tokens_sent);
  const cIn = c.map(s => s.model_stats.tokens_sent);
  const gOut = g.map(s => s.output_tokens.total);
  const cOut = c.map(s => s.output_tokens.total);
  const gCalls = g.map(s => s.model_stats.api_calls);
  const cCalls = c.map(s => s.model_stats.api_calls);
  const gTpc = g.map(s => s.tokens_per_call);
  const cTpc = c.map(s => s.tokens_per_call);
  const gOutC = g.map(s => s.output_tokens.content);
  const cOutC = c.map(s => s.output_tokens.content);
  const gOutT = g.map(s => s.output_tokens.tool_call_args);
  const cOutT = c.map(s => s.output_tokens.tool_call_args);
  const gTT = g.map(s => s.tool_time.total_seconds);
  const cTT = c.map(s => s.tool_time.total_seconds);
  const gMS = g.map(s => s.tool_time.max_seconds);
  const cMS = c.map(s => s.tool_time.max_seconds);
  const gO10 = sum(g.map(s => s.tool_time.steps_over_10s));
  const cO10 = sum(c.map(s => s.tool_time.steps_over_10s));
  const gSt = g.map(s => s.steps);
  const cSt = c.map(s => s.steps);
  const gPat = g.map(s => s.patch_chars);
  const cPat = c.map(s => s.patch_chars);
  const gPatTok = g.map(s => s.patch_tokens);
  const cPatTok = c.map(s => s.patch_tokens);

  const gCPR = gResolved ? sum(gCost) / gResolved : Infinity;
  const cCPR = cResolved ? sum(cCost) / cResolved : Infinity;
  const gMoreExp = filtered.filter((p, i) => gCost[i] > cCost[i]).length;

  // ── COST ─────────────────────────────────────────────────────────
  html += '<div class="section"><h3>Cost</h3>';
  const costRat = mean(cCost) / mean(gCost);
  html += `<p class="prose">GPT-5 is ${fmt(costRat,1)}× cheaper per task on this benchmark. Even considering caveats, the gap is large enough to matter in practice.</p>`;
  html += `<p class="prose">Each task has an API cost reported by litellm: the total bill for input tokens, output tokens, and any hidden reasoning. These costs reflect Scale AI's internal litellm proxy pricing, not public list prices. At list prices the ratio would differ because GPT-5 sends more input tokens at a lower per-token rate.</p>`;
  html += `<p class="prose">Your own costs will vary. Subscription tiers, prompt caching, Anthropic's caching cost policies, and volume discounts all shift the numbers. This report shows what the benchmark recorded. The token volumes and ratios between models are more stable than the dollar amounts.</p>`;

  html += '<div class="fig"><div class="fig-row">';
  html += histogram(gCost, cCost, {title: 'Cost per instance ($)', fmtTick: v => '$' + v.toFixed(0)});
  html += '</div></div>';
  html += '<table>' + hdr;
  html += aggRow('Mean', gCost, cCost, {dollar: true, desc: 'Average API cost per task.'});
  html += aggRow('Median', gCost, cCost, {fn: median, dollar: true, desc: 'Middle value, less affected by outliers.'});
  html += aggRow('Total', gCost, cCost, {fn: sum, dollar: true});
  html += `<tr><td>Cost per resolve<span class="metric-desc">Total spend ÷ tasks fixed.</span></td><td class="highlight-g">${fmtD(gCPR)}</td><td class="highlight-c">${fmtD(cCPR)}</td><td class="ratio">${ratio(cCPR, gCPR)}</td></tr>`;
  html += `<tr><td>GPT-5 more expensive</td><td colspan="3" class="muted">${gMoreExp}/${n} instances</td></tr>`;
  html += '</table>';

  html += '</div>';

  // ── TOKENS ───────────────────────────────────────────────────────
  html += '<div class="section"><h3>Tokens</h3>';

  const inRat = mean(gIn) / mean(cIn);
  const outRat = mean(cOut) / mean(gOut);

  // Summary
  html += `<p class="prose">Sonnet 4.5 produces ${fmt(outRat,1)}× more output tokens per task, mostly creating files it never submits.</p>`;

  // Histograms
  html += '<div class="fig"><div class="fig-row">';
  html += histogram(gIn, cIn, {title: 'Input tokens per instance', fmtTick: v => (v/1e6).toFixed(1) + 'M'});
  html += histogram(gOut, cOut, {title: 'Output tokens per instance', fmtTick: v => (v/1e3).toFixed(0) + 'K'});
  html += '</div>';
  html += `<p class="fig-caption"><b>Token distributions per instance.</b> Left: total input tokens sent across all API calls. Conversation history accumulates each turn, so this grows with each step. Right: visible output tokens (response text plus tool call arguments). X-axis is log-scaled.</p>`;
  html += '</div>';

  // Key contrast
  html += `<p class="prose">Input tokens are ${inRat > 1.05 ? 'higher for GPT-5 (' + fmt(inRat,1) + '×)' : inRat < 0.95 ? 'higher for Sonnet 4.5 (' + fmt(1/inRat,1) + '×)' : 'similar between the two models'}, across ${fmt(mean(gCalls),0)} vs ${fmt(mean(cCalls),0)} API calls. The divergence is in output. Sonnet 4.5 produces ${fmt(outRat,1)}× more visible output tokens, primarily in tool call arguments (file edits, shell commands).</p>`;
  html += `<p class="prose">Most of that gap comes from file creation. Sonnet 4.5 creates ${fmt(mean(c.map(s=>s.actions.create)),1)} new files per task to GPT-5's ${fmt(mean(g.map(s=>s.actions.create)),1)}. Many are throwaway test and reproduction scripts that never appear in the final patch. The submitted patches are actually close: ${fmt(mean(gPatTok),0)} tokens (GPT-5) vs ${fmt(mean(cPatTok),0)} tokens (Sonnet 4.5), a ${fmt(mean(cPatTok)/mean(gPatTok),1)}× ratio.</p>`;

  // Patch histogram
  html += '<div class="fig"><div class="fig-row">';
  html += histogram(gPatTok, cPatTok, {title: 'Patch size (tokens)', fmtTick: v => (v/1e3).toFixed(1) + 'K'});
  html += '</div>';
  html += `<p class="fig-caption"><b>Patch size distribution.</b> Tiktoken count of the final submitted diff. Larger patches are not necessarily better and may include unnecessary changes.</p>`;
  html += '</div>';

  // Table
  html += '<table>' + hdr;
  html += aggRow('Input tokens (mean)', gIn, cIn, {desc: 'Total tokens sent across all API calls.'});
  html += aggRow('Output tokens (mean)', gOut, cOut, {desc: 'Response text + tool call arguments, via tiktoken.'});
  html += aggRow('&nbsp;&nbsp;content', gOutC, cOutC, {desc: 'Text portion of the response.'});
  html += aggRow('&nbsp;&nbsp;tool_call args', gOutT, cOutT, {desc: 'Arguments passed to tools: file edits, shell commands, search queries.'});
  html += aggRow('Patch size (tokens)', gPatTok, cPatTok, {desc: 'Tiktoken count of the final submitted diff.'});
  html += '</table>';

  // Methodology (end of section, for those who want it)
  html += `<p class="fig-caption"><b>Methodology.</b> Output token counts are measured via tiktoken (cl100k_base) across <code>message.content</code>, <code>message.thought</code>, and <code>tool_calls[].function.arguments</code>. SWE-Agent's built-in <code>tokens_received</code> field has a <a href="https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/agent/models.py#L761">bug</a> that only counts <code>message.content</code>, undercounting output tokens by 7-8x. GPT-5's hidden chain-of-thought reasoning tokens are billed but never appear in the response, so they are not included.</p>`;

  html += '</div>';

  // ── EXECUTION ────────────────────────────────────────────────────
  html += '<div class="section"><h3>Execution</h3>';

  const stRat = mean(cSt) / mean(gSt);

  // Summary
  html += `<p class="prose">Sonnet 4.5 takes more steps and more API calls. Tool execution time is comparable. What differs is how each model spends its turns.</p>`;

  // Histograms
  html += '<div class="fig"><div class="fig-row">';
  html += histogram(gTT, cTT, {title: 'Tool time per instance (s)', fmtTick: v => v.toFixed(0) + 's'});
  html += histogram(gSt, cSt, {title: 'Steps per instance', fmtTick: v => v.toFixed(0)});
  html += '</div>';
  html += `<p class="fig-caption"><b>Execution distributions per instance.</b> Left: total seconds spent running tools (shell commands, file reads, test suites). Right: number of agent steps, where each step is one model turn followed by one tool execution. Wall-clock time and model inference latency are not recorded in this dataset. X-axis is log-scaled.</p>`;
  html += '</div>';

  // Prose
  html += `<p class="prose">Sonnet 4.5 takes ${fmt(stRat,1)}× more steps per task (${fmt(mean(cSt),0)} vs ${fmt(mean(gSt),0)}), making ${fmt(mean(cCalls),0)} API calls to GPT-5's ${fmt(mean(gCalls),0)}. GPT-5 carries more context per call: ${fmt(mean(gTpc),0)} tokens/call vs ${fmt(mean(cTpc),0)}. Different strategies, similar total tool time (${fmt(mean(gTT),0)}s vs ${fmt(mean(cTT),0)}s).</p>`;

  // Actions
  html += `<p class="prose">The action breakdown shows where those steps go.</p>`;
  const aKeys = ['bash', 'view', 'edit', 'create', 'search_find', 'submit', 'other'];
  const aDescs = {
    bash: 'Shell commands: running tests, installing deps, checking output.',
    view: 'Reading file contents.',
    edit: 'Modifying existing files.',
    create: 'Creating new files from scratch.',
    search_find: 'find/grep to locate relevant files.',
    submit: 'Final patch submission.',
    other: 'Unclassified actions.',
  };
  html += '<table>' + hdr;
  aKeys.forEach(k => {
    html += aggRow(k, g.map(s => s.actions[k]), c.map(s => s.actions[k]), {dec: 1, desc: aDescs[k]});
  });
  html += '</table>';

  // Reference table
  html += '<table>' + hdr;
  html += aggRow('Steps (mean)', gSt, cSt, {desc: 'Total model turns per task.'});
  html += aggRow('API calls (mean)', gCalls, cCalls, {desc: 'Number of model roundtrips per task.'});
  html += aggRow('Tokens/call (mean)', gTpc, cTpc, {desc: 'Average context size per call.'});
  html += aggRow('Tool time, mean (s)', gTT, cTT, {dec: 1, desc: 'Total seconds waiting for tools per task.'});
  html += aggRow('Tool time, median (s)', gTT, cTT, {fn: median, dec: 1, desc: 'Less skewed by tasks with long test suites.'});
  html += '</table>';
  html += '</div>';

  // ── REPOS ────────────────────────────────────────────────────────
  html += '<div class="section"><h3>Repos</h3><table>';
  html += '<tr><th style="text-align:left">Repo</th><th>Instances</th><th>GPT-5 resolved</th><th>Sonnet 4.5 resolved</th></tr>';
  Object.keys(repoCounts).sort().forEach(repo => {
    const rp = filtered.filter(p => p.repo === repo);
    const rg = rp.filter(p => p.gpt5.resolved === true).length;
    const rc = rp.filter(p => p.claude.resolved === true).length;
    html += `<tr><td>${repo}</td><td>${repoCounts[repo]}</td><td class="highlight-g">${rg}</td><td class="highlight-c">${rc}</td></tr>`;
  });
  html += '</table></div>';

  // ── PER-INSTANCE TABLE ───────────────────────────────────────────
  const cols = [
    { key: 'instance_id', label: 'Instance', sortFn: (p) => p.instance_id },
    { key: 'repo', label: 'Repo', sortFn: (p) => p.repo },
    { key: 'outcome', label: 'Outcome', sortFn: (p) => outcomeOf(p) },
    { key: 'gpt5_cost', label: '$ GPT-5', sortFn: (p) => p.gpt5.model_stats.instance_cost },
    { key: 'claude_cost', label: '$ Sonnet 4.5', sortFn: (p) => p.claude.model_stats.instance_cost },
    { key: 'gpt5_steps', label: 'Steps GPT-5', sortFn: (p) => p.gpt5.steps },
    { key: 'claude_steps', label: 'Steps S4.5', sortFn: (p) => p.claude.steps },
    { key: 'gpt5_input', label: 'Input GPT-5', sortFn: (p) => p.gpt5.model_stats.tokens_sent },
    { key: 'claude_input', label: 'Input S4.5', sortFn: (p) => p.claude.model_stats.tokens_sent },
    { key: 'gpt5_output', label: 'Out GPT-5', sortFn: (p) => p.gpt5.output_tokens.total },
    { key: 'claude_output', label: 'Out S4.5', sortFn: (p) => p.claude.output_tokens.total },
    { key: 'gpt5_patch', label: 'Patch GPT-5', sortFn: (p) => p.gpt5.patch_chars },
    { key: 'claude_patch', label: 'Patch S4.5', sortFn: (p) => p.claude.patch_chars },
  ];

  const colDef = cols.find(c => c.key === sortCol) || cols[3];
  const sorted = [...filtered].sort((a, b) => {
    const va = colDef.sortFn(a);
    const vb = colDef.sortFn(b);
    if (va < vb) return -1 * sortDir;
    if (va > vb) return 1 * sortDir;
    return 0;
  });

  html += '<div class="section"><h3>Per-Instance</h3><div class="per-instance-table"><table>';
  html += '<tr>';
  cols.forEach(col => {
    const active = sortCol === col.key;
    const arrow = active ? (sortDir === 1 ? ' ▲' : ' ▼') : '';
    html += `<th class="sort-btn${active ? ' active' : ''}" data-col="${col.key}">${col.label}${arrow}</th>`;
  });
  html += '</tr>';

  sorted.forEach(p => {
    const oc = outcomeOf(p);
    const ocLabel = oc === 'both' ? '✓✓' : oc === 'gpt5-only' ? '✓✗' : oc === 'claude-only' ? '✗✓' : '✗✗';
    const shortId = p.instance_id.replace('instance_', '');
    html += '<tr>';
    html += `<td class="inst-id" title="${shortId}">${shortId}</td>`;
    html += `<td>${p.repo}</td>`;
    html += `<td>${ocLabel}</td>`;
    html += `<td class="highlight-g">${fmtD(p.gpt5.model_stats.instance_cost)}</td>`;
    html += `<td class="highlight-c">${fmtD(p.claude.model_stats.instance_cost)}</td>`;
    html += `<td class="highlight-g">${p.gpt5.steps}</td>`;
    html += `<td class="highlight-c">${p.claude.steps}</td>`;
    html += `<td class="highlight-g">${fmt(p.gpt5.model_stats.tokens_sent)}</td>`;
    html += `<td class="highlight-c">${fmt(p.claude.model_stats.tokens_sent)}</td>`;
    html += `<td class="highlight-g">${fmt(p.gpt5.output_tokens.total)}</td>`;
    html += `<td class="highlight-c">${fmt(p.claude.output_tokens.total)}</td>`;
    html += `<td class="highlight-g">${fmt(p.gpt5.patch_chars)}</td>`;
    html += `<td class="highlight-c">${fmt(p.claude.patch_chars)}</td>`;
    html += '</tr>';
  });
  html += '</table></div></div>';

  document.getElementById('report').innerHTML = html;

  document.querySelectorAll('.sort-btn').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) { sortDir *= -1; }
      else { sortCol = col; sortDir = 1; }
      render(currentFiltered);
    });
  });
}

document.getElementById('f-outcome').addEventListener('change', applyFilters);
document.getElementById('f-repo').addEventListener('change', applyFilters);
applyFilters();
</script>
<p style="max-width:720px;margin:32px auto 0;color:#555;font-size:13px;">See also: <a href="unsubmitted.html">Unsubmitted tasks report</a>, covering instances where one or both models failed to submit a patch.</p>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", help="JSON stats file from extract_stats_fast.py")
    parser.add_argument("-o", "--output", default="report.html")
    args = parser.parse_args()

    with open(args.stats_file, 'rb') as f:
        data = orjson.loads(f.read())

    pairs = pair_instances(data)

    slim_pairs = []
    keep_keys = [
        'instance_id', 'repo', 'resolved', 'submitted', 'steps', 'patch_chars', 'patch_tokens',
        'model_stats', 'tokens_per_call', 'output_tokens', 'tool_time',
        'actions', 'content',
    ]
    for p in pairs:
        sp = {'instance_id': p['instance_id'], 'repo': p['repo']}
        for model in ('gpt5', 'claude'):
            sp[model] = {k: p[model][k] for k in keep_keys if k in p[model]}
        slim_pairs.append(sp)

    json_data = orjson.dumps(slim_pairs).decode('utf-8')
    html = HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', json_data)

    with open(args.output, 'w') as f:
        f.write(html)

    print(f"  Wrote {args.output} ({len(html) / 1024:.0f} KB, {len(slim_pairs)} paired instances)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
