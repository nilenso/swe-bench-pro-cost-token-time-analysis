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
<title>SWE-Bench Pro: GPT-5 vs Claude Sonnet 4.5</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; font-size: 13px; background: #0d1117; color: #c9d1d9; padding: 24px; line-height: 1.5; }
h1 { font-size: 18px; color: #f0f6fc; margin-bottom: 4px; }
h2 { font-size: 14px; color: #8b949e; font-weight: normal; margin-bottom: 20px; }
h3 { font-size: 14px; color: #f0f6fc; margin: 24px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #21262d; }

.filters { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; padding: 12px; background: #161b22; border: 1px solid #21262d; border-radius: 6px; align-items: center; }
.filter-group { display: flex; align-items: center; gap: 6px; }
.filter-group label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
select { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px 8px; font-family: inherit; font-size: 13px; }
.count-badge { background: #21262d; color: #8b949e; padding: 2px 8px; border-radius: 10px; font-size: 12px; }

table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
th, td { text-align: right; padding: 4px 12px; border-bottom: 1px solid #21262d; white-space: nowrap; }
td:first-child { white-space: normal; max-width: 420px; }
th { color: #8b949e; font-weight: normal; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
td:first-child, th:first-child { text-align: left; }
tr:hover td { background: #161b22; }
.section { margin-bottom: 32px; }
.ratio { color: #8b949e; font-size: 12px; }
.highlight-g { color: #3fb950; }
.highlight-c { color: #a371f7; }
.muted { color: #484f58; }
.per-instance-table { max-height: 600px; overflow-y: auto; }
.per-instance-table table { font-size: 12px; }
.per-instance-table th { position: sticky; top: 0; background: #0d1117; z-index: 1; }
.per-instance-table td.inst-id { max-width: 400px; overflow: hidden; text-overflow: ellipsis; font-size: 11px; }
.sort-btn { cursor: pointer; user-select: none; }
.sort-btn:hover { color: #f0f6fc; }
.sort-btn.active { color: #58a6ff; }
.metric-desc { color: #484f58; font-size: 11px; font-weight: normal; font-style: italic; display: block; }
.hist-row { display: flex; gap: 24px; flex-wrap: wrap; margin: 8px 0 16px; }
.hist-box { flex: 1; min-width: 500px; }
.hist-box h4 { font-size: 12px; color: #8b949e; font-weight: normal; margin-bottom: 4px; }
</style>
</head>
<body>

<h1>SWE-Bench Pro: GPT-5 vs Claude Sonnet 4.5</h1>
<h2>Paired comparison — both models submitted a patch on the same task</h2>

<div class="filters">
  <div class="filter-group">
    <label>Outcome</label>
    <select id="f-outcome">
      <option value="all">All submitted</option>
      <option value="both">Both resolved</option>
      <option value="gpt5-only">Only GPT-5 resolved</option>
      <option value="claude-only">Only Claude Sonnet 4.5 resolved</option>
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

<script>
const DATA = __DATA_PLACEHOLDER__;

// Populate repo filter
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

// ── Histogram renderer (SVG) ──────────────────────────────────────────
function histogram(gVals, cVals, opts) {
  opts = opts || {};
  const title = opts.title || '';
  const fmtTick = opts.fmtTick || (v => v.toFixed(0));
  const W = opts.width || 560, H = opts.height || 240;
  const PAD = {l: 55, r: 10, t: 5, b: 28};
  const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
  const nBins = opts.bins || 20;

  const all = gVals.concat(cVals).filter(v => isFinite(v) && v > 0);
  if (!all.length) return `<div class="hist-box"><h4>${title}</h4><span class="muted">No data</span></div>`;

  // Log-scale bins: equal width in log space
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
  let svg = `<svg width="${W}" height="${H}" style="display:block">`;

  // Y axis ticks
  const yTicks = [0, Math.round(maxCount / 2), maxCount];
  yTicks.forEach(t => {
    const y = PAD.t + ph - (t / maxCount) * ph;
    svg += `<line x1="${PAD.l}" y1="${y}" x2="${PAD.l + pw}" y2="${y}" stroke="#21262d" stroke-width="1"/>`;
    svg += `<text x="${PAD.l - 6}" y="${y + 4}" text-anchor="end" fill="#484f58" font-size="10">${t}</text>`;
  });

  // Bars — GPT-5 behind, Claude in front, both semi-transparent
  for (let i = 0; i < nBins; i++) {
    const x = PAD.l + i * barW;
    const gh = (gBins[i] / maxCount) * ph;
    const ch = (cBins[i] / maxCount) * ph;
    if (gBins[i] > 0)
      svg += `<rect x="${x}" y="${PAD.t + ph - gh}" width="${barW - 1}" height="${gh}" fill="#3fb950" opacity="0.5"/>`;
    if (cBins[i] > 0)
      svg += `<rect x="${x}" y="${PAD.t + ph - ch}" width="${barW - 1}" height="${ch}" fill="#a371f7" opacity="0.5"/>`;
  }

  // X axis ticks (log-spaced)
  const xTickCount = Math.min(5, nBins);
  for (let i = 0; i <= xTickCount; i++) {
    const logVal = logLo + (logRange * i / xTickCount);
    const val = Math.pow(10, logVal);
    const x = PAD.l + (i / xTickCount) * pw;
    svg += `<text x="${x}" y="${H - 4}" text-anchor="middle" fill="#484f58" font-size="10">${fmtTick(val)}</text>`;
  }

  // Baseline
  svg += `<line x1="${PAD.l}" y1="${PAD.t + ph}" x2="${PAD.l + pw}" y2="${PAD.t + ph}" stroke="#30363d" stroke-width="1"/>`;

  // Legend
  svg += `<rect x="${PAD.l + pw - 115}" y="${PAD.t}" width="10" height="10" fill="#3fb950" opacity="0.7"/>`;
  svg += `<text x="${PAD.l + pw - 102}" y="${PAD.t + 9}" fill="#8b949e" font-size="9">GPT-5</text>`;
  svg += `<rect x="${PAD.l + pw - 60}" y="${PAD.t}" width="10" height="10" fill="#a371f7" opacity="0.7"/>`;
  svg += `<text x="${PAD.l + pw - 47}" y="${PAD.t + 9}" fill="#8b949e" font-size="9">CS4.5</text>`;

  svg += '</svg>';
  return `<div class="hist-box"><h4>${title}</h4>${svg}</div>`;
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

let sortCol = 'gpt5_cost';
let sortDir = 1; // 1 = asc, -1 = desc
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

  // Outcome counts
  const nBoth = filtered.filter(p => outcomeOf(p) === 'both').length;
  const nGOnly = filtered.filter(p => outcomeOf(p) === 'gpt5-only').length;
  const nCOnly = filtered.filter(p => outcomeOf(p) === 'claude-only').length;
  const nNeither = filtered.filter(p => outcomeOf(p) === 'neither').length;
  const gResolved = nBoth + nGOnly;
  const cResolved = nBoth + nCOnly;

  // Repo counts
  const repoCounts = {};
  filtered.forEach(p => { repoCounts[p.repo] = (repoCounts[p.repo] || 0) + 1; });

  let html = '';

  // OUTCOME
  html += '<div class="section"><h3>Outcome</h3><table>';
  html += '<tr><th style="text-align:left">Category</th><th>Count</th><th>% of filtered</th></tr>';
  html += `<tr><td>Both resolved</td><td>${nBoth}</td><td>${pct(nBoth, n)}</td></tr>`;
  html += `<tr><td>Only GPT-5 resolved</td><td>${nGOnly}</td><td>${pct(nGOnly, n)}</td></tr>`;
  html += `<tr><td>Only Claude Sonnet 4.5 resolved</td><td>${nCOnly}</td><td>${pct(nCOnly, n)}</td></tr>`;
  html += `<tr><td>Neither resolved</td><td>${nNeither}</td><td>${pct(nNeither, n)}</td></tr>`;
  html += `<tr><td><b>GPT-5 resolve rate</b></td><td><b>${gResolved}/${n}</b></td><td><b>${pct(gResolved, n)}</b></td></tr>`;
  html += `<tr><td><b>Claude Sonnet 4.5 resolve rate</b></td><td><b>${cResolved}/${n}</b></td><td><b>${pct(cResolved, n)}</b></td></tr>`;
  html += '</table></div>';

  // REPOS
  html += '<div class="section"><h3>Repos</h3><table>';
  html += '<tr><th style="text-align:left">Repo</th><th>Instances</th><th>GPT-5 resolved</th><th>Claude Sonnet 4.5 resolved</th></tr>';
  Object.keys(repoCounts).sort().forEach(repo => {
    const rp = filtered.filter(p => p.repo === repo);
    const rg = rp.filter(p => p.gpt5.resolved === true).length;
    const rc = rp.filter(p => p.claude.resolved === true).length;
    html += `<tr><td>${repo}</td><td>${repoCounts[repo]}</td><td class="highlight-g">${rg}</td><td class="highlight-c">${rc}</td></tr>`;
  });
  html += '</table></div>';

  // Helper to extract arrays
  const gCost = g.map(s => s.model_stats.instance_cost);
  const cCost = c.map(s => s.model_stats.instance_cost);
  const hdr = '<tr><th style="text-align:left"></th><th>GPT-5</th><th>Claude Sonnet 4.5</th><th>Ratio C/G</th></tr>';

  // COST
  html += '<div class="section"><h3>Cost</h3><table>' + hdr;
  html += aggRow('Mean', gCost, cCost, {dollar: true, desc: 'Average API cost per task. What you\'d see on your bill for one coding task.'});
  html += aggRow('Median', gCost, cCost, {fn: median, dollar: true, desc: 'Middle value — less affected by expensive outlier tasks than the mean.'});
  html += aggRow('Total', gCost, cCost, {fn: sum, dollar: true, desc: 'Sum across all filtered tasks.'});
  const gCPR = gResolved ? sum(gCost) / gResolved : Infinity;
  const cCPR = cResolved ? sum(cCost) / cResolved : Infinity;
  html += `<tr><td>Cost per resolve<span class="metric-desc">Total spend ÷ tasks actually fixed. The real price of a successful fix.</span></td><td class="highlight-g">${fmtD(gCPR)}</td><td class="highlight-c">${fmtD(cCPR)}</td><td class="ratio">${gCPR > 0 ? ratio(cCPR, gCPR) : '—'}</td></tr>`;
  const gMoreExpensive = filtered.filter((p, i) => gCost[i] > cCost[i]).length;
  html += `<tr><td>GPT-5 more expensive</td><td colspan="3" class="muted">${gMoreExpensive}/${n} instances</td></tr>`;
  html += `<tr><td>Claude Sonnet 4.5 more expensive</td><td colspan="3" class="muted">${n - gMoreExpensive}/${n} instances</td></tr>`;
  html += '</table>';
  html += '<p style="color:#484f58; font-size:11px; font-style:italic; padding:4px 12px;">Costs reflect Scale AI\'s internal litellm pricing, not public list prices.</p>';
  html += '<div class="hist-row">';
  html += histogram(gCost, cCost, {title: 'Cost per instance ($)', fmtTick: v => '$' + v.toFixed(0)});
  html += '</div>';
  html += '</div>';

  // TOKENS
  const gIn = g.map(s => s.model_stats.tokens_sent);
  const cIn = c.map(s => s.model_stats.tokens_sent);
  const gCalls = g.map(s => s.model_stats.api_calls);
  const cCalls = c.map(s => s.model_stats.api_calls);
  const gTpc = g.map(s => s.tokens_per_call);
  const cTpc = c.map(s => s.tokens_per_call);
  const gOut = g.map(s => s.output_tokens.total);
  const cOut = c.map(s => s.output_tokens.total);
  const gOutC = g.map(s => s.output_tokens.content);
  const cOutC = c.map(s => s.output_tokens.content);
  const gOutT = g.map(s => s.output_tokens.tool_call_args);
  const cOutT = c.map(s => s.output_tokens.tool_call_args);

  html += '<div class="section"><h3>Tokens</h3><table>' + hdr;
  html += aggRow('Input tokens (mean)', gIn, cIn, {desc: 'Total tokens sent to the model across all API calls. Grows each step as conversation history accumulates — like context window usage in Claude Code.'});
  html += aggRow('API calls (mean)', gCalls, cCalls, {desc: 'Number of model roundtrips. Each call = one prompt + one response.'});
  html += aggRow('Tokens/call (mean)', gTpc, cTpc, {desc: 'Average context size per call. Higher means the model carries more history each turn.'});
  html += aggRow('Output tokens (mean)', gOut, cOut, {desc: 'What the model wrote back — response text + tool call arguments, counted via tiktoken. GPT-5\'s hidden reasoning tokens are billed but not visible here.'});
  html += aggRow('&nbsp;&nbsp;content', gOutC, cOutC, {desc: 'The text portion of the response — explanations, reasoning shown to user.'});
  html += aggRow('&nbsp;&nbsp;tool_call args', gOutT, cOutT, {desc: 'Arguments passed to tools — file edits, shell commands, search queries. The actual "work" output.'});
  const g1m = sum(gIn) > 0 ? sum(gCost) / sum(gIn) * 1e6 : 0;
  const c1m = sum(cIn) > 0 ? sum(cCost) / sum(cIn) * 1e6 : 0;
  html += `<tr><td>$/1M input tokens<span class="metric-desc">Effective rate per million input tokens. Derived from total cost ÷ total input tokens.</span></td><td class="highlight-g">${fmtD(g1m)}</td><td class="highlight-c">${fmtD(c1m)}</td><td class="ratio">${ratio(c1m, g1m)}</td></tr>`;
  html += '</table>';
  html += '<div class="hist-row">';
  html += histogram(gIn, cIn, {title: 'Input tokens per instance', fmtTick: v => (v/1e6).toFixed(1) + 'M'});
  html += histogram(gOut, cOut, {title: 'Output tokens per instance', fmtTick: v => (v/1e3).toFixed(0) + 'K'});
  html += '</div>';
  html += '</div>';

  // TOOL EXECUTION TIME
  const gTT = g.map(s => s.tool_time.total_seconds);
  const cTT = c.map(s => s.tool_time.total_seconds);
  const gMS = g.map(s => s.tool_time.max_seconds);
  const cMS = c.map(s => s.tool_time.max_seconds);
  const gO10 = sum(g.map(s => s.tool_time.steps_over_10s));
  const cO10 = sum(c.map(s => s.tool_time.steps_over_10s));

  html += '<div class="section"><h3>Tool Execution Time</h3><table>' + hdr;
  html += aggRow('Mean total (s)', gTT, cTT, {dec: 1, desc: 'Total seconds waiting for tools per task. Like the "running bash…" spinner in Claude Code. Does not include model thinking time — that\'s not recorded.'});
  html += aggRow('Median total (s)', gTT, cTT, {fn: median, dec: 1, desc: 'Middle value — less skewed by tasks with long test suites or recursive greps.'});
  html += aggRow('Mean max step (s)', gMS, cMS, {dec: 1, desc: 'The single longest tool call in each task, averaged. Usually a full test suite run.'});
  html += `<tr><td>Steps &gt;10s (total)<span class="metric-desc">Tool calls that took over 10 seconds. Typically test runs or broad file searches.</span></td><td class="highlight-g">${gO10}</td><td class="highlight-c">${cO10}</td><td class="ratio">${ratio(cO10, gO10)}</td></tr>`;
  html += '</table>';
  html += '<div class="hist-row">';
  const gSt_ = g.map(s => s.steps);
  const cSt_ = c.map(s => s.steps);
  html += histogram(gTT, cTT, {title: 'Tool time per instance (s)', fmtTick: v => v.toFixed(0) + 's'});
  html += histogram(gSt_, cSt_, {title: 'Steps per instance', fmtTick: v => v.toFixed(0)});
  html += '</div>';
  html += '</div>';

  // STEPS & ACTIONS
  const gSt = g.map(s => s.steps);
  const cSt = c.map(s => s.steps);
  const aKeys = ['bash', 'view', 'edit', 'create', 'search_find', 'submit', 'other'];

  html += '<div class="section"><h3>Steps &amp; Actions</h3><table>' + hdr;
  html += aggRow('Mean steps', gSt, cSt, {desc: 'Total model turns per task. Each step = one prompt → response → tool execution cycle.'});
  const aDescs = {
    bash: 'Shell commands — running tests, installing deps, checking output.',
    view: 'Reading file contents. Like opening a file in your editor.',
    edit: 'Modifying existing files — str_replace, insert.',
    create: 'Creating new files from scratch.',
    search_find: 'find/grep to locate relevant files before reading them.',
    submit: 'Final patch submission — the agent says "I\'m done."',
    other: 'Unclassified actions.',
  };
  aKeys.forEach(k => {
    html += aggRow('&nbsp;&nbsp;' + k, g.map(s => s.actions[k]), c.map(s => s.actions[k]), {dec: 1, desc: aDescs[k]});
  });
  html += '</table></div>';

  // CONTENT VOLUME
  const gObs = g.map(s => s.content.observation_chars);
  const cObs = c.map(s => s.content.observation_chars);
  const gAct = g.map(s => s.content.action_chars);
  const cAct = c.map(s => s.content.action_chars);
  const gTht = g.map(s => s.content.thought_chars);
  const cTht = c.map(s => s.content.thought_chars);
  const gPat = g.map(s => s.patch_chars);
  const cPat = c.map(s => s.patch_chars);

  html += '<div class="section"><h3>Content Volume (mean chars/instance)</h3><table>' + hdr;
  html += aggRow('Observations (in)', gObs, cObs, {desc: 'Tool output the model consumed — file contents, command stdout, test results.'});
  html += aggRow('Actions (out)', gAct, cAct, {desc: 'What the model asked tools to do — shell commands, edit instructions, file paths.'});
  html += aggRow('Thoughts (out)', gTht, cTht, {desc: 'Visible reasoning text. Not all thinking is visible — GPT-5 uses hidden chain-of-thought.'});
  html += aggRow('Patch size (chars)', gPat, cPat, {desc: 'Size of the final submitted diff. Larger patches aren\'t necessarily better — could mean unnecessary changes.'});
  const gPatTok = g.map(s => s.patch_tokens);
  const cPatTok = c.map(s => s.patch_tokens);
  html += aggRow('Patch size (tokens)', gPatTok, cPatTok, {desc: 'Tiktoken count of the submitted diff. Direct measure of how much code the model changed.'});
  html += '</table>';
  html += '<div class="hist-row">';
  html += histogram(gPatTok, cPatTok, {title: 'Patch size (tokens)', fmtTick: v => (v/1e3).toFixed(1) + 'K'});
  html += '</div>';
  html += '</div>';

  // PER-INSTANCE TABLE
  const cols = [
    { key: 'instance_id', label: 'Instance', sortFn: (p) => p.instance_id },
    { key: 'repo', label: 'Repo', sortFn: (p) => p.repo },
    { key: 'outcome', label: 'Outcome', sortFn: (p) => outcomeOf(p) },
    { key: 'gpt5_cost', label: '$ GPT-5', sortFn: (p) => p.gpt5.model_stats.instance_cost },
    { key: 'claude_cost', label: '$ Claude S4.5', sortFn: (p) => p.claude.model_stats.instance_cost },
    { key: 'gpt5_steps', label: 'Steps GPT-5', sortFn: (p) => p.gpt5.steps },
    { key: 'claude_steps', label: 'Steps CS4.5', sortFn: (p) => p.claude.steps },
    { key: 'gpt5_input', label: 'Input GPT-5', sortFn: (p) => p.gpt5.model_stats.tokens_sent },
    { key: 'claude_input', label: 'Input CS4.5', sortFn: (p) => p.claude.model_stats.tokens_sent },
    { key: 'gpt5_output', label: 'Out GPT-5', sortFn: (p) => p.gpt5.output_tokens.total },
    { key: 'claude_output', label: 'Out CS4.5', sortFn: (p) => p.claude.output_tokens.total },
    { key: 'gpt5_patch', label: 'Patch GPT-5', sortFn: (p) => p.gpt5.patch_chars },
    { key: 'claude_patch', label: 'Patch CS4.5', sortFn: (p) => p.claude.patch_chars },
  ];

  // Sort
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

  // Attach sort handlers
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
<p class="muted" style="margin-top:24px">See also: <a href="unsubmitted.html" style="color:#58a6ff">Unsubmitted tasks report</a> — instances where one or both models failed to submit a patch.</p>
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

    # Slim down the data — drop fields we don't need in the HTML to keep file small
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
