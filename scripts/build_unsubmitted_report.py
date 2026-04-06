"""
Build a self-contained HTML report for instances where one or both models
failed to submit a patch — the cases filtered out of the main paired report.

Shows why models gave up (exit status), how much they spent before failing,
and whether the other model succeeded on the same task.

Usage:
    python scripts/build_unsubmitted_report.py stats.json -o unsubmitted.html
"""

import argparse
import json
import sys
from collections import defaultdict

import orjson


def find_unsubmitted(data):
    """Find paired instances where at least one model didn't submit."""
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

    instances = []
    for iid in sorted(by_inst):
        m = by_inst[iid]
        if 'gpt5' not in m or 'claude' not in m:
            continue
        gs, cs = m['gpt5']['submitted'], m['claude']['submitted']
        if gs and cs:
            continue  # both submitted — that's the main report
        cat = 'neither'
        if gs and not cs:
            cat = 'gpt5_only'
        elif not gs and cs:
            cat = 'claude_only'
        instances.append({
            'instance_id': iid,
            'repo': m['gpt5']['repo'],
            'category': cat,
            'gpt5': m['gpt5'],
            'claude': m['claude'],
        })

    print(f"  Unsubmitted instances: {len(instances)}", file=sys.stderr)
    return instances


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SWE-Bench Pro: Unsubmitted Tasks</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'SF Mono', 'Consolas', 'Menlo', monospace; font-size: 13px; background: #0d1117; color: #c9d1d9; padding: 24px; line-height: 1.5; }
h1 { font-size: 18px; color: #f0f6fc; margin-bottom: 4px; }
h2 { font-size: 14px; color: #8b949e; font-weight: normal; margin-bottom: 20px; }
h3 { font-size: 14px; color: #f0f6fc; margin: 24px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #21262d; }
table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
th, td { text-align: right; padding: 4px 12px; border-bottom: 1px solid #21262d; white-space: nowrap; }
td:first-child { white-space: normal; max-width: 420px; }
th { color: #8b949e; font-weight: normal; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
td:first-child, th:first-child { text-align: left; }
tr:hover td { background: #161b22; }
.section { margin-bottom: 32px; }
.highlight-g { color: #3fb950; }
.highlight-c { color: #a371f7; }
.muted { color: #484f58; }
.metric-desc { color: #484f58; font-size: 11px; font-weight: normal; font-style: italic; display: block; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
.tag-submitted { background: #1b3826; color: #3fb950; }
.tag-not-submitted { background: #3d1f1f; color: #f85149; }
.tag-resolved { background: #1b3826; color: #3fb950; }
.tag-failed { background: #3d1f1f; color: #f85149; }
a { color: #58a6ff; }
</style>
</head>
<body>

<h1>SWE-Bench Pro: Unsubmitted Tasks</h1>
<h2>Instances where one or both models failed to submit a patch — filtered out of the main paired report</h2>

<div id="report"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;

function fmt(n, dec) {
  if (dec === undefined) dec = 0;
  if (n === Infinity || n === -Infinity || isNaN(n)) return '—';
  return n.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}
function fmtD(n) { return '$' + fmt(n, 2); }
function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }

function tag(submitted, resolved) {
  if (!submitted) return '<span class="tag tag-not-submitted">not submitted</span>';
  if (resolved) return '<span class="tag tag-resolved">resolved ✓</span>';
  return '<span class="tag tag-failed">submitted, wrong</span>';
}

function render() {
  let html = '';

  // Summary
  const nGOnly = DATA.filter(d => d.category === 'gpt5_only').length;
  const nCOnly = DATA.filter(d => d.category === 'claude_only').length;
  const nNeither = DATA.filter(d => d.category === 'neither').length;

  html += '<div class="section"><h3>Summary</h3><table>';
  html += '<tr><th style="text-align:left">Category</th><th>Count</th><th>Description</th></tr>';
  html += `<tr><td>GPT-5 submitted, Claude didn't</td><td>${nGOnly}</td><td class="muted">Claude hit a limit or errored out</td></tr>`;
  html += `<tr><td>Claude submitted, GPT-5 didn't</td><td>${nCOnly}</td><td class="muted">GPT-5 hit a limit or errored out</td></tr>`;
  html += `<tr><td>Neither submitted</td><td>${nNeither}</td><td class="muted">Both gave up</td></tr>`;
  html += `<tr><td><b>Total</b></td><td><b>${DATA.length}</b></td><td></td></tr>`;
  html += '</table></div>';

  // Exit status breakdown
  const exitCounts = {};
  DATA.forEach(d => {
    ['gpt5', 'claude'].forEach(model => {
      if (!d[model].submitted) {
        const key = model + ': ' + (d[model].exit_status || 'unknown');
        exitCounts[key] = (exitCounts[key] || 0) + 1;
      }
    });
  });

  html += '<div class="section"><h3>Why models didn\'t submit</h3>';
  html += '<p class="metric-desc">Exit status from SWE-Agent when the agent stopped without submitting a patch.</p>';
  html += '<table>';
  html += '<tr><th style="text-align:left">Exit status</th><th>Count</th></tr>';
  Object.entries(exitCounts).sort((a, b) => b[1] - a[1]).forEach(([key, count]) => {
    html += `<tr><td>${key}</td><td>${count}</td></tr>`;
  });
  html += '</table></div>';

  // Spending before giving up
  const unsub = DATA.flatMap(d => {
    const out = [];
    if (!d.gpt5.submitted) out.push({model: 'GPT-5', ...d.gpt5});
    if (!d.claude.submitted) out.push({model: 'Claude Sonnet 4.5', ...d.claude});
    return out;
  });
  const gUnsub = unsub.filter(s => s.model === 'GPT-5');
  const cUnsub = unsub.filter(s => s.model === 'Claude Sonnet 4.5');

  html += '<div class="section"><h3>Resources spent before giving up</h3>';
  html += '<p class="metric-desc">What each model consumed on tasks where it failed to submit — wasted spend.</p>';
  html += '<table>';
  html += '<tr><th style="text-align:left"></th><th>GPT-5</th><th>Claude Sonnet 4.5</th></tr>';

  function row(label, gArr, cArr, dollar) {
    const gv = gArr.length ? mean(gArr) : 0;
    const cv = cArr.length ? mean(cArr) : 0;
    const gs = dollar ? fmtD(gv) : fmt(gv, 0);
    const cs = dollar ? fmtD(cv) : fmt(cv, 0);
    const gn = gArr.length ? ` (n=${gArr.length})` : ' —';
    const cn = cArr.length ? ` (n=${cArr.length})` : ' —';
    html += `<tr><td>${label}</td><td class="highlight-g">${gArr.length ? gs : '—'}${gArr.length ? '' : ''}</td><td class="highlight-c">${cArr.length ? cs : '—'}</td></tr>`;
  }

  row('Mean cost', gUnsub.map(s => s.model_stats.instance_cost), cUnsub.map(s => s.model_stats.instance_cost), true);
  row('Mean steps', gUnsub.map(s => s.steps), cUnsub.map(s => s.steps));
  row('Mean input tokens', gUnsub.map(s => s.model_stats.tokens_sent), cUnsub.map(s => s.model_stats.tokens_sent));
  row('Mean output tokens', gUnsub.map(s => s.output_tokens.total), cUnsub.map(s => s.output_tokens.total));
  html += `<tr><td>Unsubmitted instances</td><td class="muted">${gUnsub.length}</td><td class="muted">${cUnsub.length}</td></tr>`;
  html += '</table></div>';

  // Did the OTHER model succeed?
  html += '<div class="section"><h3>Did the other model succeed?</h3>';
  html += '<p class="metric-desc">When one model gave up, did the other submit and resolve the task?</p>';
  html += '<table>';
  html += '<tr><th style="text-align:left">Scenario</th><th>Count</th></tr>';

  let gQuit_cResolved = 0, gQuit_cSubmitted = 0, gQuit_cFailed = 0;
  let cQuit_gResolved = 0, cQuit_gSubmitted = 0, cQuit_gFailed = 0;
  DATA.forEach(d => {
    if (!d.gpt5.submitted && d.claude.submitted) {
      gQuit_cSubmitted++;
      if (d.claude.resolved) gQuit_cResolved++;
      else gQuit_cFailed++;
    }
    if (!d.claude.submitted && d.gpt5.submitted) {
      cQuit_gSubmitted++;
      if (d.gpt5.resolved) cQuit_gResolved++;
      else cQuit_gFailed++;
    }
  });

  html += `<tr><td>GPT-5 quit → Claude resolved</td><td>${gQuit_cResolved}</td></tr>`;
  html += `<tr><td>GPT-5 quit → Claude submitted but wrong</td><td>${gQuit_cFailed}</td></tr>`;
  html += `<tr><td>Claude quit → GPT-5 resolved</td><td>${cQuit_gResolved}</td></tr>`;
  html += `<tr><td>Claude quit → GPT-5 submitted but wrong</td><td>${cQuit_gFailed}</td></tr>`;
  html += '</table></div>';

  // Per-instance detail
  html += '<div class="section"><h3>Per-Instance Detail</h3><table>';
  html += '<tr><th style="text-align:left">Instance</th><th style="text-align:left">Repo</th>';
  html += '<th>GPT-5 status</th><th>Claude status</th>';
  html += '<th>$ GPT-5</th><th>$ Claude</th>';
  html += '<th>Steps G</th><th>Steps C</th>';
  html += '<th style="text-align:left">GPT-5 exit</th><th style="text-align:left">Claude exit</th></tr>';

  DATA.forEach(d => {
    const shortId = d.instance_id.replace('instance_', '');
    const truncId = shortId.length > 55 ? shortId.slice(0, 55) + '…' : shortId;
    html += '<tr>';
    html += `<td title="${shortId}" style="font-size:11px;max-width:350px;overflow:hidden;text-overflow:ellipsis">${truncId}</td>`;
    html += `<td>${d.repo}</td>`;
    html += `<td>${tag(d.gpt5.submitted, d.gpt5.resolved)}</td>`;
    html += `<td>${tag(d.claude.submitted, d.claude.resolved)}</td>`;
    html += `<td class="highlight-g">${fmtD(d.gpt5.model_stats.instance_cost)}</td>`;
    html += `<td class="highlight-c">${fmtD(d.claude.model_stats.instance_cost)}</td>`;
    html += `<td class="highlight-g">${d.gpt5.steps}</td>`;
    html += `<td class="highlight-c">${d.claude.steps}</td>`;
    html += `<td style="text-align:left">${d.gpt5.exit_status}</td>`;
    html += `<td style="text-align:left">${d.claude.exit_status}</td>`;
    html += '</tr>';
  });
  html += '</table></div>';

  html += '<p class="muted" style="margin-top:24px">Data: ' + DATA.length + ' instances where at least one model did not submit. ';
  html += 'See <a href="index.html">main report</a> for the ' + __PAIRED_COUNT__ + ' paired submitted instances.</p>';

  document.getElementById('report').innerHTML = html;
}

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file")
    parser.add_argument("-o", "--output", default="unsubmitted.html")
    parser.add_argument("--paired-count", type=int, default=0,
                        help="Number of paired submitted instances (for cross-link text)")
    args = parser.parse_args()

    with open(args.stats_file, 'rb') as f:
        data = orjson.loads(f.read())

    instances = find_unsubmitted(data)

    keep_keys = [
        'instance_id', 'repo', 'resolved', 'submitted', 'steps', 'patch_chars',
        'patch_tokens', 'exit_status', 'model_stats', 'tokens_per_call',
        'output_tokens', 'tool_time', 'actions', 'content',
    ]
    slim = []
    for inst in instances:
        si = {'instance_id': inst['instance_id'], 'repo': inst['repo'], 'category': inst['category']}
        for model in ('gpt5', 'claude'):
            si[model] = {k: inst[model][k] for k in keep_keys if k in inst[model]}
        slim.append(si)

    json_data = orjson.dumps(slim).decode('utf-8')
    html = HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', json_data)
    html = html.replace('__PAIRED_COUNT__', str(args.paired_count))

    with open(args.output, 'w') as f:
        f.write(html)

    print(f"  Wrote {args.output} ({len(html) / 1024:.0f} KB, {len(slim)} instances)", file=sys.stderr)


if __name__ == "__main__":
    main()
