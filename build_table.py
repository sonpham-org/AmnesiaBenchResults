"""Build a static HTML table:
   rows = problems, columns = models, cell = user-selected metric.

Metrics:
  n_reliable            — recomputed from traces (smallest passing N).
  n_unbounded           — n_while_unbounded from the unbounded phase.
  cost_unbounded        — nanodollars spent in the unbounded phase.
  cost_of_n_reliable    — nanodollars of the single sweep/binary_search entry
                          whose N == n_reliable. If that entry only has
                          'reused_unbounded' trials (cost=0), fall back to
                          cost_unbounded (per user rule).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from make_plot import MODELS, ROOT, compute_n_reliable, entry_cost_nanodollars


def _is_reused(entry: dict) -> bool:
    trials = entry.get("trials", []) or []
    if not trials:
        return False
    return all(t.get("finish_reason") == "reused_unbounded" for t in trials)


def build_data():
    data: dict[str, dict[str, dict]] = {}  # pid -> model -> metrics
    problem_ids: list[str] = []

    for model in MODELS:
        res_path = ROOT / model / "results_scott25.json"
        tr_path = ROOT / model / "traces_scott25.json"
        if not res_path.exists() or not tr_path.exists():
            continue

        results = json.loads(res_path.read_text())
        traces = json.loads(tr_path.read_text())
        tr_by_pid = {t["problem_id"]: t for t in traces}

        for r in results:
            pid = r["problem_id"]
            if pid not in data:
                data[pid] = {}
                problem_ids.append(pid)

            trace = tr_by_pid.get(pid)
            nr, entry, src = compute_n_reliable(trace) if trace else (math.inf, None, None)

            nwu = r.get("n_while_unbounded")
            cost_unb = (
                r.get("phase_breakdown", {}).get("unbounded", {}).get("cost_nanodollars")
            )

            if entry is not None:
                raw_cost = entry_cost_nanodollars(entry)
                if raw_cost == 0 and _is_reused(entry) and cost_unb is not None:
                    cost_nr = int(cost_unb)
                    cost_nr_note = "reused→unb"
                else:
                    cost_nr = int(raw_cost)
                    cost_nr_note = src or ""
            else:
                cost_nr = None
                cost_nr_note = ""

            def _finite(x):
                return x if isinstance(x, (int, float)) and math.isfinite(x) else None

            data[pid][model] = {
                "n_reliable": _finite(nr),
                "n_unbounded": _finite(nwu),
                "cost_unbounded": int(cost_unb) if cost_unb is not None else None,
                "cost_nr_run": cost_nr,
                "cost_nr_note": cost_nr_note,
            }

    problem_ids.sort()
    return problem_ids, data


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AmnesiaBench Scott-25 — per-problem table</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:24px; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  .controls { margin: 12px 0 20px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  select, button { background:#161b22; color:#c9d1d9; border:1px solid #30363d; padding:6px 10px; border-radius:6px; font-size:13px; }
  table { border-collapse: collapse; font-size: 12px; }
  th, td { border:1px solid #30363d; padding:6px 10px; text-align:right; white-space:nowrap; }
  th { background:#161b22; position:sticky; top:0; z-index:2; }
  th.sticky-col, td.sticky-col { position:sticky; left:0; background:#161b22; text-align:left; z-index:1; }
  td.val { font-variant-numeric: tabular-nums; }
  td.null { color:#484f58; }
  td.reused { color:#8957e5; }
  .legend { color:#8b949e; font-size:11px; margin-top:12px; }
  .hdr-note { color:#7d8590; font-size:11px; }
</style>
</head>
<body>
<h1>AmnesiaBench — Scott-25 · rows = problems · columns = models</h1>
<div class="controls">
  <label>Show:
    <select id="metric">
      <option value="n_reliable">n_reliable</option>
      <option value="n_unbounded">n_unbounded</option>
      <option value="cost_unbounded">cost_unbounded (nd)</option>
      <option value="cost_nr_run">cost of n_reliable run (nd, reused→unb)</option>
    </select>
  </label>
  <span class="hdr-note">Click column header to sort · violet = reused_unbounded (cost fell back to unbounded)</span>
</div>
<div id="table-wrap"></div>
<div class="legend">
  n_reliable is the smallest N where the model passes (≥2/3 in binary_search, or 1/1 in outer sweep).
  "∞" = never reached threshold. cost_nr_run uses only the trials at that exact N;
  if those were all reused_unbounded, we use cost_unbounded instead (user rule).
</div>

<script>
const PROBLEMS = __PROBLEMS__;
const MODELS   = __MODELS__;
const DATA     = __DATA__;

function fmt(v, metric) {
  if (v === null || v === undefined) return '—';
  if (!isFinite(v)) return '∞';
  if (metric.startsWith('cost') || metric.startsWith('n_')) {
    // integers for tokens; nanodollars formatted with commas
    if (metric.startsWith('cost')) return v.toLocaleString();
    return Math.round(v).toLocaleString();
  }
  return String(v);
}

let sortBy = null; // {model or '__pid__', asc}

function render() {
  const metric = document.getElementById('metric').value;
  const rows = PROBLEMS.slice();
  if (sortBy) {
    rows.sort((a,b) => {
      const va = sortBy.model === '__pid__' ? a : (DATA[a]?.[sortBy.model]?.[metric]);
      const vb = sortBy.model === '__pid__' ? b : (DATA[b]?.[sortBy.model]?.[metric]);
      const na = (va === null || va === undefined) ? Infinity : (va === Infinity ? 1e30 : va);
      const nb = (vb === null || vb === undefined) ? Infinity : (vb === Infinity ? 1e30 : vb);
      if (typeof va === 'string' || typeof vb === 'string') {
        return sortBy.asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
      }
      return sortBy.asc ? na - nb : nb - na;
    });
  }

  let h = '<table><thead><tr>';
  h += `<th class="sticky-col" data-col="__pid__">problem</th>`;
  for (const m of MODELS) h += `<th data-col="${m}">${m}</th>`;
  h += '</tr></thead><tbody>';
  for (const pid of rows) {
    h += `<tr><td class="sticky-col">${pid}</td>`;
    for (const m of MODELS) {
      const cell = DATA[pid]?.[m];
      const v = cell ? cell[metric] : null;
      const reused = metric === 'cost_nr_run' && cell && cell.cost_nr_note === 'reused→unb';
      const cls = ['val'];
      if (v === null || v === undefined) cls.push('null');
      if (reused) cls.push('reused');
      h += `<td class="${cls.join(' ')}">${fmt(v, metric)}</td>`;
    }
    h += '</tr>';
  }
  h += '</tbody></table>';
  document.getElementById('table-wrap').innerHTML = h;

  document.querySelectorAll('th[data-col]').forEach(th => {
    th.style.cursor = 'pointer';
    th.onclick = () => {
      const col = th.dataset.col;
      if (sortBy && sortBy.model === col) sortBy.asc = !sortBy.asc;
      else sortBy = { model: col, asc: true };
      render();
    };
  });
}

document.getElementById('metric').addEventListener('change', render);
render();
</script>
</body>
</html>
"""


def main():
    problem_ids, data = build_data()
    html = (HTML
            .replace("__PROBLEMS__", json.dumps(problem_ids))
            .replace("__MODELS__", json.dumps(MODELS))
            .replace("__DATA__", json.dumps(data, default=lambda v: None)))
    out = ROOT / "scott25_table.html"
    out.write_text(html)
    print(f"Wrote {out}  ({len(problem_ids)} problems × {len(MODELS)} models)")


if __name__ == "__main__":
    main()
