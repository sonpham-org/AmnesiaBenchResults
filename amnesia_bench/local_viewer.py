#!/usr/bin/env python3
"""
AmnesiaBench Local Viewer — inspect results + traces with proper UI.

Usage:
    python3 local_viewer.py              # http://localhost:5555
    python3 local_viewer.py --port 8888
"""

import argparse
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_results():
    cells = {}
    models_set = set()
    problems_set = set()
    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.suffix == ".json" or f.name.endswith("_summary.json") or f.name.endswith("_prediction.json"):
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except:
            continue
        model = d.get("model_name") or d.get("model", "")
        pid = d.get("problem_id", "")
        if not model or not pid:
            continue
        models_set.add(model)
        problems_set.add(pid)
        cell = cells.setdefault(pid, {}).setdefault(model, {})
        config_raw = d.get("config", {})
        if config_raw == "Unbounded":
            cell["ub_avg_tokens"] = d.get("avg_tokens")
            cell["ub_min_tokens"] = d.get("min_tokens")
            cell["ub_max_tokens"] = d.get("max_tokens")
            cell["ub_solve_rate"] = d.get("solve_rate")
            cell["ub_n_runs"] = d.get("n_runs")
            cell["ub_runs"] = d.get("runs", [])
        elif isinstance(config_raw, dict) and config_raw.get("name") == "Sweep":
            cell["sweep"] = d.get("sweep_results", [])
            cell["sweep_min"] = d.get("min_passing_truncation")
            cell["pass_curve"] = d.get("pass_curve", [])
        elif isinstance(config_raw, dict) and "Compact" in config_raw.get("name", ""):
            bs = d.get("binary_search", [])
            prediction = d.get("prediction", {})
            cell["co_min_window"] = d.get("minimum_window")
            # Compute pass rate across all steps
            total_success = sum(s.get("n_success", 0) for s in bs)
            total_trials = sum(s.get("n_trials", 0) for s in bs)
            if d.get("minimum_window"):
                cell["co_result_state"] = "CONVERGED"
            elif total_trials > 0:
                cell["co_result_state"] = f"{total_success}/{total_trials} ({total_success*100//total_trials}%)"
            elif bs:
                cell["co_result_state"] = "running"
            else:
                cell["co_result_state"] = "N/A"
            cell["n_reliable_prediction"] = (prediction or {}).get("n_reliable_prediction")
            cell["co_steps"] = bs
    # Compute where_is_the_result — fast regex on text, no LLM calls
    # But skip loading full conversations into the API response to keep it fast
    import re as _re
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from ollama_runner import load_all_problems as _lap
        all_probs = _lap()
    except Exception:
        all_probs = {}
    for pid, model_cells in cells.items():
        correct = (all_probs.get(pid) or {}).get("correct_answer")
        for model_name, c in model_cells.items():
            runs = c.get("ub_runs", [])
            if not runs:
                continue
            ap = []
            for ri, run in enumerate(runs):
                # Find assistant thinking + content
                thinking = ""
                content = ""
                for msg in run.get("conversation", []):
                    if msg.get("role") == "assistant":
                        thinking = msg.get("thinking", "")
                        content = msg.get("content", "")
                        break
                full = thinking + "\n" + content if thinking else content
                total_tok = run.get("total_tokens", len(full) // 4)
                matches = list(_re.finditer(r"\\boxed\{([^}]+)\}", full))
                if matches:
                    def _parse(m):
                        v = m.group(1).strip()
                        try: return int(v.replace(",",""))
                        except: return v
                    f, l = matches[0], matches[-1]
                    fp, lp = f.start()//4, l.start()//4
                    fv, lv = _parse(f), _parse(l)
                    ap.append({"trial_idx":ri,"answer_found":True,"n_boxed":len(matches),
                        "first_answer":fv,"first_token_pos":fp,"first_pct":round(fp/max(total_tok,1)*100,1),"first_correct":str(fv)==str(correct) if correct else None,
                        "last_answer":lv,"last_token_pos":lp,"last_pct":round(lp/max(total_tok,1)*100,1),"last_correct":str(lv)==str(correct) if correct else None,
                        "total_tokens":total_tok})
                else:
                    ap.append({"trial_idx":ri,"answer_found":False,"n_boxed":0,"total_tokens":total_tok,
                        "first_answer":None,"first_token_pos":None,"first_pct":None,"first_correct":None,
                        "last_answer":None,"last_token_pos":None,"last_pct":None,"last_correct":None})
            c["answer_positions"] = ap

    for prob in cells.values():
        for c in prob.values():
            c.setdefault("ub_avg_tokens", None)
            c.setdefault("ub_solve_rate", None)
            c.setdefault("ub_n_runs", None)
            c.setdefault("ub_runs", [])
            c.setdefault("co_min_window", None)
            c.setdefault("co_result_state", "N/A")
            c.setdefault("co_steps", [])
            c.setdefault("n_reliable_prediction", None)
            c.setdefault("answer_positions", [])
            c.setdefault("sweep", [])
            c.setdefault("sweep_min", None)
            c.setdefault("pass_curve", [])
    return {"problems": sorted(problems_set), "models": sorted(models_set), "cells": cells}


_results_cache = {"data": None, "mtime": 0}

def load_results_cached():
    """Cache results — only reload if results/ dir has changed."""
    mtime = max((f.stat().st_mtime for f in RESULTS_DIR.iterdir() if f.suffix == ".json"), default=0)
    if _results_cache["data"] and mtime <= _results_cache["mtime"]:
        return _results_cache["data"]
    data = load_results()
    _results_cache["data"] = data
    _results_cache["mtime"] = mtime
    return data


# ── Prompt tuning results loader ─────────────────────────────────────────────

PROMPT_TUNING_DIR = Path(__file__).resolve().parent.parent / "results_prompt_tuning"

_pt_cache = {"data": None, "mtime": 0}


def load_prompt_tuning():
    """Load prompt tuning results from results_prompt_tuning/.

    File naming: {model}_{pid}_t{trial}_w{window}_{variant}.json
    Returns: {models, problems, variants, cells: {pid: {model: {variant: [trials]}}}}
    """
    import re as _re

    models_set = set()
    problems_set = set()
    variants_set = set()
    # cells[pid][model][variant] = list of trial dicts
    cells = {}

    # Scan both prompt_tuning dir and main results dir for variant-tagged trial files
    scan_dirs = []
    if PROMPT_TUNING_DIR.exists():
        scan_dirs.append(PROMPT_TUNING_DIR)
    if RESULTS_DIR.exists():
        scan_dirs.append(RESULTS_DIR)
    if not scan_dirs:
        return {"models": [], "problems": [], "variants": [], "cells": {}}

    for scan_dir in scan_dirs:
      for f in sorted(scan_dir.iterdir()):
        if not f.suffix == ".json":
            continue
        # Parse filename: {model}_{pid}_t{trial}_w{window}_{variant}.json
        m = _re.match(r"^(.+?)_(.+?)_t(\d+)_w(\d+)_(.+)\.json$", f.name)
        if not m:
            continue
        model_safe, pid, trial_idx, window, variant = m.groups()
        trial_idx = int(trial_idx)
        window = int(window)

        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        model = d.get("model") or d.get("model_name") or model_safe
        models_set.add(model)
        problems_set.add(pid)
        variants_set.add(variant)

        trial_entry = {
            "trial_idx": trial_idx,
            "window": window,
            "success": d.get("success", False),
            "answer": d.get("answer"),
            "correct_answer": d.get("correct_answer"),
            "n_compactions": d.get("n_compactions", 0),
            "wall_time_s": d.get("wall_time_s", 0),
            "finish_reason": d.get("finish_reason"),
            "conversation": d.get("conversation", []),
        }

        cells.setdefault(pid, {}).setdefault(model, {}).setdefault(variant, []).append(trial_entry)

    return {
        "models": sorted(models_set),
        "problems": sorted(problems_set),
        "variants": sorted(variants_set),
        "cells": cells,
    }


def load_dashboard_data():
    """Aggregate data for the dashboard visualization.
    Returns: {models[], problems[], grid: {pid: {model: {solve_rate, avg_tokens, min_window, ratio}}},
              prompt_tuning: {pid: {model: {variant: {pass_rate, n}}}}}
    """
    models_set = set()
    problems_set = set()
    grid = {}  # pid -> model -> stats

    # Scan grouped Unbounded files (not individual trials)
    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.suffix == ".json" or not f.name.endswith("_Unbounded.json"):
            continue
        if "_t" in f.stem:
            continue
        try:
            d = json.load(open(f))
        except:
            continue
        model = d.get("model_name") or d.get("model", "")
        pid = d.get("problem_id", "")
        if not model or not pid:
            continue
        # Only scott problems for now
        if not pid.startswith("scott_"):
            continue
        models_set.add(model)
        problems_set.add(pid)
        entry = grid.setdefault(pid, {}).setdefault(model, {})
        entry["solve_rate"] = d.get("solve_rate", 0)
        entry["avg_tokens"] = d.get("avg_tokens", 0)
        entry["n_runs"] = d.get("n_runs", 0)

    # Scan Compact files
    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.suffix == ".json" or "_Compact" not in f.name:
            continue
        if "_t" in f.stem:
            continue
        try:
            d = json.load(open(f))
        except:
            continue
        model = d.get("model_name") or d.get("model", "")
        pid = d.get("problem_id", "")
        if not model or not pid or not pid.startswith("scott_"):
            continue
        entry = grid.setdefault(pid, {}).setdefault(model, {})
        mw = d.get("minimum_window")
        entry["min_window"] = mw
        if mw and entry.get("avg_tokens"):
            entry["ratio"] = round(mw / entry["avg_tokens"], 2)
        # Extract avg compactions from the binary search steps
        steps = d.get("binary_search", [])
        if steps:
            all_compactions = []
            for step in steps:
                for trial in step.get("trials", []):
                    if trial.get("success"):
                        all_compactions.append(trial.get("n_compactions", 0))
            if all_compactions:
                entry["avg_compactions"] = round(sum(all_compactions) / len(all_compactions), 1)

    # Scan Sweep files
    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.suffix == ".json" or "_Sweep" not in f.name:
            continue
        if "_t" in f.stem:
            continue
        try:
            d = json.load(open(f))
        except:
            continue
        model = d.get("model_name") or d.get("model", "")
        pid = d.get("problem_id", "")
        if not model or not pid or not pid.startswith("scott_"):
            continue
        entry = grid.setdefault(pid, {}).setdefault(model, {})
        entry["sweep_min"] = d.get("min_passing_truncation")
        entry["sweep_curve"] = d.get("pass_curve", [])
        # Extract avg compactions from sweep results
        sweep_results = d.get("sweep_results", [])
        if sweep_results:
            sweep_compactions = []
            for sr in sweep_results:
                for trial in sr.get("trials", []):
                    sweep_compactions.append(trial.get("n_compactions", 0))
            if sweep_compactions:
                entry["sweep_avg_compactions"] = round(sum(sweep_compactions) / len(sweep_compactions), 1)

    # Prompt tuning data
    pt = {}
    if PROMPT_TUNING_DIR.exists():
        import re as _re
        for f in sorted(PROMPT_TUNING_DIR.iterdir()):
            if not f.suffix == ".json":
                continue
            m = _re.match(r"^(.+?)_(.+?)_t(\d+)_w(\d+)_(.+)\.json$", f.name)
            if not m:
                continue
            _, pid, _, _, variant = m.groups()
            if not pid.startswith("scott_"):
                continue
            try:
                d = json.load(open(f))
            except:
                continue
            model = d.get("model") or d.get("model_name", "")
            success = d.get("success", False)
            vdata = pt.setdefault(pid, {}).setdefault(model, {}).setdefault(variant, {"pass": 0, "total": 0})
            vdata["total"] += 1
            if success:
                vdata["pass"] += 1

    return {
        "models": sorted(models_set),
        "problems": sorted(problems_set),
        "grid": grid,
        "prompt_tuning": pt,
    }


def load_prompt_tuning_cached():
    if not PROMPT_TUNING_DIR.exists():
        return {"models": [], "problems": [], "variants": [], "cells": {}}
    mtime = max((f.stat().st_mtime for f in PROMPT_TUNING_DIR.iterdir() if f.suffix == ".json"), default=0)
    if _pt_cache["data"] and mtime <= _pt_cache["mtime"]:
        return _pt_cache["data"]
    data = load_prompt_tuning()
    _pt_cache["data"] = data
    _pt_cache["mtime"] = mtime
    return data


HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>AmnesiaBench Local Viewer</title>
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea','code'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; display: flex; flex-direction: column; height: 100vh; }
  #table-pane { flex: 1; overflow: auto; padding: 12px; }
  #trace-pane { width: 48%; overflow: auto; padding: 12px; border-left: 1px solid #21262d; background: #0d1117; }

  table { border-collapse: collapse; font-size: 11px; width: 100%; }
  th, td { border: 1px solid #21262d; padding: 4px 8px; text-align: center; cursor: pointer; }
  th { background: #161b22; position: sticky; top: 0; z-index: 10; }
  .pass { background: #0d2818; color: #3fb950; }
  .fail { background: #2d1012; color: #f85149; }
  .partial { background: #2d2200; color: #d29922; }
  .na { color: #484f58; }
  td.active-cell { outline: 2px solid #58a6ff; outline-offset: -2px; }

  .trace-header { font-size: 14px; font-weight: bold; margin-bottom: 12px; color: #58a6ff; }

  /* Tabs */
  .tab-bar { display: flex; gap: 0; border-bottom: 1px solid #21262d; margin: 8px 0 0 0; }
  .tab { padding: 6px 14px; font-size: 11px; cursor: pointer; border: 1px solid transparent; border-bottom: none;
         border-radius: 6px 6px 0 0; color: #8b949e; background: none; }
  .tab:hover { color: #c9d1d9; }
  .tab.active { background: #161b22; color: #58a6ff; border-color: #21262d; border-bottom: 1px solid #161b22; margin-bottom: -1px; }
  .tab.pass-tab { color: #3fb950; }
  .tab.fail-tab { color: #f85149; }
  .tab-content { display: none; padding: 8px 0; }
  .tab-content.active { display: block; }

  /* Sections */
  .section { margin: 10px 0; border: 1px solid #21262d; border-radius: 6px; overflow: hidden; }
  .section-title { font-weight: bold; font-size: 12px; color: #58a6ff; padding: 8px 12px; background: #161b22; cursor: pointer; }
  .section-body { padding: 8px 12px; }
  .meta { font-size: 11px; color: #8b949e; margin: 2px 0; }
  .meta strong { color: #c9d1d9; }

  /* Binary search steps */
  .step-bar { display: flex; gap: 4px; flex-wrap: wrap; margin: 8px 0; }
  .step-chip { padding: 3px 10px; border-radius: 12px; font-size: 10px; cursor: pointer; border: 1px solid #21262d; }
  .step-chip.passed { background: #0d2818; color: #3fb950; border-color: #238636; }
  .step-chip.failed { background: #2d1012; color: #f85149; border-color: #da3633; }
  .step-chip.active { outline: 2px solid #58a6ff; }

  /* Trial tabs inside a step */
  .trial-tabs { display: flex; gap: 0; margin: 6px 0; border-bottom: 1px solid #21262d; }
  .trial-tab { padding: 4px 12px; font-size: 10px; cursor: pointer; border-radius: 4px 4px 0 0; }
  .trial-tab:hover { background: #161b22; }
  .trial-tab.active { background: #161b22; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
  .trial-tab.pass-tab { color: #3fb950; }
  .trial-tab.fail-tab { color: #f85149; }

  /* Conversation */
  .turn { margin: 4px 0; padding: 6px 10px; border-radius: 6px; font-size: 11px; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .turn.system { background: #161b22; color: #8b949e; border-left: 3px solid #484f58; }
  .turn.user { background: #0d2240; border-left: 3px solid #1f6feb; }
  .turn.assistant { background: #161b22; border-left: 3px solid #3fb950; }
  .turn-role { font-weight: bold; font-size: 10px; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.5px; }
  .turn-role.system { color: #484f58; }
  .turn-role.user { color: #58a6ff; }
  .turn-role.assistant { color: #3fb950; }
  .turn-tokens { font-size: 9px; color: #484f58; float: right; }
  .think { background: #1c1230; border-left: 3px solid #8957e5; padding: 6px 10px; margin: 4px 0; border-radius: 4px; }
  .think-label { font-size: 9px; color: #8957e5; cursor: pointer; font-weight: bold; }
  .think-body { display: block; font-size: 10px; color: #8b949e; margin-top: 4px; }

  /* Compaction markers */
  .compact-event { background: #2d2200; border-left: 4px solid #d29922; padding: 8px 12px; margin: 8px 0; border-radius: 4px; }
  .compact-event .label { font-weight: bold; color: #d29922; font-size: 11px; }
  .compact-event .detail { font-size: 10px; color: #8b949e; margin-top: 2px; }
  .restart-event { background: #0d2818; border-left: 4px solid #238636; padding: 6px 12px; margin: 4px 0; border-radius: 4px;
                   font-size: 10px; color: #3fb950; font-weight: bold; }

  /* Stats bar */
  .stats-bar { display: flex; gap: 16px; padding: 8px 12px; background: #161b22; border-radius: 6px; margin: 8px 0; font-size: 11px; }
  .stat { display: flex; flex-direction: column; }
  .stat-label { font-size: 9px; color: #484f58; text-transform: uppercase; }
  .stat-value { font-weight: bold; color: #c9d1d9; }

  /* Top-level page tabs */
  .page-tabs { display: flex; gap: 0; padding: 8px 12px 0; background: #0d1117; border-bottom: 2px solid #21262d; }
  .page-tab { padding: 8px 20px; font-size: 13px; font-weight: bold; cursor: pointer; color: #8b949e;
              border: 1px solid transparent; border-bottom: none; border-radius: 8px 8px 0 0; }
  .page-tab:hover { color: #c9d1d9; }
  .page-tab.active { background: #161b22; color: #58a6ff; border-color: #21262d; border-bottom: 1px solid #161b22; margin-bottom: -2px; }

  /* Prompt tuning table */
  .pt-pass { background: #0d2818; color: #3fb950; font-weight: bold; }
  .pt-fail { background: #2d1012; color: #f85149; }
  .pt-partial { background: #2d2200; color: #d29922; }
  .pt-best { outline: 2px solid #3fb950; outline-offset: -2px; }
</style>
</head><body>
<div class="page-tabs">
  <div class="page-tab active" onclick="switchPage('dash')">Dashboard</div>
  <div class="page-tab" onclick="switchPage('mxp')">Model × Problem</div>
  <div class="page-tab" onclick="switchPage('pt')">Prompt Tuning</div>
</div>
<div id="page-dash" style="display:flex;flex:1;overflow:auto;padding:16px;">
  <div id="dash-pane" style="width:100%;">Loading dashboard...</div>
</div>
<div id="page-mxp" style="display:none;flex:1;overflow:hidden;">
  <div id="table-pane" style="flex:1;overflow:auto;padding:12px;">Loading...</div>
  <div id="trace-pane" style="width:48%;overflow:auto;padding:12px;border-left:1px solid #21262d;"><div style="padding:20px;color:#484f58;">Click a cell to view traces.</div></div>
</div>
<div id="page-pt" style="display:none;flex:1;overflow:hidden;">
  <div id="pt-table-pane" style="flex:1;overflow:auto;padding:12px;">Loading prompt tuning data...</div>
  <div id="pt-trace-pane" style="width:48%;overflow:auto;padding:12px;border-left:1px solid #21262d;"><div style="padding:20px;color:#484f58;">Click a cell to view traces.</div></div>
</div>
<script>
var DATA = null, PT_DATA = null, DASH_DATA = null;
var ACTIVE = null, ACTIVE_STEP = null, ACTIVE_TRIAL = 0, ACTIVE_UB_TAB = 0, ACTIVE_MAIN_TAB = 'ub';
var ACTIVE_SWEEP_POINT = null, ACTIVE_SWEEP_TRIAL = 0;
var PT_ACTIVE = null, PT_ACTIVE_TRIAL = 0;
var CURRENT_PAGE = 'dash';

function switchPage(page) {
  CURRENT_PAGE = page;
  var tabs = ['dash','mxp','pt'];
  document.querySelectorAll('.page-tab').forEach(function(t,i){ t.classList.toggle('active', tabs[i]===page); });
  tabs.forEach(function(t){ document.getElementById('page-'+t).style.display = t===page ? 'flex' : 'none'; });
  if (page === 'pt' && !PT_DATA) {
    fetch('/api/prompt_tuning').then(r=>r.json()).then(d=>{PT_DATA=d; renderPTTable();});
  }
  if (page === 'mxp' && !DATA) {
    fetch('/api/results').then(r=>r.json()).then(d=>{DATA=d; renderTable();});
  }
  if (page === 'dash' && !DASH_DATA) {
    fetch('/api/dashboard').then(r=>r.json()).then(d=>{DASH_DATA=d; renderDashboard();});
  }
}

// Load dashboard on start
fetch('/api/dashboard').then(r=>r.json()).then(d=>{DASH_DATA=d; renderDashboard();});

function renderDashboard() {
  var d = DASH_DATA;
  if (!d || !d.problems.length) {
    document.getElementById('dash-pane').innerHTML = '<div style="color:#484f58;">No data yet.</div>';
    return;
  }

  var h = '<h2 style="color:#58a6ff;margin:0 0 16px;">AmnesiaBench Dashboard</h2>';

  // ── 0. CompactBench Leaderboard (rendered first, built later) ─────────
  var LEADERBOARD_PLACEHOLDER = '<!--LEADERBOARD-->';
  h += LEADERBOARD_PLACEHOLDER;

  // ── 1. Heatmap: Model × Problem ──────────────────────────────────────
  h += '<h3 style="color:#c9d1d9;margin:16px 0 8px;">Benchmark Overview</h3>';
  h += '<table style="font-size:11px;"><thead><tr><th style="text-align:left;">Problem</th>';
  for (var m of d.models) h += '<th style="min-width:80px;">' + esc(m.split(':')[0]) + '</th>';
  h += '</tr></thead><tbody>';
  for (var p of d.problems) {
    h += '<tr><td style="text-align:left;font-size:10px;white-space:nowrap;">' + esc(p.replace('scott_','')) + '</td>';
    for (var m of d.models) {
      var c = (d.grid[p]||{})[m];
      if (!c) { h += '<td class="na" style="font-size:10px;">—</td>'; continue; }
      var rate = c.solve_rate || 0;
      var bg = rate >= 0.6 ? 'rgba(63,185,80,'+(0.15+rate*0.4)+')' : rate > 0 ? 'rgba(210,153,34,'+(0.15+rate*0.3)+')' : 'rgba(248,81,73,0.15)';
      var fg = rate >= 0.6 ? '#3fb950' : rate > 0 ? '#d29922' : '#f85149';
      h += '<td style="background:'+bg+';color:'+fg+';font-size:10px;text-align:center;">';
      h += Math.round(rate*100)+'%';
      if (c.min_window) h += '<br><span style="font-size:9px;color:#8b949e;">w='+fmtK(c.min_window)+'</span>';
      if (c.avg_compactions != null) h += '<br><span style="font-size:9px;color:#a371f7;">'+c.avg_compactions+' comp</span>';
      h += '</td>';
    }
    h += '</tr>';
  }
  h += '</tbody></table>';

  // ── 2. Model Summary Cards ───────────────────────────────────────────
  h += '<h3 style="color:#c9d1d9;margin:24px 0 8px;">Model Summary</h3>';
  h += '<div style="display:flex;gap:12px;flex-wrap:wrap;">';
  for (var m of d.models) {
    var solved = 0, total = 0, totalTokens = 0, compactable = 0, avgRatio = 0, ratioCount = 0;
    var totalCompactions = 0, compactionCount = 0;
    for (var p of d.problems) {
      var c = (d.grid[p]||{})[m];
      if (!c) continue;
      total++;
      if (c.solve_rate > 0) solved++;
      totalTokens += c.avg_tokens || 0;
      if (c.min_window && c.avg_tokens) {
        compactable++;
        avgRatio += c.min_window / c.avg_tokens;
        ratioCount++;
      }
      if (c.avg_compactions != null) {
        totalCompactions += c.avg_compactions;
        compactionCount++;
      }
    }
    var ar = ratioCount ? (avgRatio/ratioCount) : null;
    var avgComp = compactionCount ? (totalCompactions/compactionCount) : null;
    h += '<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 16px;min-width:200px;">';
    h += '<div style="font-weight:bold;color:#58a6ff;font-size:13px;margin-bottom:8px;">' + esc(m) + '</div>';
    h += '<div style="font-size:11px;color:#c9d1d9;">';
    h += 'Solved: <strong style="color:'+(solved>0?'#3fb950':'#f85149')+';">' + solved + '/' + total + '</strong><br>';
    h += 'Avg tokens: <strong>' + (total?Math.round(totalTokens/total):'—') + '</strong><br>';
    h += 'Compactable: <strong>' + compactable + '/' + solved + '</strong><br>';
    h += 'Avg compact ratio: <strong>' + (ar?ar.toFixed(2)+'×':'—') + '</strong><br>';
    h += 'Avg compactions: <strong style="color:#a371f7;">' + (avgComp!=null?avgComp.toFixed(1):'—') + '</strong>';
    h += '</div></div>';
  }
  h += '</div>';

  // ── 3. Compaction Ratio Bar Chart (CSS-based) ────────────────────────
  h += '<h3 style="color:#c9d1d9;margin:24px 0 8px;">Compaction Ratio (min_window / avg_tokens)</h3>';
  h += '<div style="font-size:10px;color:#484f58;margin-bottom:8px;">Lower = better compression. Green ≤ 1× means compaction found a smaller window than unbounded average.</div>';
  for (var p of d.problems) {
    var hasAny = false;
    for (var m of d.models) { if ((d.grid[p]||{})[m] && (d.grid[p]||{})[m].ratio) hasAny = true; }
    if (!hasAny) continue;
    h += '<div style="margin:8px 0 4px;font-size:11px;color:#c9d1d9;font-weight:bold;">' + esc(p.replace('scott_','')) + '</div>';
    for (var m of d.models) {
      var c = (d.grid[p]||{})[m];
      if (!c || !c.ratio) continue;
      var ratio = c.ratio;
      var pct = Math.min(ratio * 50, 100); // 2× = 100% width
      var col = ratio <= 1.0 ? '#3fb950' : ratio <= 1.5 ? '#d29922' : '#f85149';
      h += '<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">';
      h += '<span style="width:120px;font-size:10px;color:#8b949e;text-align:right;">' + esc(m.split(':')[0]) + '</span>';
      h += '<div style="flex:1;background:#21262d;border-radius:3px;height:14px;position:relative;">';
      h += '<div style="width:'+pct+'%;background:'+col+';height:100%;border-radius:3px;"></div>';
      h += '</div>';
      h += '<span style="width:50px;font-size:10px;color:'+col+';">'+ratio.toFixed(2)+'×</span>';
      h += '</div>';
    }
  }

  // ── 4. Compactions Required ───────────────────────────────────────────
  h += '<h3 style="color:#c9d1d9;margin:24px 0 8px;">Compactions Required</h3>';
  h += '<div style="font-size:10px;color:#484f58;margin-bottom:8px;">Average number of compactions needed to solve at minimum window. Lower = model retains more across resets.</div>';
  h += '<table style="font-size:11px;"><thead><tr><th style="text-align:left;">Problem</th>';
  for (var m of d.models) h += '<th style="min-width:70px;">' + esc(m.split(':')[0]) + '</th>';
  h += '</tr></thead><tbody>';
  for (var p of d.problems) {
    var hasAny = false;
    for (var m of d.models) { var c2 = (d.grid[p]||{})[m]; if (c2 && c2.avg_compactions != null) hasAny = true; }
    if (!hasAny) continue;
    h += '<tr><td style="text-align:left;font-size:10px;white-space:nowrap;">' + esc(p.replace('scott_','')) + '</td>';
    for (var m of d.models) {
      var c2 = (d.grid[p]||{})[m];
      if (!c2 || c2.avg_compactions == null) { h += '<td class="na" style="font-size:10px;">—</td>'; continue; }
      var nc = c2.avg_compactions;
      var col2 = nc <= 1 ? '#3fb950' : nc <= 2 ? '#d29922' : nc <= 3 ? '#f0883e' : '#f85149';
      h += '<td style="text-align:center;color:'+col2+';font-size:11px;font-weight:bold;">'+nc.toFixed(1)+'</td>';
    }
    h += '</tr>';
  }
  h += '</tbody></table>';

  // ── 5. Sweep Curves ──────────────────────────────────────────────────
  h += '<h3 style="color:#c9d1d9;margin:24px 0 8px;">Sweep Pass Curves</h3>';
  h += '<div style="font-size:10px;color:#484f58;margin-bottom:8px;">Pass rate at each truncation point. Shows how aggressively you can compress.</div>';
  var modelColors = {};
  var palette = ['#58a6ff','#3fb950','#f0883e','#f85149','#a371f7','#d29922','#79c0ff','#7ee787'];
  d.models.forEach(function(m,i){ modelColors[m] = palette[i % palette.length]; });

  for (var p of d.problems) {
    var hasAny = false;
    for (var m of d.models) { if ((d.grid[p]||{})[m] && (d.grid[p]||{})[m].sweep_curve && (d.grid[p]||{})[m].sweep_curve.length) hasAny = true; }
    if (!hasAny) continue;
    h += '<div style="margin:12px 0;padding:8px 12px;background:#161b22;border-radius:6px;border:1px solid #21262d;">';
    h += '<div style="font-size:11px;font-weight:bold;color:#c9d1d9;margin-bottom:6px;">' + esc(p.replace('scott_','')) + '</div>';
    // Simple text-based sweep display
    for (var m of d.models) {
      var c = (d.grid[p]||{})[m];
      if (!c || !c.sweep_curve || !c.sweep_curve.length) continue;
      var col = modelColors[m];
      h += '<div style="font-size:10px;color:'+col+';margin:2px 0;">' + esc(m.split(':')[0]) + ': ';
      for (var si=0; si<c.sweep_curve.length; si++) {
        var sc = c.sweep_curve[si];
        var pRate = Math.round((sc.rate||0)*100);
        var bg2 = pRate >= 60 ? 'rgba(63,185,80,0.3)' : pRate > 0 ? 'rgba(210,153,34,0.3)' : 'rgba(248,81,73,0.15)';
        h += '<span style="display:inline-block;padding:1px 6px;margin:1px;border-radius:3px;background:'+bg2+';">';
        h += fmtK(sc.point) + ':' + pRate + '%</span> ';
      }
      h += '</div>';
    }
    h += '</div>';
  }

  // ── 6. Prompt Tuning Summary ─────────────────────────────────────────
  if (d.prompt_tuning && Object.keys(d.prompt_tuning).length) {
    h += '<h3 style="color:#c9d1d9;margin:24px 0 8px;">Prompt Variant Comparison</h3>';
    var allVariants = {};
    for (var pp in d.prompt_tuning) for (var mm in d.prompt_tuning[pp]) for (var vv in d.prompt_tuning[pp][mm]) allVariants[vv] = true;
    var variants = Object.keys(allVariants).sort();
    h += '<table style="font-size:11px;"><thead><tr><th>Problem</th><th>Model</th>';
    for (var v of variants) h += '<th>' + esc(v) + '</th>';
    h += '</tr></thead><tbody>';
    for (var pp in d.prompt_tuning) {
      for (var mm in d.prompt_tuning[pp]) {
        h += '<tr><td style="text-align:left;font-size:10px;">' + esc(pp.replace('scott_','')) + '</td>';
        h += '<td style="font-size:10px;">' + esc(mm.split(':')[0]) + '</td>';
        var bestRate = 0;
        for (var v of variants) { var vd = (d.prompt_tuning[pp][mm]||{})[v]; if (vd && vd.total) bestRate = Math.max(bestRate, vd.pass/vd.total); }
        for (var v of variants) {
          var vd = (d.prompt_tuning[pp][mm]||{})[v];
          if (!vd || !vd.total) { h += '<td class="na">—</td>'; continue; }
          var vRate = vd.pass/vd.total;
          var cls2 = vRate >= 0.6 ? 'pt-pass' : vRate > 0 ? 'pt-partial' : 'pt-fail';
          if (vRate === bestRate && vRate > 0) cls2 += ' pt-best';
          h += '<td class="'+cls2+'">'+vd.pass+'/'+vd.total+'</td>';
        }
        h += '</tr>';
      }
    }
    h += '</tbody></table>';
  }

  // ── 7. CompactBench Leaderboard (built here, inserted at top) ────────
  var lb = '<h3 style="color:#c9d1d9;margin:0 0 8px;">CompactBench Leaderboard</h3>';
  lb += '<div style="font-size:10px;color:#484f58;margin-bottom:8px;">'
     + '<strong>CompactBench</strong>: geometric mean of minimum context windows needed (2/3+ solve threshold). '
     + 'Unsolvable = 131k penalty. Lower = better.<br>'
     + '<strong>Compression</strong>: geometric mean of (min_window / unbounded_tokens). Lower = better compression.'
     + '</div>';

  var PENALTY = 131072;
  var benchModels = [];
  for (var m of d.models) {
    var windows = [], solvedCount = 0, totalCount = 0, ratioLogs = [], totalComp = 0, compCount = 0;
    for (var p of d.problems) {
      var c = (d.grid[p]||{})[m];
      if (!c) continue;
      totalCount++;
      if (c.solve_rate > 0) solvedCount++;
      if (c.min_window) {
        windows.push(c.min_window);
        if (c.avg_tokens && c.avg_tokens > 0) ratioLogs.push(Math.log(c.min_window / c.avg_tokens));
        if (c.avg_compactions != null) { totalComp += c.avg_compactions; compCount++; }
      } else if (c.solve_rate > 0) {
        windows.push(c.avg_tokens || PENALTY);
      } else {
        windows.push(PENALTY);
      }
    }
    if (totalCount === 0) continue;
    var logSum = 0;
    for (var w of windows) logSum += Math.log(w);
    var geoMean = Math.exp(logSum / windows.length);
    var solveRate = solvedCount / totalCount;
    var geoCompress = ratioLogs.length ? Math.exp(ratioLogs.reduce(function(a,b){return a+b;},0) / ratioLogs.length) : null;
    var avgCompactions = compCount ? totalComp / compCount : null;
    benchModels.push({
      model: m, compactBench: geoMean, solveRate: solveRate,
      geoCompress: geoCompress, avgCompactions: avgCompactions,
      solved: solvedCount, total: totalCount
    });
  }
  benchModels.sort(function(a,b){ return a.compactBench - b.compactBench; });

  lb += '<table style="font-size:12px;width:100%;">';
  lb += '<thead><tr style="background:#161b22;">';
  lb += '<th style="padding:6px 10px;text-align:left;">#</th>';
  lb += '<th style="padding:6px 10px;text-align:left;">Model</th>';
  lb += '<th style="padding:6px 10px;">CompactBench \u2193</th>';
  lb += '<th style="padding:6px 10px;">Solve Rate</th>';
  lb += '<th style="padding:6px 10px;">Compression</th>';
  lb += '<th style="padding:6px 10px;">Avg Compactions</th>';
  lb += '</tr></thead><tbody>';
  for (var i=0; i<benchModels.length; i++) {
    var bm = benchModels[i];
    var rowBg = i===0 ? 'rgba(63,185,80,0.12)' : (i===1 ? 'rgba(63,185,80,0.06)' : '');
    lb += '<tr style="border-bottom:1px solid #21262d;'+(rowBg?'background:'+rowBg+';':'')+'">';
    lb += '<td style="padding:6px 10px;color:#8b949e;">' + (i+1) + '</td>';
    lb += '<td style="padding:6px 10px;font-weight:bold;color:#58a6ff;">' + esc(bm.model) + '</td>';
    lb += '<td style="padding:6px 10px;text-align:center;font-weight:bold;color:#c9d1d9;">' + fmtK(Math.round(bm.compactBench)) + '</td>';
    var srCol = bm.solveRate >= 0.6 ? '#3fb950' : bm.solveRate > 0.3 ? '#d29922' : '#f85149';
    lb += '<td style="padding:6px 10px;text-align:center;color:'+srCol+';">' + Math.round(bm.solveRate*100) + '% (' + bm.solved + '/' + bm.total + ')</td>';
    lb += '<td style="padding:6px 10px;text-align:center;color:'+(bm.geoCompress!=null?(bm.geoCompress<=1?'#3fb950':'#d29922'):'#484f58')+';">' + (bm.geoCompress!=null?bm.geoCompress.toFixed(2)+'\u00d7':'\u2014') + '</td>';
    lb += '<td style="padding:6px 10px;text-align:center;color:#a371f7;">' + (bm.avgCompactions!=null?bm.avgCompactions.toFixed(1):'\u2014') + '</td>';
    lb += '</tr>';
  }
  lb += '</tbody></table>';

  h = h.replace(LEADERBOARD_PLACEHOLDER, lb);
  document.getElementById('dash-pane').innerHTML = h;
}


function renderTable() {
  var d = DATA, h = '<table><thead><tr><th>Problem</th>';
  for (var m of d.models) h += '<th colspan="3">' + esc(m) + '</th>';
  h += '</tr><tr><th></th>';
  for (var m of d.models) h += '<th>Unbounded</th><th>Compact</th><th>Sweep</th>';
  h += '</tr></thead><tbody>';
  for (var p of d.problems) {
    h += '<tr><td style="text-align:left;font-size:10px;">' + esc(p) + '</td>';
    for (var m of d.models) {
      var c = (d.cells[p]||{})[m] || {};
      var ub = c.ub_avg_tokens, rate = c.ub_solve_rate;
      var cls = rate != null ? (rate >= 0.6 ? 'pass' : rate > 0 ? 'partial' : 'fail') : 'na';
      var id = 'cell-'+p+'-'+m;
      h += '<td class="'+cls+'" id="'+id+'-ub" onclick="sel(\''+esc(p)+'\',\''+esc(m)+'\',\'ub\')">';
      h += ub != null ? Math.round(ub) + (rate != null ? ' <small>('+Math.round(rate*100)+'%)</small>' : '') : '-';
      h += '</td>';
      var co = c.co_min_window, st = c.co_result_state || 'N/A';
      cls = st === 'CONVERGED' ? 'pass' : st === 'FAILED' ? 'fail' : 'na';
      h += '<td class="'+cls+'" id="'+id+'-co" onclick="sel(\''+esc(p)+'\',\''+esc(m)+'\',\'co\')">';
      h += co != null ? fmtK(co) : (st !== 'N/A' ? st : '-');
      h += '</td>';
      // Sweep column
      var sw = c.sweep || [], swMin = c.sweep_min;
      var swCls = swMin != null ? 'pass' : (sw.length ? 'partial' : 'na');
      h += '<td class="'+swCls+'" id="'+id+'-sw" onclick="sel(\''+esc(p)+'\',\''+esc(m)+'\',\'sweep\')">';
      h += swMin != null ? fmtK(swMin) : (sw.length ? sw.length+' pts' : '-');
      h += '</td>';
    }
    h += '</tr>';
  }
  h += '</tbody></table>';
  document.getElementById('table-pane').innerHTML = h;
}

function sel(pid, model, cfg) {
  document.querySelectorAll('.active-cell').forEach(e=>e.classList.remove('active-cell'));
  var el = document.getElementById('cell-'+pid+'-'+model+'-'+cfg);
  if (el) el.classList.add('active-cell');
  ACTIVE = {pid, model, cfg}; ACTIVE_STEP = null; ACTIVE_TRIAL = 0; ACTIVE_UB_TAB = 0;
  ACTIVE_SWEEP_POINT = null; ACTIVE_SWEEP_TRIAL = 0;
  ACTIVE_MAIN_TAB = cfg === 'sweep' ? 'sweep' : (cfg === 'co' ? 'co' : 'ub');
  renderTrace();
}
function selMainTab(t) { ACTIVE_MAIN_TAB = t; renderTrace(); }

function selStep(i) { ACTIVE_STEP = i; ACTIVE_TRIAL = 0; renderTrace(); }
function selTrial(t) { ACTIVE_TRIAL = t; renderTrace(); }
function selUbTab(t) { ACTIVE_UB_TAB = t; renderTrace(); }
function selSweepPoint(i) { ACTIVE_SWEEP_POINT = (ACTIVE_SWEEP_POINT===i?null:i); ACTIVE_SWEEP_TRIAL = 0; renderTrace(); }
function selSweepTrial(t) { ACTIVE_SWEEP_TRIAL = t; renderTrace(); }

function renderTrace() {
  var a = ACTIVE; if (!a) return;
  var c = ((DATA.cells[a.pid]||{})[a.model]) || {};
  var h = '<div class="trace-header">' + esc(a.model) + ' / ' + esc(a.pid) + '</div>';

  // Stats bar
  var pred = c.n_reliable_prediction;
  h += '<div class="stats-bar"><div class="stat"><span class="stat-label">Predicted N</span><span class="stat-value">'+(pred!=null?fmtK(pred):'\u2014')+'</span></div>';
  if (c.ub_avg_tokens != null) h += '<div class="stat"><span class="stat-label">Unbounded Avg</span><span class="stat-value">'+fmtK(c.ub_avg_tokens)+'</span></div><div class="stat"><span class="stat-label">Solve Rate</span><span class="stat-value">'+(c.ub_solve_rate!=null?Math.round(c.ub_solve_rate*100)+'%':'\u2014')+'</span></div>';
  if (c.co_min_window != null) h += '<div class="stat"><span class="stat-label">Min Window</span><span class="stat-value">'+fmtK(c.co_min_window)+'</span></div>';
  h += '</div>';

  // === TOP-LEVEL TABS ===
  var ubCount = (c.ub_runs||[]).length;
  var coCount = (c.co_steps||[]).length;
  var swCount = (c.sweep||[]).length;
  h += '<div class="tab-bar" style="margin-top:12px;">';
  h += '<div class="tab'+(ACTIVE_MAIN_TAB==='stats'?' active':'')+'" onclick="selMainTab(\'stats\')">Stats</div>';
  h += '<div class="tab'+(ACTIVE_MAIN_TAB==='ub'?' active':'')+'" onclick="selMainTab(\'ub\')">Unbounded'+(ubCount?' ('+ubCount+')':'')+'</div>';
  h += '<div class="tab'+(ACTIVE_MAIN_TAB==='co'?' active':'')+'" onclick="selMainTab(\'co\')">Compact'+(coCount?' ('+coCount+')':'')+'</div>';
  h += '<div class="tab'+(ACTIVE_MAIN_TAB==='sweep'?' active':'')+'" onclick="selMainTab(\'sweep\')">Sweep'+(swCount?' ('+swCount+')':'')+'</div>';
  h += '</div>';

  if (ACTIVE_MAIN_TAB === 'stats') {
    // === STATS TAB ===
    h += '<div style="padding:8px 0;">';
    // Where Is The Result
    var ap = c.answer_positions || [];
    if (ap.length) {
      h += '<div class="section-title" style="margin:8px 0;">Where Is The Result?</div>';
      h += '<table style="width:100%;font-size:11px;border-collapse:collapse;">';
      h += '<tr style="background:#161b22;"><th style="padding:4px 8px;text-align:left;">Run</th><th>First \\\\boxed{}</th><th>Last \\\\boxed{}</th><th>Total Tok</th><th># boxed</th></tr>';
      for (var i=0; i<ap.length; i++) {
        var a = ap[i];
        var fc = a.first_correct ? '#3fb950' : '#f85149';
        var lc = a.last_correct ? '#3fb950' : '#f85149';
        h += '<tr style="border-bottom:1px solid #21262d;">';
        h += '<td style="padding:4px 8px;">Run '+(i+1)+'</td>';
        if (a.answer_found) {
          h += '<td style="text-align:center;color:'+fc+';">'+esc(String(a.first_answer))+' @ '+a.first_pct+'% ('+a.first_token_pos+' tok)</td>';
          h += '<td style="text-align:center;color:'+lc+';">'+esc(String(a.last_answer))+' @ '+a.last_pct+'% ('+a.last_token_pos+' tok)</td>';
          h += '<td style="text-align:center;">'+a.total_tokens+'</td>';
          h += '<td style="text-align:center;">'+a.n_boxed+'</td>';
        } else {
          h += '<td colspan="4" style="text-align:center;color:#484f58;">No answer found</td>';
        }
        h += '</tr>';
      }
      h += '</table>';
    } else {
      h += '<div class="meta">No unbounded runs — cannot compute answer positions.</div>';
    }
    // Summary stats
    h += '<div class="section-title" style="margin:12px 0 8px 0;">Summary</div>';
    h += '<table style="width:100%;font-size:11px;border-collapse:collapse;">';
    h += '<tr><td style="padding:3px 8px;">Predicted N</td><td style="padding:3px 8px;">'+(pred!=null?fmtK(pred):'\u2014')+'</td></tr>';
    h += '<tr><td style="padding:3px 8px;">Unbounded Avg Tokens</td><td style="padding:3px 8px;">'+(c.ub_avg_tokens!=null?c.ub_avg_tokens:'\u2014')+'</td></tr>';
    h += '<tr><td style="padding:3px 8px;">Unbounded Solve Rate</td><td style="padding:3px 8px;">'+(c.ub_solve_rate!=null?Math.round(c.ub_solve_rate*100)+'%':'\u2014')+'</td></tr>';
    h += '<tr><td style="padding:3px 8px;">Compact Min Window</td><td style="padding:3px 8px;">'+(c.co_min_window!=null?fmtK(c.co_min_window):'\u2014')+'</td></tr>';
    h += '<tr><td style="padding:3px 8px;">Sweep Min Truncation</td><td style="padding:3px 8px;">'+(c.sweep_min!=null?fmtK(c.sweep_min):'\u2014')+'</td></tr>';
    h += '</table>';
    h += '</div>';

  } else if (ACTIVE_MAIN_TAB === 'sweep') {
    // === SWEEP TAB ===
    var sw = c.sweep || [];
    if (!sw.length) { h += '<div class="meta" style="padding:12px;">No sweep data. Run with --sweep-only.</div>'; }
    else {
      h += '<div style="padding:8px 0;">';
      h += '<table style="width:100%;font-size:11px;border-collapse:collapse;">';
      h += '<tr style="background:#161b22;"><th style="padding:4px 8px;">Truncation</th><th>Window</th><th>Pass Rate</th><th>Compactions (avg)</th></tr>';
      for (var i=0; i<sw.length; i++) {
        var s = sw[i];
        var isActive = ACTIVE_SWEEP_POINT === i;
        var bgc = isActive ? 'rgba(88,166,255,0.15)' : (s.passed ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.05)');
        var avgC = 0;
        if (s.trials && s.trials.length) {
          avgC = s.trials.reduce(function(a,t){return a+t.n_compactions;},0) / s.trials.length;
        }
        h += '<tr style="border-bottom:1px solid #21262d;background:'+bgc+';cursor:pointer;'+(isActive?'outline:1px solid #58a6ff;':'')+'" onclick="selSweepPoint('+i+')">';
        h += '<td style="padding:4px 8px;">'+fmtK(s.truncation_point)+'</td>';
        h += '<td style="padding:4px 8px;">'+fmtK(s.window)+'</td>';
        h += '<td style="padding:4px 8px;text-align:center;color:'+(s.passed?'#3fb950':'#f85149')+';">'+s.n_success+'/'+s.n_trials+' ('+Math.round(s.pass_rate*100)+'%)</td>';
        h += '<td style="padding:4px 8px;text-align:center;">'+avgC.toFixed(1)+'</td>';
        h += '</tr>';
      }
      h += '</table>';
      if (c.sweep_min != null) h += '<div class="meta" style="margin-top:8px;">Min passing truncation: <strong>'+fmtK(c.sweep_min)+'</strong></div>';

      // === SWEEP TRIAL VIEWER (below table) ===
      if (ACTIVE_SWEEP_POINT != null && sw[ACTIVE_SWEEP_POINT]) {
        var sp = sw[ACTIVE_SWEEP_POINT];
        h += '<div style="margin-top:12px;border-top:1px solid #30363d;padding-top:8px;">';
        h += '<div class="meta">Window: <strong>'+fmtK(sp.window)+'</strong> | Truncation: '+fmtK(sp.truncation_point)+' | '+(sp.passed?'PASSED':'FAILED')+' | '+sp.n_success+'/'+sp.n_trials+'</div>';
        var trials = sp.trials || [];
        if (trials.length) {
          h += '<div class="trial-tabs">';
          for (var t=0; t<trials.length; t++) {
            var tr = trials[t], tcls = tr.success ? 'pass-tab' : 'fail-tab';
            h += '<div class="trial-tab '+tcls+(ACTIVE_SWEEP_TRIAL===t?' active':'')+'" onclick="event.stopPropagation();selSweepTrial('+t+')">Trial '+(t+1)+' '+(tr.success?'\u2713':'\u2717')+' | '+tr.n_compactions+' compact</div>';
          }
          h += '</div>';
          var at = trials[ACTIVE_SWEEP_TRIAL];
          if (at) {
            h += '<div class="meta">Answer: <strong>'+esc(String(at.answer))+'</strong> | Tokens: '+at.total_tokens_peak+' | Compactions: <strong>'+at.n_compactions+'</strong> | Time: '+(at.wall_time_s?at.wall_time_s.toFixed(1)+'s':'\u2014')+' | Finish: '+at.finish_reason+'</div>';
            h += renderConv(at.conversation || []);
          }
        }
        h += '</div>';
      }
      h += '</div>';
    }

  } else if (ACTIVE_MAIN_TAB === 'ub') {
    // === UNBOUNDED TAB ===
    var runs = c.ub_runs || [];
    if (!runs.length) { h += '<div class="meta" style="padding:12px;">No unbounded data.</div>'; }
    else {
      h += '<div class="tab-bar">';
      for (var i=0; i<runs.length; i++) {
        var r = runs[i], tcls = r.success ? 'pass-tab' : 'fail-tab';
        h += '<div class="tab '+tcls+(ACTIVE_UB_TAB===i?' active':'')+'" onclick="selUbTab('+i+')">Run '+(i+1)+' '+(r.success?'\u2713':'\u2717')+'</div>';
      }
      h += '</div>';
      var ar = runs[ACTIVE_UB_TAB];
      if (ar) {
        h += '<div class="meta" style="padding:8px 0;">Tokens: <strong>'+ar.total_tokens+'</strong> | Think: '+(ar.thinking_tokens||0)+' | Time: '+(ar.wall_time_s?ar.wall_time_s.toFixed(1)+'s':'\u2014')+' | Answer: <strong>'+esc(String(ar.answer))+'</strong></div>';
        h += renderConv(ar.conversation || []);
      }
    }
  } else {
    // === COMPACT TAB ===
    var steps = c.co_steps || [];
    if (!steps.length) { h += '<div class="meta" style="padding:12px;">No compact data.</div>'; }
    else {
      h += '<div class="step-bar">';
      for (var i=0; i<steps.length; i++) {
        var s = steps[i], scls = s.passed ? 'passed' : 'failed';
        if (ACTIVE_STEP===i) scls += ' active';
        h += '<div class="step-chip '+scls+'" onclick="selStep('+i+')" title="'+(s.n_success||0)+'/'+(s.n_trials||0)+' passed">'+fmtK(s.window)+'</div>';
      }
      h += '</div>';

      if (ACTIVE_STEP != null && steps[ACTIVE_STEP]) {
        var step = steps[ACTIVE_STEP];
        h += '<div class="meta">Window: <strong>'+fmtK(step.window)+'</strong> | '+(step.passed?'PASSED':'FAILED')+' | '+(step.n_success||0)+'/'+(step.n_trials||0)+' trials</div>';
        var trials = step.trials || [];
        if (trials.length) {
          h += '<div class="trial-tabs">';
          for (var t=0; t<trials.length; t++) {
            var tr = trials[t], tcls2 = tr.success ? 'pass-tab' : 'fail-tab';
            h += '<div class="trial-tab '+tcls2+(ACTIVE_TRIAL===t?' active':'')+'" onclick="selTrial('+t+')">Trial '+(t+1)+' '+(tr.success?'\u2713':'\u2717')+' | '+tr.n_compactions+' compact</div>';
          }
          h += '</div>';
          var at = trials[ACTIVE_TRIAL];
          if (at) {
            h += '<div class="meta">Answer: <strong>'+esc(String(at.answer))+'</strong> | Tokens: '+at.total_tokens_peak+' | Compactions: <strong>'+at.n_compactions+'</strong> | Time: '+(at.wall_time_s?at.wall_time_s.toFixed(1)+'s':'\u2014')+' | Finish: '+at.finish_reason+'</div>';
            h += renderConv(at.conversation || []);
          }
        }
      } else { h += '<div class="meta" style="padding:12px;color:#484f58;">Click a step chip above to view trials.</div>'; }
    }
  }

  document.getElementById('trace-pane').innerHTML = h;
}

function renderConv(conv) {
  var h = '';
  for (var idx = 0; idx < conv.length; idx++) {
    var msg = conv[idx];
    var role = msg.role || 'unknown';
    var content = msg.content || '';
    var tokens = msg.total_tokens || msg.tokens;

    // Compaction events
    if (content.indexOf('[COMPACTION TRIGGERED') !== -1) {
      h += '<div class="compact-event"><div class="label">\u26A1 Compaction Triggered</div><div class="detail">'+esc(content)+'</div></div>';
      continue;
    }
    if (content.indexOf('[COMPACTION #') !== -1 || content.indexOf('[FORCED COMPACTION') !== -1) {
      h += '<div class="compact-event"><div class="label">\uD83D\uDCE6 '+esc(content.split(']')[0]+']')+'</div></div>';
      continue;
    }
    if (content.indexOf('[RESTART') !== -1 || (msg.label && msg.label.indexOf('[RESTART') !== -1)) {
      var label = msg.label || content;
      var actualContent = (msg.label && content !== msg.label) ? content : null;
      if (!actualContent) {
        // Old format — try to reconstruct
        var resumeHtml = _reconstructResume(conv, idx);
        if (resumeHtml) actualContent = resumeHtml;
      }
      if (actualContent) {
        var rid = 'resume-'+Math.random().toString(36).substr(2,6);
        h += '<div class="restart-event" style="cursor:pointer;" onclick="var b=document.getElementById(\''+rid+'\');b.style.display=b.style.display===\'none\'?\'block\':\'none\';">\u21BB '+esc(label)+' <span style="font-size:9px;color:#484f58;">(click to expand)</span></div>';
        h += '<div id="'+rid+'" class="conv-msg" style="border-left:3px solid #3fb950;display:none;">';
        h += '<div class="role">\u21BB resume prompt</div>';
        h += '<div class="body">'+escMath(actualContent)+'</div></div>';
        continue;
      }
      h += '<div class="restart-event">\u21BB '+esc(label)+'</div>';
      continue;
    }

    h += '<div class="turn '+role+'">';
    h += '<div class="turn-role '+role+'">'+role+(tokens ? '<span class="turn-tokens">'+tokens+' tok</span>' : '')+'</div>';

    // Show thinking field (separate from content) if present
    var thinking = msg.thinking || '';
    if (thinking) {
      var tid = 'think-'+Math.random().toString(36).substr(2,6);
      h += '<div class="think"><div class="think-label" onclick="var b=document.getElementById(\''+tid+'\');b.style.display=b.style.display===\'none\'?\'block\':\'none\';this.textContent=b.style.display===\'none\'?\'\u25B6 thinking (' +Math.round(thinking.length/4)+ ' tok)...\':\'\u25BC thinking (' +Math.round(thinking.length/4)+ ' tok)\';">\u25BC thinking ('+Math.round(thinking.length/4)+' tok)</div><div class="think-body" id="'+tid+'" style="display:block">'+escMath(thinking)+'</div></div>';
    }

    // Split inline <think> blocks in content (some models embed them)
    var parts = content.split(/<think>([\s\S]*?)<\/think>/g);
    for (var i=0; i<parts.length; i++) {
      if (i % 2 === 1) {
        var tid2 = 'think-'+Math.random().toString(36).substr(2,6);
        h += '<div class="think"><div class="think-label" onclick="var b=document.getElementById(\''+tid2+'\');b.style.display=b.style.display===\'none\'?\'block\':\'none\';">\u25BC thinking ('+Math.round(parts[i].length/4)+' tok)</div><div class="think-body" id="'+tid2+'" style="display:block">'+escMath(parts[i])+'</div></div>';
      } else if (parts[i].trim()) {
        h += escMath(parts[i]);
      }
    }
    h += '</div>';
  }
  return h;
}

function esc(s) {
  if (!s) return '';
  var d = document.createElement('div');
  d.textContent = s;
  var out = d.innerHTML;
  out = out.replace(/\n/g, '<br>');
  return out;
}
function _reconstructResume(conv, restartIdx) {
  var userMsg = '';
  for (var j = 0; j < conv.length; j++) {
    if (conv[j].role === 'user' && conv[j].content && conv[j].content.indexOf('[') !== 0) {
      userMsg = conv[j].content; break;
    }
  }
  if (!userMsg) return null;
  var summary = '';
  for (var j = restartIdx - 1; j >= 0; j--) {
    if (conv[j].role === 'assistant') {
      var c = conv[j].content || '';
      var m = c.match(/<compact>([\\s\\S]*?)<\/compact>/);
      if (m) { summary = m[1].trim(); break; }
      if (c.length > 0) { summary = c.substring(0, 500); break; }
    }
  }
  if (!summary) return null;
  var nDone = 0;
  for (var j = 0; j <= restartIdx; j++) {
    if (conv[j].content && conv[j].content.indexOf('[COMPACTION #') !== -1) nDone++;
  }
  var prompt = userMsg + '\n\n'
    + '\u2550\u2550\u2550 SAVED CONTEXT (compaction #' + nDone + ') \u2550\u2550\u2550\n'
    + '<prior_work>\n' + summary + '\n</prior_work>\n\n'
    + 'Continue solving. Give your final answer as \\boxed{integer}.';
  return escMath(prompt);
}
function escMath(s) {
  // Escape HTML but preserve LaTeX delimiters for MathJax
  if (!s) return '';
  // Split on LaTeX delimiters, escape non-math parts, keep math parts raw
  // Handle $$...$$ , $...$ , \[...\] , \(...\)
  var result = '';
  var i = 0;
  while (i < s.length) {
    // Check for $$ (display math)
    if (s[i] === '$' && s[i+1] === '$') {
      var end = s.indexOf('$$', i+2);
      if (end !== -1) {
        result += '$$' + s.substring(i+2, end) + '$$';
        i = end + 2; continue;
      }
    }
    // Check for \[ (display math)
    if (s[i] === '\\' && s[i+1] === '[') {
      var end = s.indexOf('\\]', i+2);
      if (end !== -1) {
        result += '\\[' + s.substring(i+2, end) + '\\]';
        i = end + 2; continue;
      }
    }
    // Check for \( (inline math)
    if (s[i] === '\\' && s[i+1] === '(') {
      var end = s.indexOf('\\)', i+2);
      if (end !== -1) {
        result += '\\(' + s.substring(i+2, end) + '\\)';
        i = end + 2; continue;
      }
    }
    // Check for $ (inline math) — be careful not to match currency
    if (s[i] === '$' && s[i+1] !== '$') {
      var end = s.indexOf('$', i+1);
      if (end !== -1 && end - i < 500) { // reasonable math block length
        var inner = s.substring(i+1, end);
        if (inner.indexOf('\n\n') === -1) { // no double newlines in inline math
          result += '$' + inner + '$';
          i = end + 1; continue;
        }
      }
    }
    // Check for \boxed
    if (s.substring(i, i+7) === '\\boxed') {
      var j = i+7;
      if (s[j] === '{') {
        var depth = 1; j++;
        while (j < s.length && depth > 0) {
          if (s[j] === '{') depth++;
          if (s[j] === '}') depth--;
          j++;
        }
        result += '$' + s.substring(i, j) + '$';
        i = j; continue;
      }
    }
    // Regular character — escape it
    var ch = s[i];
    if (ch === '<') result += '&lt;';
    else if (ch === '>') result += '&gt;';
    else if (ch === '&') result += '&amp;';
    else if (ch === '\n') result += '<br>';
    else result += ch;
    i++;
  }
  return result;
}
function fmtK(v) { if (v==null) return '\u2014'; if (v >= 1024) return (v/1024).toFixed(v%1024===0?0:1)+'k'; return String(v); }
// Re-render LaTeX after DOM update
function retypeset() { if (window.MathJax && MathJax.typesetPromise) { MathJax.typesetPromise().catch(function(){}); } }

// Override renderTrace to call retypeset after
var _origRenderTrace = renderTrace;
renderTrace = function() { _origRenderTrace(); setTimeout(retypeset, 50); };

// ════════════════════════════════════════════════════════════════════
// PROMPT TUNING TAB
// ════════════════════════════════════════════════════════════════════

function renderPTTable() {
  var d = PT_DATA;
  if (!d || !d.problems.length) {
    document.getElementById('pt-table-pane').innerHTML = '<div style="padding:20px;color:#484f58;">No prompt tuning data found in results_prompt_tuning/</div>';
    return;
  }
  // Table: rows = (problem, model), columns = variants
  var h = '<table><thead><tr><th>Problem</th><th>Model</th>';
  for (var v of d.variants) h += '<th>' + esc(v) + '</th>';
  h += '</tr></thead><tbody>';

  for (var p of d.problems) {
    var mc = d.cells[p] || {};
    for (var m of d.models) {
      var vc = mc[m] || {};
      // Skip if no data for this model+problem
      var hasData = false;
      for (var v of d.variants) { if ((vc[v]||[]).length) hasData = true; }
      if (!hasData) continue;

      h += '<tr><td style="text-align:left;font-size:10px;">' + esc(p) + '</td>';
      h += '<td style="font-size:10px;">' + esc(m) + '</td>';

      // Find best variant pass rate for highlighting
      var bestRate = 0;
      for (var v of d.variants) {
        var trials = vc[v] || [];
        if (trials.length) {
          var rate = trials.filter(function(t){return t.success;}).length / trials.length;
          if (rate > bestRate) bestRate = rate;
        }
      }

      for (var v of d.variants) {
        var trials = vc[v] || [];
        if (!trials.length) { h += '<td class="na">-</td>'; continue; }
        var passes = trials.filter(function(t){return t.success;}).length;
        var total = trials.length;
        var rate = passes/total;
        var cls = rate >= 0.6 ? 'pt-pass' : rate > 0 ? 'pt-partial' : 'pt-fail';
        if (rate === bestRate && rate > 0) cls += ' pt-best';
        var avgComp = trials.reduce(function(a,t){return a+t.n_compactions;},0) / total;
        h += '<td class="'+cls+'" style="cursor:pointer;" onclick="ptSel(\''+esc(p)+'\',\''+esc(m)+'\',\''+esc(v)+'\')">';
        h += passes+'/'+total + ' <small>('+avgComp.toFixed(1)+' comp)</small>';
        h += '</td>';
      }
      h += '</tr>';
    }
  }
  h += '</tbody></table>';
  document.getElementById('pt-table-pane').innerHTML = h;
}

function ptSel(pid, model, variant) {
  PT_ACTIVE = {pid:pid, model:model, variant:variant};
  PT_ACTIVE_TRIAL = 0;
  renderPTTrace();
}

function ptSelTrial(t) { PT_ACTIVE_TRIAL = t; renderPTTrace(); }

function renderPTTrace() {
  var a = PT_ACTIVE; if (!a) return;
  var trials = ((PT_DATA.cells[a.pid]||{})[a.model]||{})[a.variant] || [];
  var h = '<div class="trace-header">' + esc(a.model) + ' / ' + esc(a.pid) + ' / ' + esc(a.variant) + '</div>';

  // Summary bar
  var passes = trials.filter(function(t){return t.success;}).length;
  h += '<div class="stats-bar">';
  h += '<div class="stat"><span class="stat-label">Variant</span><span class="stat-value">'+esc(a.variant)+'</span></div>';
  h += '<div class="stat"><span class="stat-label">Pass Rate</span><span class="stat-value">'+passes+'/'+trials.length+'</span></div>';
  if (trials.length) {
    var w = trials[0].window;
    h += '<div class="stat"><span class="stat-label">Window</span><span class="stat-value">'+fmtK(w)+'</span></div>';
  }
  h += '</div>';

  // Trial tabs
  if (trials.length) {
    h += '<div class="tab-bar">';
    for (var i=0; i<trials.length; i++) {
      var tr = trials[i], tcls = tr.success ? 'pass-tab' : 'fail-tab';
      h += '<div class="tab '+tcls+(PT_ACTIVE_TRIAL===i?' active':'')+'" onclick="ptSelTrial('+i+')">Trial '+(i+1)+' '+(tr.success?'\u2713':'\u2717')+'</div>';
    }
    h += '</div>';

    var tr = trials[PT_ACTIVE_TRIAL];
    if (tr) {
      h += '<div class="meta" style="padding:8px 0;">Answer: <strong>'+esc(String(tr.answer))+'</strong> (correct: '+esc(String(tr.correct_answer))+') | Compactions: <strong>'+tr.n_compactions+'</strong> | Time: '+(tr.wall_time_s?tr.wall_time_s.toFixed(1)+'s':'\u2014')+' | '+tr.finish_reason+'</div>';
      h += renderConv(tr.conversation || []);
    }
  } else {
    h += '<div class="meta">No trials for this variant.</div>';
  }

  document.getElementById('pt-trace-pane').innerHTML = h;
}
</script>
</body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/results":
            data = load_results_cached()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif parsed.path == "/api/dashboard":
            data = load_dashboard_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif parsed.path == "/api/prompt_tuning":
            data = load_prompt_tuning_cached()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            self.send_error(404)
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Local viewer at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
