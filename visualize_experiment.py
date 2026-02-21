#!/usr/bin/env python3
"""
Sweep experiment visualizer.

Loads training metrics from Aim and serves an interactive browser UI
for exploring loss curves across experimental axes.

Usage:
    python visualize_experiment.py                      # newest experiment
    python visualize_experiment.py debug_smoke_...      # specific experiment
    python visualize_experiment.py --open-browser
    python visualize_experiment.py --port 43801
    python visualize_experiment.py --aim-repo aim://host:port  # remote server
"""

import argparse
import json
import math
import os
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import aim as aim_sdk
from aim.sdk.types import QueryReportMode

# Default to the local .aim directory (direct RocksDB reads, no HTTP overhead).
# Pass --aim-repo aim://host:port to use a remote server instead.
DEFAULT_AIM_REPO = os.path.dirname(os.path.abspath(__file__))


# ── Aim data loading ───────────────────────────────────────────────────────────

def _parse_tag_value(s):
    """Convert a tag value string back to a typed Python value."""
    if s == "True":  return True
    if s == "False": return False
    try:    return int(s)
    except ValueError: pass
    try:    return float(s)
    except ValueError: pass
    return s


def list_experiments(repo_url):
    """Return all Aim experiment names, sorted newest-first."""
    repo = aim_sdk.Repo(repo_url)
    latest = {}  # experiment name -> newest created_at seen
    for r in repo.query_runs('').iter_runs():
        run = r.run
        if run.experiment:
            t = run.created_at
            if run.experiment not in latest or t > latest[run.experiment]:
                latest[run.experiment] = t
    return sorted(latest, key=lambda e: latest[e], reverse=True)


def load_experiment_meta(experiment_name, repo_url):
    """Return axes, run combos, and metric names — no metric data loaded.

    Fast (a few seconds) because it only reads run metadata and metric names,
    not the actual time-series values.  Metric data is fetched on demand via
    load_metric_data().

    Returns {"experiment", "axes", "runs", "metricNames"}.  Each run contains
    {"name", "hash", "combo"}.  Combos are reconstructed from Aim tags
    ("key=value" strings set by run_sweep.py).
    """
    repo = aim_sdk.Repo(repo_url)
    aim_runs = [r.run for r in repo.query_runs('').iter_runs() if r.run.experiment == experiment_name]

    axis_values = {}
    runs        = []
    metric_names = set()

    for run in aim_runs:
        combo = {}
        for tag in run.tags:
            if '=' in tag:
                k, v = tag.split('=', 1)
                combo[k] = _parse_tag_value(v)
        for k, v in combo.items():
            axis_values.setdefault(k, set()).add(v)
        runs.append({"name": run.name, "hash": run.hash, "combo": combo})
        # Collect metric names only — no .dataframe() calls, so this stays fast.
        for metric in run.metrics():
            if not metric.name.startswith('__'):
                metric_names.add(metric.name)

    def val_sort_key(v):
        if isinstance(v, bool):         return (0, str(v))
        if isinstance(v, (int, float)): return (1, v)
        return                                  (2, str(v))

    axes = {k: sorted(vs, key=val_sort_key) for k, vs in axis_values.items()}

    # Detect sub-axes: axes that only appear in runs where some other axis has a specific value.
    # e.g. "approach" only appears when "optimizer"="muon".
    all_hashes = {r["hash"] for r in runs}
    hashes_with = {ax: {r["hash"] for r in runs if ax in r["combo"]} for ax in axes}
    sub_axes = {}
    for axis in axes:
        if hashes_with[axis] == all_hashes:
            continue  # universal axis
        for parent_axis in axes:
            if parent_axis == axis:
                continue
            for parent_val in axes[parent_axis]:
                hashes_with_parent = {r["hash"] for r in runs if r["combo"].get(parent_axis) == parent_val}
                if hashes_with_parent == hashes_with[axis]:
                    sub_axes[axis] = {"parentAxis": parent_axis, "parentValue": parent_val}
                    break
            if axis in sub_axes:
                break

    return {"experiment": experiment_name, "axes": axes, "runs": runs,
            "metricNames": sorted(metric_names), "subAxes": sub_axes}


def load_metric_data(experiment_name, metric_name, repo_url):
    """Load one metric's values for all runs in an experiment (~1–2 s).

    Uses Aim's AQL query to filter to a single metric name, skipping all
    other metrics.  Returns {run_hash: {"steps": [...], "values": [...]}}.
    """
    repo = aim_sdk.Repo(repo_url)
    q = repo.query_metrics(
        f'run.experiment == "{experiment_name}" and metric.name == "{metric_name}"',
        report_mode=QueryReportMode.DISABLED,
    )
    result = {}
    for metric in q.iter():
        df = metric.dataframe()
        if df.empty:
            continue
        values = [None if (isinstance(v, float) and math.isnan(v)) else v
                  for v in df['value'].tolist()]
        result[metric.run.hash] = {"steps": df['step'].tolist(), "values": values}
    return result


# ── HTML / JS ─────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sweep Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f5f5f5; --surface: #fff; --border: #ddd; --border-input: #ccc;
  --text: #222; --text-muted: #999; --text-dim: #888; --text-label: #555; --text-strong: #333;
  --btn-bg: white; --btn-hover: #f0f0f0; --btn-text: #666;
  --radio-bg: #fafafa; --radio-checked-bg: #e8f0fe;
  --radio-checked-border: #4c9be8; --radio-checked-color: #1a56c4;
  --plot-bg: white; --plot-grid: #eee; --plot-text: #444;
  --swatch-border: rgba(0,0,0,0.15);
}
[data-theme=dark] {
  --bg: #1c1c1c; --surface: #252525; --border: #383838; --border-input: #444;
  --text: #ddd; --text-muted: #666; --text-dim: #777; --text-label: #aaa; --text-strong: #ccc;
  --btn-bg: #2e2e2e; --btn-hover: #383838; --btn-text: #aaa;
  --radio-bg: #2e2e2e; --radio-checked-bg: #1e2f50;
  --radio-checked-border: #4c9be8; --radio-checked-color: #82b4f0;
  --plot-bg: #1e1e1e; --plot-grid: #2e2e2e; --plot-text: #aaa;
  --swatch-border: rgba(255,255,255,0.15);
}

body { font-family: system-ui, -apple-system, sans-serif; display: flex;
       height: 100vh; background: var(--bg); color: var(--text); }

#sidebar { width: 270px; min-width: 270px; background: var(--surface);
           border-right: 1px solid var(--border); overflow-y: auto;
           padding: 14px 12px; display: flex; flex-direction: column; gap: 16px; }

#main { flex: 1; display: flex; flex-direction: column; min-width: 0; padding: 12px; gap: 8px; }

#chart { flex: 1; min-height: 0; border: 1px solid var(--border);
         border-radius: 6px; background: var(--plot-bg); }

h1 { font-size: 13px; font-weight: 700; color: var(--text-strong); }
.exp-name { font-size: 10px; color: var(--text-muted); word-break: break-all; margin-top: 2px; }

@keyframes rainbow-sweep {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
.exp-name.loading {
  background: linear-gradient(90deg, #f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00,#ff0,#0f0);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
  font-size: 12px;
  animation: rainbow-sweep 1s linear infinite;
}

.section { display: flex; flex-direction: column; gap: 6px; }
.section-title { font-size: 10px; font-weight: 700; color: var(--text-dim);
                 text-transform: uppercase; letter-spacing: 0.06em; }

select { width: 100%; padding: 5px 7px; border: 1px solid var(--border-input);
         border-radius: 4px; font-size: 12px; background: var(--btn-bg); color: var(--text); }
select:focus { outline: none; border-color: #4c9be8; }

.radio-row { display: flex; flex-wrap: wrap; gap: 3px; }
.radio-row input[type=radio] { display: none; }
.radio-row label { font-size: 11px; padding: 2px 7px; border: 1px solid var(--border-input);
                   border-radius: 3px; cursor: pointer; background: var(--radio-bg);
                   white-space: nowrap; color: var(--text); }
.radio-row label:hover { background: var(--btn-hover); }
.radio-row label.checked { background: var(--radio-checked-bg); border-color: var(--radio-checked-border);
                           color: var(--radio-checked-color); font-weight: 600; }

.filter-group { display: flex; flex-direction: column; gap: 4px; }
.filter-header { display: flex; justify-content: space-between; align-items: center; }
.filter-header .axis-label { font-size: 11px; font-weight: 600; color: var(--text-label); }
.filter-btns { display: flex; gap: 3px; }
.filter-btns button { font-size: 9px; padding: 1px 5px; border: 1px solid var(--border-input);
                      background: var(--btn-bg); border-radius: 3px; cursor: pointer; color: var(--btn-text); }
.filter-btns button:hover { background: var(--btn-hover); }

.checkbox-list { display: flex; flex-direction: column; gap: 3px; padding-left: 2px; }
.check-row { display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer;
             padding: 1px 0; }
.check-row input[type=checkbox] { cursor: pointer; width: 13px; height: 13px; }
.color-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
                border: 1px solid var(--swatch-border); }
.check-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.sub-filter-group { margin-top: 4px; margin-left: 10px; padding-left: 4px;
                    border-left: 2px solid var(--border); }
.sub-filter-group .axis-label { font-size: 10px; color: var(--text-dim); }
.sub-filter-group.hidden { display: none; }

.row { display: flex; gap: 8px; align-items: center; }
.row label { font-size: 12px; white-space: nowrap; }
.row input[type=range] { flex: 1; }
#smooth-val { font-size: 11px; color: var(--text-dim); min-width: 28px; text-align: right; }

#statusbar { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
#status { font-size: 11px; color: var(--text-muted); }

#theme-toggle { background: none; border: none; font-size: 16px; cursor: pointer;
                color: var(--text-dim); padding: 0; line-height: 1; flex-shrink: 0; }
#theme-toggle:hover { color: var(--text); }
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>Sweep Visualizer</h1>
    <div class="exp-name loading" id="exp-name">Fetching data…</div>
  </div>

  <div class="section">
    <div class="section-title">Experiment</div>
    <select id="experiment-sel"></select>
  </div>

  <div class="section">
    <div class="section-title">Metric</div>
    <select id="metric-sel"></select>
  </div>

  <div class="section">
    <div class="section-title">Color by</div>
    <div id="color-by" class="radio-row"></div>
  </div>

  <div class="section">
    <div class="section-title">Smoothing (EMA)</div>
    <div class="row">
      <input type="range" id="smoothing" min="0" max="0.99" step="0.01" value="0">
      <span id="smooth-val">0.00</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Filters</div>
    <div id="filters"></div>
  </div>
</div>

<div id="main">
  <div id="statusbar">
    <div id="status"></div>
    <button id="theme-toggle" title="Switch to dark mode">☾</button>
  </div>
  <div id="chart"></div>
</div>

<script>
// ── Constants & state ─────────────────────────────────────────────────────────

const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
];

let DATA         = null;  // experiment metadata from /data.json
let METRIC_CACHE = {};    // metric_name → {run_hash: {steps, values}}, loaded on demand
let colorBy      = null;  // axis name to color by (set on load)
let metric       = "loss";
let smoothAlpha  = 0;     // EMA alpha; 0 = disabled
let filters      = {};    // axis → Set of visible values
let METRIC_NAMES = [];    // populated by buildMetricSelect

// ── Theme ─────────────────────────────────────────────────────────────────────

(function() {
  const btn  = document.getElementById("theme-toggle");
  const root = document.documentElement;

  function applyTheme(dark) {
    dark ? root.setAttribute("data-theme", "dark") : root.removeAttribute("data-theme");
    btn.textContent = dark ? "☀" : "☾";
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
  }

  applyTheme(localStorage.getItem("theme") !== "light");

  btn.addEventListener("click", () => {
    const dark = root.getAttribute("data-theme") !== "dark";
    localStorage.setItem("theme", dark ? "dark" : "light");
    applyTheme(dark);
    if (DATA) buildChart();  // re-render so Plotly picks up new CSS colors
  });
})();

// ── Helpers ───────────────────────────────────────────────────────────────────

// Exponential moving average; treats null/non-finite values as gaps.
function ema(vals, alpha) {
  if (alpha === 0) return vals;
  const out = [];
  let s = null;
  for (const v of vals) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : alpha * s + (1 - alpha) * v;
    out.push(s);
  }
  return out;
}

// Map an array of axis values to palette colors.
function colorMap(axisValues) {
  const m = {};
  axisValues.forEach((v, i) => { m[v] = PALETTE[i % PALETTE.length]; });
  return m;
}

// Interpolate blue (#1565C0) → red (#C62828) for metric-based coloring.
function lerpColor(t) {
  const r = Math.round(0x15 + (0xC6 - 0x15) * t);
  const g = Math.round(0x65 + (0x28 - 0x65) * t);
  const b = Math.round(0xC0 + (0x28 - 0xC0) * t);
  return `rgb(${r},${g},${b})`;
}

// Return a run-name → color map based on each run's final value of metricName.
// Colors are scaled relative to all runs (not just visible ones) for consistency.
function metricColorsForRuns(runs, metricName) {
  const cache = METRIC_CACHE[metricName] || {};
  const finals = runs.map(r => {
    const vals = (cache[r.hash] || {}).values || [];
    for (let i = vals.length - 1; i >= 0; i--)
      if (vals[i] !== null && isFinite(vals[i])) return vals[i];
    return null;
  });
  const finite = finals.filter(v => v !== null);
  if (!finite.length) return new Map(runs.map(r => [r.name, "#888"]));
  const lo = Math.min(...finite), hi = Math.max(...finite);
  return new Map(runs.map((r, i) => {
    const v = finals[i];
    const t = v === null ? 0.5 : (hi === lo ? 0.5 : (v - lo) / (hi - lo));
    return [r.name, lerpColor(t)];
  }));
}

// Iterate combo entries in axis order (matches sidebar), skipping missing keys.
function orderedComboEntries(combo, skipAxis) {
  return Object.keys(DATA.axes)
    .filter(k => k !== skipAxis && k in combo)
    .map(k => [k, combo[k]]);
}

// Hover label: all combo key=value pairs except the color axis.
function comboLabel(combo, skipAxis) {
  return orderedComboEntries(combo, skipAxis)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
}

function hoverText(combo) {
  return orderedComboEntries(combo, null)
    .map(([k, v]) => `<b>${k}</b>: ${v}`)
    .join("<br>");
}

function visibleRuns() {
  return DATA.runs.filter(r =>
    Object.entries(filters).every(([axis, vals]) => !(axis in r.combo) || vals.has(r.combo[axis]))
  );
}

// Compute y-axis range with a small pad. Always uses ALL runs (not just visible)
// so the axis stays fixed when toggling checkboxes.
function yRangeOf(runs) {
  const cache = METRIC_CACHE[metric] || {};
  let lo = Infinity, hi = -Infinity;
  for (const r of runs)
    for (const v of ((cache[r.hash] || {}).values || []))
      if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) return undefined;
  const pad = Math.max((hi - lo) * 0.05, 1e-6);
  return [lo - pad, hi + pad];
}

// ── Chart ─────────────────────────────────────────────────────────────────────

function buildChart() {
  let runs = visibleRuns();

  // When coloring by a sub-axis, hide runs that don't have that axis.
  const subAxisInfo = DATA.subAxes || {};
  if (!colorBy.startsWith("_m:") && subAxisInfo[colorBy])
    runs = runs.filter(r => colorBy in r.combo);

  // Set up per-run color and legend-group helpers depending on color mode.
  // colorBy is either a plain axis name or "_m:<metric>" for metric coloring.
  const isMetricColor = colorBy.startsWith("_m:");
  let getColor, getGroup, showLegendFor, legendGroupTitle;

  if (isMetricColor) {
    const colorMap_ = metricColorsForRuns(DATA.runs, colorBy.slice(3));
    getColor        = r     => colorMap_.get(r.name) || "#888";
    getGroup        = r     => r.name;
    showLegendFor   = ()    => false;      // no discrete legend for continuous color
    legendGroupTitle = ()   => undefined;
  } else {
    const cmap      = colorMap(DATA.axes[colorBy] || []);
    const firstSeen = new Set();
    getColor  = r => cmap[r.combo[colorBy]] || "#888";
    getGroup  = r => String(r.combo[colorBy]);
    // showLegendFor tracks which groups have been seen so only the first
    // trace per group gets a legend entry.
    showLegendFor   = group => { const f = !firstSeen.has(group); if (f) firstSeen.add(group); return f; };
    legendGroupTitle = (group, isFirst) => isFirst ? { text: colorBy, font: { size: 11 } } : undefined;
  }

  // Read theme colors up front (needed for dot border color).
  const cs = getComputedStyle(document.documentElement);
  const plotBg   = cs.getPropertyValue("--plot-bg").trim();
  const plotGrid = cs.getPropertyValue("--plot-grid").trim();
  const plotText = cs.getPropertyValue("--plot-text").trim();

  const mcache = METRIC_CACHE[metric] || {};
  const lineTraces = runs.map(r => {
    const color   = getColor(r);
    const group   = getGroup(r);
    const isFirst = showLegendFor(group);
    const mdata   = mcache[r.hash] || {};
    return {
      x: mdata.steps  || [],
      y: ema(mdata.values || [], smoothAlpha),
      mode: "lines",
      line: { color, width: 1.5 },
      name: group,
      legendgroup: group,
      legendgrouptitle: legendGroupTitle(group, isFirst),
      showlegend: isFirst,
      hovertemplate: hoverText(r.combo) + "<br><b>step</b>: %{x}<br><b>" + metric + "</b>: %{y:.4f}<extra></extra>",
      customdata: [comboLabel(r.combo, colorBy)],
    };
  });

  // One dot per run at the final data point.
  const dotTraces = runs.map(r => {
    const color = getColor(r);
    const group = getGroup(r);
    const mdata = mcache[r.hash] || {};
    const steps  = mdata.steps  || [];
    const vals   = ema(mdata.values || [], smoothAlpha);
    const lastX  = steps.length  ? [steps[steps.length - 1]]  : [];
    const lastY  = vals.length   ? [vals[vals.length - 1]]    : [];
    return {
      x: lastX, y: lastY,
      mode: "markers",
      marker: { color, size: 6, line: { color: plotBg, width: 1.5 } },
      legendgroup: group,
      showlegend: false,
      hoverinfo: "skip",
    };
  });

  const traces = [...lineTraces, ...dotTraces];

  Plotly.react("chart", traces, {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    xaxis: { title: "step", gridcolor: plotGrid, color: plotText },
    yaxis: { title: metric, gridcolor: plotGrid, color: plotText,
             range: yRangeOf(DATA.runs), autorange: yRangeOf(DATA.runs) === undefined },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { groupclick: "toggleitem", font: { size: 11, color: plotText } },
    hovermode: "closest",
  }, { responsive: true, scrollZoom: true });

  document.getElementById("status").textContent =
    `${runs.length} / ${DATA.runs.length} runs visible`;
}

// ── Sidebar builders ──────────────────────────────────────────────────────────

function buildFilters() {
  const subAxisInfo = DATA.subAxes || {};
  const subAxisSet  = new Set(Object.keys(subAxisInfo));

  // childrenOf["parentAxis:parentValue"] = [childAxis, ...]
  const childrenOf = {};
  for (const [axis, info] of Object.entries(subAxisInfo)) {
    const key = `${info.parentAxis}:${info.parentValue}`;
    (childrenOf[key] = childrenOf[key] || []).push(axis);
  }

  const container = document.getElementById("filters");
  container.innerHTML = "";

  // Build a filter-group div for a given axis (top-level or sub).
  function makeFilterGroup(axis, values, extraClass) {
    const isColorAxis = !colorBy.startsWith("_m:") && axis === colorBy;
    const cmap_       = isColorAxis ? colorMap(values) : {};

    const group = document.createElement("div");
    group.className = "filter-group" + (extraClass ? " " + extraClass : "");

    const header = document.createElement("div");
    header.className = "filter-header";
    header.innerHTML = `<span class="axis-label">${axis}</span>`;
    const btns = document.createElement("div");
    btns.className = "filter-btns";
    ["all", "none"].forEach(action => {
      const b = document.createElement("button");
      b.textContent = action;
      b.onclick = () => {
        filters[axis] = action === "all" ? new Set(values) : new Set();
        group.querySelectorAll(`input[data-axis="${CSS.escape(axis)}"]`)
             .forEach(cb => { cb.checked = action === "all"; });
        buildChart();
      };
      btns.appendChild(b);
    });
    header.appendChild(btns);
    group.appendChild(header);

    const list = document.createElement("div");
    list.className = "checkbox-list";
    for (const val of values) {
      const row = document.createElement("label");
      row.className = "check-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.dataset.axis = axis;
      cb.checked = filters[axis]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[axis].add(val); else filters[axis].delete(val);
        buildChart();
      });
      row.appendChild(cb);

      if (isColorAxis) {
        const swatch = document.createElement("div");
        swatch.className = "color-swatch";
        swatch.style.background = cmap_[val] || "#ccc";
        row.appendChild(swatch);
      }

      const lbl = document.createElement("span");
      lbl.className = "check-label";
      lbl.textContent = val;
      row.appendChild(lbl);
      list.appendChild(row);

      // Nest any child axes under this value, hidden when the value is unchecked.
      for (const childAxis of (childrenOf[`${axis}:${val}`] || [])) {
        const childGroup = makeFilterGroup(childAxis, DATA.axes[childAxis] || [], "sub-filter-group");
        if (!cb.checked) childGroup.classList.add("hidden");
        cb.addEventListener("change", () => childGroup.classList.toggle("hidden", !cb.checked));
        list.appendChild(childGroup);
      }
    }

    group.appendChild(list);
    return group;
  }

  for (const [axis, values] of Object.entries(DATA.axes)) {
    if (subAxisSet.has(axis)) continue;  // rendered inline under parent
    container.appendChild(makeFilterGroup(axis, values, null));
  }
}

function buildColorBySelect() {
  const container   = document.getElementById("color-by");
  const subAxisInfo = DATA.subAxes || {};
  container.innerHTML = "";

  for (const axis of Object.keys(DATA.axes)) {
    const id    = "cbr-" + axis;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "color-by"; input.id = id; input.value = axis;

    const lbl = document.createElement("label");
    lbl.htmlFor = id;
    const info = subAxisInfo[axis];
    lbl.textContent = info ? `${axis} (${info.parentValue})` : axis;
    if (axis === colorBy) lbl.classList.add("checked");

    input.addEventListener("change", () => {
      container.querySelectorAll("label").forEach(l => l.classList.remove("checked"));
      lbl.classList.add("checked");
      colorBy = axis;
      buildFilters();
      buildChart();
    });

    container.appendChild(input);
    container.appendChild(lbl);
  }
}

function buildMetricSelect() {
  METRIC_NAMES = DATA.metricNames;

  const sel = document.getElementById("metric-sel");
  sel.innerHTML = "";
  for (const m of METRIC_NAMES) {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    if (m === metric) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.onchange = () => loadMetric(sel.value);
}

// ── Init & loading ────────────────────────────────────────────────────────────

// Load one metric's data from the server (cached after first fetch).
// Updates the chart once the data arrives.
function loadMetric(name) {
  const expNameEl = document.getElementById("exp-name");

  // Already cached — switch instantly.
  if (METRIC_CACHE[name]) {
    metric = name;
    buildChart();
    return;
  }

  const expName = document.getElementById("experiment-sel").value;
  expNameEl.classList.add("loading");
  expNameEl.textContent = `Loading ${name}…`;

  fetch(`/metric.json?name=${encodeURIComponent(expName)}&metric=${encodeURIComponent(name)}`)
    .then(r => r.json())
    .then(data => {
      METRIC_CACHE[name] = data;
      metric = name;
      expNameEl.classList.remove("loading");
      expNameEl.textContent = DATA.experiment;
      buildChart();
    })
    .catch(e => {
      expNameEl.classList.remove("loading");
      expNameEl.textContent = DATA.experiment;
      document.getElementById("status").textContent = "Error loading metric: " + e;
    });
}

function init(data) {
  DATA         = data;
  METRIC_CACHE = {};   // clear cached metric data on experiment switch
  colorBy      = Object.keys(DATA.axes)[0];

  // All values visible by default.
  filters = Object.fromEntries(
    Object.entries(DATA.axes).map(([axis, vals]) => [axis, new Set(vals)])
  );

  // Wire up the smoothing slider once (init is called again on experiment switch).
  if (!init._smoothingWired) {
    init._smoothingWired = true;
    document.getElementById("smoothing").addEventListener("input", e => {
      smoothAlpha = parseFloat(e.target.value);
      document.getElementById("smooth-val").textContent = smoothAlpha.toFixed(2);
      buildChart();
    });
  }

  // Pick initial metric BEFORE building the select so the dropdown reflects it.
  metric = DATA.metricNames.includes(metric) ? metric
    : (DATA.metricNames.find(m => m.includes("loss")) || DATA.metricNames[0]);

  buildMetricSelect();
  buildColorBySelect();
  buildFilters();

  loadMetric(metric);
}

function loadExperiment(name) {
  const expNameEl = document.getElementById("exp-name");
  expNameEl.classList.add("loading");
  expNameEl.textContent = "Fetching data…";

  fetch(`/data.json?name=${encodeURIComponent(name)}`)
    .then(r => { expNameEl.textContent = "Parsing JSON…"; return r.json(); })
    .then(data => {
      metric = "loss";  // reset metric on experiment switch
      init(data);
    })
    .catch(e => {
      expNameEl.classList.remove("loading");
      expNameEl.textContent = "Error";
      document.getElementById("status").textContent = "Error loading data: " + e;
    });
}

// Bootstrap: fetch experiment list, populate dropdown, then load the default.
fetch("/experiments")
  .then(r => r.json())
  .then(({ experiments, default: def }) => {
    const sel = document.getElementById("experiment-sel");
    for (const name of experiments) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      if (name === def) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.onchange = () => loadExperiment(sel.value);
    loadExperiment(def);
  })
  .catch(e => {
    document.getElementById("status").textContent = "Error fetching experiments: " + e;
  });
</script>
</body>
</html>
"""


# ── HTTP server ────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    _meta_cache   = {}  # experiment name -> json bytes (axes + combos + metric names)
    _metric_cache = {}  # (experiment, metric_name) -> json bytes
    _default      = None
    aim_repo      = DEFAULT_AIM_REPO

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs     = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.encode())

        elif parsed.path == "/experiments":
            names = list_experiments(self.aim_repo)
            body  = json.dumps({"experiments": names, "default": self._default}).encode()
            self._send(200, "application/json", body)

        elif parsed.path == "/data.json":
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            if name not in Handler._meta_cache:
                print(f"Loading metadata: {name}", flush=True)
                data = load_experiment_meta(name, self.aim_repo)
                print(f"  {len(data['runs'])} runs, {len(data['axes'])} axes, "
                      f"{len(data['metricNames'])} metrics", flush=True)
                Handler._meta_cache[name] = json.dumps(data).encode()
            self._send(200, "application/json", Handler._meta_cache[name])

        elif parsed.path == "/metric.json":
            name   = qs.get("name",   [None])[0]
            metric = qs.get("metric", [None])[0]
            if not name or not metric:
                self.send_response(400); self.end_headers(); return
            key = (name, metric)
            if key not in Handler._metric_cache:
                print(f"  Loading metric '{metric}' for {name}…", flush=True)
                data = load_metric_data(name, metric, self.aim_repo)
                Handler._metric_cache[key] = json.dumps(data).encode()
            self._send(200, "application/json", Handler._metric_cache[key])

        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?",
                        help="Aim experiment name to pre-load (default: newest)")
    parser.add_argument("--aim-repo", default=DEFAULT_AIM_REPO,
                        help="Path to local .aim repo dir, or aim://host:port for remote "
                             f"(default: script directory)")
    parser.add_argument("--port", type=int, default=43801)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    Handler.aim_repo = args.aim_repo

    # Find and pre-load the default experiment so the first page load is fast.
    experiments = list_experiments(args.aim_repo)
    if not experiments:
        sys.exit(f"No experiments found in Aim repo: {args.aim_repo}")

    default_exp = args.experiment or experiments[0]
    if default_exp not in experiments:
        sys.exit(f"Experiment not found: {default_exp}")

    print(f"Loading metadata: {default_exp}")
    data = load_experiment_meta(default_exp, args.aim_repo)
    print(f"  {len(data['runs'])} runs, {len(data['axes'])} axes, "
          f"{len(data['metricNames'])} metrics")

    Handler._meta_cache[default_exp] = json.dumps(data).encode()
    Handler._default = default_exp

    url = f"http://localhost:{args.port}"
    print(f"  Serving at {url}  (Ctrl+C to stop)")

    if args.open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    HTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
