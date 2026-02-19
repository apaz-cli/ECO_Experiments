#!/usr/bin/env python3
"""
Sweep experiment visualizer.

Parses training logs from a sweep and serves an interactive browser UI
for exploring loss curves across experimental axes.

Usage:
    python visualize_experiment.py                      # newest experiment
    python visualize_experiment.py debug_smoke_...      # specific experiment
    python visualize_experiment.py --open-browser
    python visualize_experiment.py --port 43801
"""

import argparse
import importlib.util
import itertools
import json
import re
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).parent
SWEEPS_DIR = REPO_ROOT / "sweeps"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "sweeps"
ANSI_RE = re.compile(r"\033\[[0-9;]*m")
KV_RE = re.compile(r"\b(\w+):\s+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?|nan)\b")


# ── Data loading ──────────────────────────────────────────────────────────────

def find_sweep_dir(arg):
    """Return the sweep output directory for `arg`, defaulting to newest."""
    if arg is None:
        dirs = sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
        if not dirs:
            sys.exit(f"No experiments found in {OUTPUTS_DIR}")
        return dirs[-1]
    p = Path(arg)
    if p.is_dir():
        return p
    candidate = OUTPUTS_DIR / arg
    if candidate.is_dir():
        return candidate
    sys.exit(f"Experiment directory not found: {arg}")


def load_sweep_options(sweep_dir):
    """Import the sweep .py file and return (sweep_name, OPTIONS, EXCLUDE).

    The sweep name is inferred by stripping the _YYYYMMDD_HHMM timestamp suffix
    from the experiment directory name.
    """
    exp_name = sweep_dir.name
    sweep_name = re.sub(r"_\d{8}_\d{4}$", "", exp_name)
    sweep_file = SWEEPS_DIR / f"{sweep_name}.py"
    if not sweep_file.exists():
        sys.exit(f"Sweep file not found: {sweep_file}\n"
                 f"(inferred from experiment name: {exp_name})")
    spec = importlib.util.spec_from_file_location(sweep_name, sweep_file)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SWEEPS_DIR))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return sweep_name, mod.OPTIONS, getattr(mod, "EXCLUDE", None)


def generate_name_to_combo(sweep_name, options, exclude_fn):
    """Map run directory names → hyperparameter combos.

    Replicates run_sweep.py's naming logic: non-singular axes are sorted
    alphabetically first; singular (diagnostic) axes follow. Each axis with a
    "name" key contributes a segment like "lr0.001" or "muonT" to the dir name.
    """
    lex_keys  = sorted(k for k in options if not options[k].get("singular"))
    diag_keys = sorted(k for k in options if     options[k].get("singular"))
    all_keys  = lex_keys + diag_keys

    mapping = {}
    for vals in itertools.product(*(options[k]["values"] for k in all_keys)):
        combo = dict(zip(all_keys, vals))
        if exclude_fn and exclude_fn(combo):
            continue
        parts = [
            f"{options[k]['name']}{'T' if v is True else 'F' if v is False else v}"
            for k, v in combo.items() if options[k].get("name") is not None
        ]
        name = f"{sweep_name}_{'_'.join(parts) if parts else 'default'}"
        mapping[name] = combo
    return mapping


def parse_training_log(log_path):
    """Parse a training.log and return (steps, metrics).

    metrics maps metric name → list of float|None (one entry per logged step).
    Non-finite values (nan) are stored as None for valid JSON serialization.
    """
    steps, metrics = [], {}
    with open(log_path, errors="replace") as f:
        for line in f:
            if "step:" not in line:
                continue
            clean = ANSI_RE.sub("", line).replace(",", "")  # strip ANSI + thousand separators
            kvs = dict(KV_RE.findall(clean))
            if "step" not in kvs or "loss" not in kvs:
                continue
            try:
                step = int(float(kvs.pop("step")))
            except ValueError:
                continue
            steps.append(step)
            for k, v in kvs.items():
                try:
                    val = float(v)
                    metrics.setdefault(k, []).append(None if val != val else val)  # nan != nan
                except ValueError:
                    metrics.setdefault(k, []).append(None)
    return steps, metrics


def load_experiment(sweep_dir):
    """Parse all runs in a sweep directory and return the experiment data dict."""
    sweep_name, options, exclude_fn = load_sweep_options(sweep_dir)
    name_to_combo = generate_name_to_combo(sweep_name, options, exclude_fn)

    candidates = [
        d for d in sorted(sweep_dir.iterdir())
        if d.is_dir() and (d / "training.log").exists() and d.name in name_to_combo
    ]
    runs = []
    for i, run_dir in enumerate(candidates, 1):
        print(f"  Parsing run {i}/{len(candidates)}: {run_dir.name}", flush=True)
        steps, metrics = parse_training_log(run_dir / "training.log")
        if steps:
            runs.append({"name": run_dir.name, "combo": name_to_combo[run_dir.name],
                         "steps": steps, "metrics": metrics})

    return {
        "experiment": sweep_dir.name,
        "axes": {k: opts["values"] for k, opts in options.items()},
        "runs": runs,
    }


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

// Metrics shown first in the metric dropdown, in this order.
const PREFERRED_METRICS = ["loss", "grad_norm", "tflops", "mfu", "tps"];

let DATA        = null;  // experiment data loaded from /data.json
let colorBy     = null;  // axis name to color by (set on load)
let metric      = "loss";
let smoothAlpha = 0;     // EMA alpha; 0 = disabled
let filters     = {};    // axis → Set of visible values
let METRIC_NAMES = [];   // ordered metric names, populated by buildMetricSelect

// ── Theme ─────────────────────────────────────────────────────────────────────

(function() {
  const btn  = document.getElementById("theme-toggle");
  const root = document.documentElement;

  function applyTheme(dark) {
    dark ? root.setAttribute("data-theme", "dark") : root.removeAttribute("data-theme");
    btn.textContent = dark ? "☀" : "☾";
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
  }

  applyTheme(localStorage.getItem("theme") === "dark");

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
  const finals = runs.map(r => {
    const vals = r.metrics[metricName] || [];
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

// Hover label: all combo key=value pairs except the color axis.
function comboLabel(combo, skipAxis) {
  return Object.entries(combo)
    .filter(([k]) => k !== skipAxis)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
}

function hoverText(combo) {
  return Object.entries(combo).map(([k, v]) => `<b>${k}</b>: ${v}`).join("<br>");
}

function visibleRuns() {
  return DATA.runs.filter(r =>
    Object.entries(filters).every(([axis, vals]) => vals.has(r.combo[axis]))
  );
}

// Compute y-axis range with a small pad. Always uses ALL runs (not just visible)
// so the axis stays fixed when toggling checkboxes.
function yRangeOf(runs) {
  let lo = Infinity, hi = -Infinity;
  for (const r of runs)
    for (const v of (r.metrics[metric] || []))
      if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) return undefined;
  const pad = Math.max((hi - lo) * 0.05, 1e-6);
  return [lo - pad, hi + pad];
}

// ── Chart ─────────────────────────────────────────────────────────────────────

function buildChart() {
  const runs = visibleRuns();

  // Set up per-run color and legend-group helpers depending on color mode.
  // colorBy is either a plain axis name or "_m:<metric>" for metric coloring.
  const isMetricColor = colorBy.startsWith("_m:");
  let getColor, getGroup, showLegendFor, legendGroupTitle;

  if (isMetricColor) {
    const colorMap_ = metricColorsForRuns(DATA.runs, colorBy.slice(3));
    getColor       = r     => colorMap_.get(r.name) || "#888";
    getGroup       = r     => r.name;
    showLegendFor  = ()    => false;      // no discrete legend for continuous color
    legendGroupTitle = ()  => undefined;
  } else {
    const cmap     = colorMap(DATA.axes[colorBy] || []);
    const firstSeen = new Set();
    getColor       = r     => cmap[r.combo[colorBy]] || "#888";
    getGroup       = r     => String(r.combo[colorBy]);
    // showLegendFor has a side effect: it tracks which groups have been seen
    // so only the first trace per group gets a legend entry.
    showLegendFor  = group => { const f = !firstSeen.has(group); if (f) firstSeen.add(group); return f; };
    legendGroupTitle = (group, isFirst) => isFirst ? { text: colorBy, font: { size: 11 } } : undefined;
  }

  const traces = runs.map(r => {
    const color   = getColor(r);
    const group   = getGroup(r);
    const isFirst = showLegendFor(group);
    return {
      x: r.steps,
      y: ema(r.metrics[metric] || [], smoothAlpha),
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

  // Read theme colors from CSS variables so the chart matches light/dark mode.
  const cs = getComputedStyle(document.documentElement);
  const plotBg   = cs.getPropertyValue("--plot-bg").trim();
  const plotGrid = cs.getPropertyValue("--plot-grid").trim();
  const plotText = cs.getPropertyValue("--plot-text").trim();

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
  const cmap      = colorMap(DATA.axes[colorBy] || []);
  const container = document.getElementById("filters");
  container.innerHTML = "";

  for (const [axis, values] of Object.entries(DATA.axes)) {
    // Show color swatches next to checkboxes only for the active color axis.
    const isColorAxis = !colorBy.startsWith("_m:") && axis === colorBy;

    const group = document.createElement("div");
    group.className = "filter-group";

    // Header: axis label + all/none buttons
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
        group.querySelectorAll("input[type=checkbox]").forEach(cb => { cb.checked = action === "all"; });
        buildChart();
      };
      btns.appendChild(b);
    });
    header.appendChild(btns);
    group.appendChild(header);

    // Checkbox list
    const list = document.createElement("div");
    list.className = "checkbox-list";
    for (const val of values) {
      const row = document.createElement("label");
      row.className = "check-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = filters[axis]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[axis].add(val); else filters[axis].delete(val);
        buildChart();
      });
      row.appendChild(cb);

      if (isColorAxis) {
        const swatch = document.createElement("div");
        swatch.className = "color-swatch";
        swatch.style.background = cmap[val] || "#ccc";
        row.appendChild(swatch);
      }

      const lbl = document.createElement("span");
      lbl.className = "check-label";
      lbl.textContent = val;
      row.appendChild(lbl);
      list.appendChild(row);
    }

    group.appendChild(list);
    container.appendChild(group);
  }
}

function buildColorBySelect() {
  const container = document.getElementById("color-by");
  container.innerHTML = "";

  for (const axis of Object.keys(DATA.axes)) {
    const id    = "cbr-" + axis;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "color-by"; input.id = id; input.value = axis;

    const lbl = document.createElement("label");
    lbl.htmlFor = id;
    lbl.textContent = axis;
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
  const allMetrics = new Set();
  DATA.runs.forEach(r => Object.keys(r.metrics).forEach(k => allMetrics.add(k)));

  // Preferred metrics first, then any extras in discovery order.
  METRIC_NAMES = [
    ...PREFERRED_METRICS.filter(m => allMetrics.has(m)),
    ...[...allMetrics].filter(m => !PREFERRED_METRICS.includes(m)),
  ];

  const sel = document.getElementById("metric-sel");
  sel.innerHTML = "";
  for (const m of METRIC_NAMES) {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    if (m === metric) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.onchange = () => { metric = sel.value; buildChart(); };
}

// ── Init & loading ────────────────────────────────────────────────────────────

function init(data) {
  const expNameEl = document.getElementById("exp-name");
  DATA    = data;
  colorBy = Object.keys(DATA.axes)[0];

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

  expNameEl.textContent = "Building UI…";
  buildMetricSelect();
  buildColorBySelect();
  buildFilters();

  expNameEl.textContent = "Rendering chart…";
  buildChart();

  expNameEl.classList.remove("loading");
  expNameEl.textContent = data.experiment;
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


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    _cache = {}          # experiment name -> json bytes
    _default = None      # name of the pre-loaded experiment

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            body = HTML.encode()
            self._send(200, "text/html; charset=utf-8", body)

        elif parsed.path == "/experiments":
            names = sorted(
                (d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()),
                key=lambda n: (OUTPUTS_DIR / n).stat().st_mtime,
                reverse=True,
            )
            body = json.dumps({"experiments": names, "default": self._default}).encode()
            self._send(200, "application/json", body)

        elif parsed.path == "/data.json":
            qs = urllib.parse.parse_qs(parsed.query)
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            if name not in Handler._cache:
                sweep_dir = OUTPUTS_DIR / name
                if not sweep_dir.is_dir():
                    self.send_response(404); self.end_headers(); return
                print(f"Loading: {name}", flush=True)
                data = load_experiment(sweep_dir)
                print(f"  {len(data['runs'])} runs, axes: {list(data['axes'].keys())}", flush=True)
                Handler._cache[name] = json.dumps(data).encode()
            self._send(200, "application/json", Handler._cache[name])

        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="experiment name or directory (default: newest)")
    parser.add_argument("--port", type=int, default=43801)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    sweep_dir = find_sweep_dir(args.experiment)
    print(f"Loading: {sweep_dir.name}")
    data = load_experiment(sweep_dir)
    print(f"  {len(data['runs'])} runs, axes: {list(data['axes'].keys())}")

    Handler._cache[sweep_dir.name] = json.dumps(data).encode()
    Handler._default = sweep_dir.name

    url = f"http://localhost:{args.port}"
    print(f"  Serving at {url}  (Ctrl+C to stop)")

    if args.open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    HTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
