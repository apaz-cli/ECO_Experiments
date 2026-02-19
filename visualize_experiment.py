#!/usr/bin/env python3
"""
Sweep experiment visualizer.

Parses training logs from a sweep and serves an interactive browser UI
for exploring loss curves across experimental axes.

Usage:
    python visualize_experiment.py                      # newest experiment
    python visualize_experiment.py debug_smoke_...      # specific experiment
    python visualize_experiment.py --port 43801
"""

import argparse
import importlib.util
import itertools
import json
import re
import sys
import threading
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
    """Replicate run_sweep.py naming logic to map run dir names → combos.

    run_sweep.py sorts keys alphabetically (lex_keys = sorted(...)), so names
    are built in alphabetical axis order regardless of OPTIONS dict order.
    """
    lex_keys = sorted(k for k in options if not options[k].get("singular"))
    diag_keys = sorted(k for k in options if options[k].get("singular"))
    all_keys = lex_keys + diag_keys

    mapping = {}
    for vals in itertools.product(*(options[k]["values"] for k in all_keys)):
        combo = dict(zip(all_keys, vals))
        if exclude_fn and exclude_fn(combo):
            continue
        parts = []
        for key in all_keys:
            val = combo[key]
            nm = options[key].get("name")
            if nm is not None:
                parts.append(f"{nm}{'T' if val is True else 'F' if val is False else val}")
        name = f"{sweep_name}_{'_'.join(parts) if parts else 'default'}"
        mapping[name] = combo
    return mapping


def parse_training_log(log_path):
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
                    metrics.setdefault(k, []).append(float(v))
                except ValueError:
                    metrics.setdefault(k, []).append(float("nan"))
    return steps, metrics


def load_experiment(sweep_dir):
    sweep_name, options, exclude_fn = load_sweep_options(sweep_dir)
    name_to_combo = generate_name_to_combo(sweep_name, options, exclude_fn)

    runs = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log = run_dir / "training.log"
        if not log.exists():
            continue
        combo = name_to_combo.get(run_dir.name)
        if combo is None:
            continue
        steps, metrics = parse_training_log(log)
        if not steps:
            continue
        runs.append({"name": run_dir.name, "combo": combo, "steps": steps, "metrics": metrics})

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
body { font-family: system-ui, -apple-system, sans-serif; display: flex;
       height: 100vh; background: #f5f5f5; color: #222; }

#sidebar { width: 270px; min-width: 270px; background: #fff;
           border-right: 1px solid #ddd; overflow-y: auto;
           padding: 14px 12px; display: flex; flex-direction: column; gap: 16px; }

#main { flex: 1; display: flex; flex-direction: column; min-width: 0; padding: 12px; gap: 8px; }

#chart { flex: 1; min-height: 0; border: 1px solid #ddd;
         border-radius: 6px; background: white; }

h1 { font-size: 13px; font-weight: 700; color: #333; }
.exp-name { font-size: 10px; color: #999; word-break: break-all; margin-top: 2px; }

.section { display: flex; flex-direction: column; gap: 6px; }
.section-title { font-size: 10px; font-weight: 700; color: #888;
                 text-transform: uppercase; letter-spacing: 0.06em; }

select { width: 100%; padding: 5px 7px; border: 1px solid #ccc;
         border-radius: 4px; font-size: 12px; background: white; }
select:focus { outline: none; border-color: #4c9be8; }

.filter-group { display: flex; flex-direction: column; gap: 4px; }
.filter-header { display: flex; justify-content: space-between; align-items: center; }
.filter-header .axis-label { font-size: 11px; font-weight: 600; color: #555; }
.filter-btns { display: flex; gap: 3px; }
.filter-btns button { font-size: 9px; padding: 1px 5px; border: 1px solid #ccc;
                      background: white; border-radius: 3px; cursor: pointer; color: #666; }
.filter-btns button:hover { background: #f0f0f0; }

.checkbox-list { display: flex; flex-direction: column; gap: 3px; padding-left: 2px; }
.check-row { display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer;
             padding: 1px 0; }
.check-row input[type=checkbox] { cursor: pointer; width: 13px; height: 13px; }
.color-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
                border: 1px solid rgba(0,0,0,0.15); }
.check-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.row { display: flex; gap: 8px; align-items: center; }
.row label { font-size: 12px; white-space: nowrap; }
.row input[type=range] { flex: 1; }
#smooth-val { font-size: 11px; color: #888; min-width: 28px; text-align: right; }

#status { font-size: 11px; color: #999; text-align: right; }
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>Sweep Visualizer</h1>
    <div class="exp-name" id="exp-name">Loading…</div>
  </div>

  <div class="section">
    <div class="section-title">Color by</div>
    <select id="color-by"></select>
  </div>

  <div class="section">
    <div class="section-title">Metric</div>
    <select id="metric-sel"></select>
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
  <div id="status"></div>
  <div id="chart"></div>
</div>

<script>
const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
];

let DATA = null;       // loaded from /data.json
let colorBy = null;
let metric = "loss";
let smoothAlpha = 0;
let filters = {};      // axis -> Set of visible values

function ema(vals, alpha) {
  if (alpha === 0) return vals;
  const out = [];
  let s = null;
  for (const v of vals) {
    if (!isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : alpha * s + (1 - alpha) * v;
    out.push(s);
  }
  return out;
}

function colorMap(axisValues) {
  const m = {};
  axisValues.forEach((v, i) => { m[v] = PALETTE[i % PALETTE.length]; });
  return m;
}

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

function yRangeOf(runs) {
  let lo = Infinity, hi = -Infinity;
  for (const r of runs) {
    for (const v of (r.metrics[metric] || [])) {
      if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  if (!isFinite(lo)) return undefined;
  const pad = Math.max((hi - lo) * 0.05, 1e-6);
  return [lo - pad, hi + pad];
}

function buildChart() {
  const runs = visibleRuns();
  const cmap = colorMap(DATA.axes[colorBy] || []);
  const firstSeen = new Set();

  const traces = runs.map(r => {
    const color = cmap[r.combo[colorBy]] || "#888";
    const group = String(r.combo[colorBy]);
    const isFirst = !firstSeen.has(group);
    if (isFirst) firstSeen.add(group);

    const yRaw = r.metrics[metric] || [];
    const y = ema(yRaw, smoothAlpha);

    return {
      x: r.steps,
      y,
      mode: "lines",
      line: { color, width: 1.5 },
      name: group,
      legendgroup: group,
      legendgrouptitle: isFirst ? { text: colorBy, font: { size: 11 } } : undefined,
      showlegend: isFirst,
      hovertemplate: hoverText(r.combo) + "<br><b>step</b>: %{x}<br><b>" + metric + "</b>: %{y:.4f}<extra></extra>",
      customdata: [comboLabel(r.combo, colorBy)],
    };
  });

  const yrange = yRangeOf(runs);
  const layout = {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    xaxis: { title: "step", gridcolor: "#eee" },
    yaxis: { title: metric, gridcolor: "#eee", range: yrange, autorange: yrange === undefined },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    legend: { groupclick: "toggleitem", font: { size: 11 } },
    hovermode: "closest",
  };

  Plotly.react("chart", traces, layout, { responsive: true, scrollZoom: true });

  document.getElementById("status").textContent =
    `${runs.length} / ${DATA.runs.length} runs visible`;
}

function buildFilters() {
  const cmap = colorMap(DATA.axes[colorBy] || []);
  const container = document.getElementById("filters");
  container.innerHTML = "";

  for (const [axis, values] of Object.entries(DATA.axes)) {
    const isColorAxis = axis === colorBy;
    const group = document.createElement("div");
    group.className = "filter-group";

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
        // Refresh checkboxes
        group.querySelectorAll("input[type=checkbox]").forEach(cb => {
          cb.checked = action === "all";
        });
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
      cb.checked = filters[axis]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[axis].add(val);
        else filters[axis].delete(val);
        buildChart();
      });

      if (isColorAxis) {
        const swatch = document.createElement("div");
        swatch.className = "color-swatch";
        swatch.style.background = cmap[val] || "#ccc";
        row.appendChild(cb);
        row.appendChild(swatch);
      } else {
        row.appendChild(cb);
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
  const sel = document.getElementById("color-by");
  sel.innerHTML = "";
  for (const axis of Object.keys(DATA.axes)) {
    const opt = document.createElement("option");
    opt.value = axis;
    opt.textContent = axis;
    if (axis === colorBy) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => {
    colorBy = sel.value;
    buildFilters();
    buildChart();
  });
}

function buildMetricSelect() {
  const metrics = new Set();
  DATA.runs.forEach(r => Object.keys(r.metrics).forEach(k => metrics.add(k)));
  const sel = document.getElementById("metric-sel");
  sel.innerHTML = "";
  for (const m of ["loss", "grad_norm", "tflops", "mfu", "tps"]) {
    if (!metrics.has(m)) continue;
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === metric) opt.selected = true;
    sel.appendChild(opt);
  }
  // Any remaining metrics not in preferred list
  for (const m of metrics) {
    if (["loss","grad_norm","tflops","mfu","tps"].includes(m)) continue;
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => { metric = sel.value; buildChart(); });
}

function init(data) {
  DATA = data;
  colorBy = Object.keys(DATA.axes)[0];

  // Init filters: all values visible
  for (const [axis, vals] of Object.entries(DATA.axes)) {
    filters[axis] = new Set(vals);
  }

  document.getElementById("exp-name").textContent = data.experiment;
  document.getElementById("smoothing").addEventListener("input", e => {
    smoothAlpha = parseFloat(e.target.value);
    document.getElementById("smooth-val").textContent = smoothAlpha.toFixed(2);
    buildChart();
  });

  buildColorBySelect();
  buildMetricSelect();
  buildFilters();
  buildChart();
}

fetch("/data.json")
  .then(r => r.json())
  .then(init)
  .catch(e => {
    document.getElementById("status").textContent = "Error loading data: " + e;
  });
</script>
</body>
</html>
"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    data_json = b""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/data.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(self.data_json))
            self.end_headers()
            self.wfile.write(self.data_json)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="experiment name or directory (default: newest)")
    parser.add_argument("--port", type=int, default=43801)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    sweep_dir = find_sweep_dir(args.experiment)
    print(f"Loading: {sweep_dir.name}")
    data = load_experiment(sweep_dir)
    print(f"  {len(data['runs'])} runs, axes: {list(data['axes'].keys())}")

    Handler.data_json = json.dumps(data, allow_nan=True).encode()

    url = f"http://localhost:{args.port}"
    print(f"  Serving at {url}  (Ctrl+C to stop)")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    HTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
