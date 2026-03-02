#!/usr/bin/env python3
"""Experiment tracking server for ECO/torchtitan experiments.

Lightweight HTTP server that replaces Aim for storing and serving training metrics.
Data is stored as JSONL files co-located with training output — no database needed.

Storage layout:
    <exp-dir>/<experiment>/<run_name>/
        run_meta.json   — written once at run start, updated at end
        metrics.jsonl   — one JSON line per log() call, append-only

Usage:
    python exp_server.py --exp-dir outputs/sweeps --port 53800 --host 0.0.0.0

API:
    POST /run/start    — register a new run, write run_meta.json
    POST /run/end      — finalize a run with end_time + status
    POST /metrics      — append one JSONL line to a run's metrics.jsonl
    GET  /experiments  — list experiments sorted newest-first
    GET  /data.json?name=...          — experiment metadata (30s TTL cache)
    GET  /metric.json?name=...&metric=... — metric time-series for all runs
"""

import argparse
import json
import os
import socketserver
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path


# ── In-memory index ────────────────────────────────────────────────────────────

# run_key = f"{experiment}/{run_name}"
_experiments: dict[str, list[str]] = {}   # experiment → [run_name, ...]
_run_dirs: dict[str, str] = {}             # run_key → absolute path
_open_files: dict[str, object] = {}        # run_key → open append file handle

_lock = threading.Lock()        # guards _experiments, _run_dirs
_files_lock = threading.Lock()  # guards _open_files

_root_dir: str = ""

# GET /data.json cache: experiment_name → (timestamp, json_bytes)
_meta_cache: dict[str, tuple[float, bytes]] = {}
_META_CACHE_TTL = 30.0  # seconds


def _run_key(experiment: str, run_name: str) -> str:
    return f"{experiment}/{run_name}"


def _scan_disk(root_dir: str) -> None:
    """Rebuild in-memory index from disk. Called once at startup."""
    global _root_dir
    _root_dir = os.path.abspath(root_dir)
    stale_cutoff = time.time() - 3600  # runs older than 1h and still "running" → interrupted

    with _lock:
        for meta_path in sorted(Path(_root_dir).glob("*/*/run_meta.json")):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            run_name = meta.get("run_name")
            experiment = meta.get("experiment")
            if not run_name or not experiment:
                continue

            run_dir = str(meta_path.parent)
            key = _run_key(experiment, run_name)
            _run_dirs[key] = run_dir

            if experiment not in _experiments:
                _experiments[experiment] = []
            if run_name not in _experiments[experiment]:
                _experiments[experiment].append(run_name)

            # Mark stale running runs as interrupted (cosmetic only)
            if (
                meta.get("status") == "running"
                and meta.get("start_time", 0) < stale_cutoff
            ):
                meta["status"] = "interrupted"
                tmp = str(meta_path) + ".tmp"
                try:
                    with open(tmp, "w") as f:
                        json.dump(meta, f, indent=2)
                    os.replace(tmp, str(meta_path))
                except OSError:
                    pass

    n_exp = len(_experiments)
    n_runs = len(_run_dirs)
    print(f"Scanned {_root_dir}: {n_runs} runs across {n_exp} experiments", flush=True)


def _get_or_open_file(experiment: str, run_name: str):
    """Return the open (line-buffered) append file handle for a run, opening if needed."""
    key = _run_key(experiment, run_name)
    with _files_lock:
        if key in _open_files:
            return _open_files[key]
        with _lock:
            run_dir = _run_dirs.get(key)
        if run_dir is None:
            return None
        path = os.path.join(run_dir, "metrics.jsonl")
        fh = open(path, "a", buffering=1)  # line-buffered: flush on each newline
        _open_files[key] = fh
        return fh


# ── HTTP handler ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            self._handle_post()
        except Exception:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def do_GET(self):
        try:
            self._handle_get()
        except Exception:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(data).encode())

    def _handle_post(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/run/start":
            meta = self._read_body()
            run_name = meta.get("run_name")
            experiment = meta.get("experiment")
            if not run_name or not experiment:
                self.send_response(400)
                self.end_headers()
                return

            run_dir = os.path.join(_root_dir, experiment, run_name)
            os.makedirs(run_dir, exist_ok=True)

            # Write run_meta.json atomically
            meta_path = os.path.join(run_dir, "run_meta.json")
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_path)

            key = _run_key(experiment, run_name)
            with _lock:
                _run_dirs[key] = run_dir
                if experiment not in _experiments:
                    _experiments[experiment] = []
                if run_name not in _experiments[experiment]:
                    _experiments[experiment].append(run_name)
                # Invalidate TTL cache so next /data.json picks up the new run
                _meta_cache.pop(experiment, None)

            print(f"[start] {experiment}/{run_name}", flush=True)
            self._send_json({"ok": True})

        elif parsed.path == "/run/end":
            data = self._read_body()
            run_name = data.get("run_name")
            experiment = data.get("experiment")
            if not run_name or not experiment:
                self.send_response(400)
                self.end_headers()
                return

            key = _run_key(experiment, run_name)
            with _lock:
                run_dir = _run_dirs.get(key)
            if run_dir is None:
                self.send_response(404)
                self.end_headers()
                return

            meta_path = os.path.join(run_dir, "run_meta.json")
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = {}
            meta["end_time"] = data.get("end_time", time.time())
            meta["status"] = data.get("status", "completed")
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_path)

            # Close and remove the cached file handle for this run
            with _files_lock:
                fh = _open_files.pop(key, None)
                if fh:
                    try:
                        fh.close()
                    except OSError:
                        pass

            print(f"[end]   {experiment}/{run_name}  status={meta['status']}", flush=True)
            self._send_json({"ok": True})

        elif parsed.path == "/metrics":
            data = self._read_body()
            run_name = data.get("run_name")
            experiment = data.get("experiment")
            record = data.get("record")
            if not run_name or not experiment or record is None:
                self.send_response(400)
                self.end_headers()
                return

            fh = _get_or_open_file(experiment, run_name)
            if fh is None:
                self.send_response(404)
                self.end_headers()
                return

            fh.write(json.dumps(record) + "\n")
            self._send_json({"ok": True})

        else:
            self.send_response(404)
            self.end_headers()

    def _handle_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/experiments":
            with _lock:
                exps = list(_experiments.keys())

            # Sort newest-first by the highest start_time among runs
            def newest_start(exp: str) -> float:
                best = 0.0
                with _lock:
                    runs = list(_experiments.get(exp, []))
                for rn in runs:
                    key = _run_key(exp, rn)
                    with _lock:
                        rd = _run_dirs.get(key)
                    if rd:
                        try:
                            with open(os.path.join(rd, "run_meta.json")) as f:
                                m = json.load(f)
                            best = max(best, m.get("start_time", 0.0))
                        except (OSError, json.JSONDecodeError):
                            pass
                return best

            experiments = sorted(exps, key=newest_start, reverse=True)
            self._send_json({
                "experiments": experiments,
                "default": experiments[0] if experiments else None,
            })

        elif parsed.path == "/data.json":
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400)
                self.end_headers()
                return

            now = time.time()
            if name in _meta_cache:
                ts, cached_body = _meta_cache[name]
                if now - ts < _META_CACHE_TTL:
                    self._send(200, "application/json", cached_body)
                    return

            data = _build_experiment_meta(name)
            body = json.dumps(data).encode()
            _meta_cache[name] = (now, body)
            self._send(200, "application/json", body)

        elif parsed.path == "/metric.json":
            name = qs.get("name", [None])[0]
            metric = qs.get("metric", [None])[0]
            if not name or not metric:
                self.send_response(400)
                self.end_headers()
                return
            data = _load_metric_data(name, metric)
            self._send_json(data)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


# ── Data builders ──────────────────────────────────────────────────────────────

def _parse_tag_value(s: str):
    """Convert a tag value string to a typed Python value."""
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _build_experiment_meta(experiment_name: str) -> dict:
    """Build metadata dict for an experiment by reading run_meta.json files."""
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))

    axis_values: dict[str, set] = {}
    runs = []
    metric_names: set[str] = set()

    for run_name in run_names:
        key = _run_key(experiment_name, run_name)
        with _lock:
            run_dir = _run_dirs.get(key)
        if not run_dir:
            continue

        try:
            with open(os.path.join(run_dir, "run_meta.json")) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        # Reconstruct combo from tags dict
        tags = meta.get("tags", {})
        combo = {k: _parse_tag_value(str(v)) for k, v in tags.items()}
        for k, v in combo.items():
            axis_values.setdefault(k, set()).add(v)

        # Use run_name as the "hash" so the JS can key into METRIC_CACHE
        runs.append({"name": run_name, "hash": run_name, "combo": combo})

        # Discover metric names from first line of metrics.jsonl (fast)
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        try:
            with open(metrics_path) as f:
                first_line = f.readline()
            if first_line.strip():
                rec = json.loads(first_line)
                for k in rec:
                    if k not in ("step", "t"):
                        metric_names.add(k)
        except (OSError, json.JSONDecodeError):
            pass

    def val_sort_key(v):
        if isinstance(v, bool):
            return (0, str(v))
        if isinstance(v, (int, float)):
            return (1, v)
        return (2, str(v))

    axes = {k: sorted(vs, key=val_sort_key) for k, vs in axis_values.items()}

    # Detect sub-axes (axes that only appear when a parent axis has a specific value)
    all_names = {r["hash"] for r in runs}
    names_with = {ax: {r["hash"] for r in runs if ax in r["combo"]} for ax in axes}
    sub_axes: dict = {}
    for axis in axes:
        if names_with[axis] == all_names:
            continue  # universal axis
        for parent_axis in axes:
            if parent_axis == axis:
                continue
            for parent_val in axes[parent_axis]:
                names_with_parent = {
                    r["hash"] for r in runs if r["combo"].get(parent_axis) == parent_val
                }
                if names_with_parent == names_with[axis]:
                    sub_axes[axis] = {"parentAxis": parent_axis, "parentValue": parent_val}
                    break
            if axis in sub_axes:
                break

    return {
        "experiment": experiment_name,
        "axes": axes,
        "runs": runs,
        "metricNames": sorted(metric_names),
        "subAxes": sub_axes,
    }


def _load_metric_data(experiment_name: str, metric_name: str) -> dict:
    """Load one metric's values for all runs in an experiment.

    Returns {run_name: {"steps": [...], "values": [...]}} — keyed by run_name
    (same as "hash" in /data.json, so the JS METRIC_CACHE can look it up).
    Skips malformed JSONL lines (handles truncated last line on live runs).
    """
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))

    result = {}
    for run_name in run_names:
        key = _run_key(experiment_name, run_name)
        with _lock:
            run_dir = _run_dirs.get(key)
        if not run_dir:
            continue

        steps = []
        values = []
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        try:
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip truncated last line
                    if metric_name in rec:
                        steps.append(rec["step"])
                        v = rec[metric_name]
                        # Normalize NaN to null for JSON
                        values.append(None if (isinstance(v, float) and v != v) else v)
        except OSError:
            continue

        if steps:
            result[run_name] = {"steps": steps, "values": values}

    return result


# ── Server ─────────────────────────────────────────────────────────────────────

class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        default="outputs/sweeps",
        help="Root directory containing experiment runs (default: outputs/sweeps)",
    )
    parser.add_argument("--port", type=int, default=53800)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _scan_disk(args.exp_dir)

    server = ThreadingServer((args.host, args.port), Handler)
    print(
        f"Experiment server at http://{args.host}:{args.port}  (Ctrl+C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        server.shutdown()
        with _files_lock:
            for fh in list(_open_files.values()):
                try:
                    fh.close()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
