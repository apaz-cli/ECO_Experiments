# Open-Source Extraction Plan: Experiment Infrastructure

**Date:** 2026-03-02 (revised 2026-03-02)
**Scope:** `exp_logger.py`, `exp_server.py`, `visualize_experiment.py`, `run_sweep.py`, logging system, metrics routing

---

## Overview

This codebase contains three genuinely distinct subsystems worth extracting as open-source
tools. They evolved together around ECO/torchtitan research, but the core ideas are fully
general and would be useful to anyone running ML experiments — or honestly, any
parameterized computational job. The systems are:

1. **The sweep engine** (`run_sweep.py`) — a structured parameter exploration framework
2. **The experiment tracker** (`exp_logger.py` + `exp_server.py`) — a file-native, offline-first
   metrics collection and serving system
3. **The logging system** (`torchtitan/tools/logging.py`, `metrics.py`) — a multi-backend logger
   hierarchy with a clean abstract interface

These are described in detail below: what they do, what's already general, what's still
coupled to this repo, and what needs to change to make them a compelling open-source release.

**Revision notes (v2):** This version incorporates two additional requirements identified
during review: (1) live auto-updating visualization without page refresh or process restart,
and (2) correct behavior when the tracking server is hosted on a remote machine and goes
temporarily offline — specifically, fixing the data-scatter problem where metrics written
locally during an outage are never reconciled with the server.

**Revision notes (v3):** Resolved four open design questions: (1) sync replay triggers
immediately on reconnect, not only at close(); (2) live UI updates use an experiment
manifest written at sweep start rather than incremental axis discovery during polling — the
manifest solves the `/data.json`-is-slow problem and makes the poll loop structurally
simpler; (3) chart updates use `Plotly.extendTraces` + `Plotly.restyle` for zero-flicker
incremental rendering; (4) `BaseLogger` and backend implementations stay in torchtitan —
the new package contains only the sweep engine and the tracker trio (logger, server,
visualizer).

**Revision notes (v4):** External review addressed. Valid complaints incorporated:
replay chunking moved to Priority 2; sub-axis detection extracted to shared utility;
manifest run-list made lazy (only dispatched runs appear — axes remain upfront);
EMA/extendTraces interaction made explicit; 1-hour interrupted heuristic replaced with
heartbeat; `/run_status.json` given in-memory status cache; `since_step` semantics
clarified (`step > N`); manifest `metricNames` pre-population specified. One complaint
rejected (P1/P2 dependency — `POST /run/replay` was already in Priority 1). Thread-safety
race noted as benign under CPython GIL but worth locking for clarity.

---

## Part 1: The Sweep Engine (`run_sweep.py`)

### What it does

`run_sweep.py` is a parameter sweep system. You define a set of *dimensions* (axes of
variation), each dimension has a list of values and a corresponding set of CLI flags to
pass to the training command. The engine generates the full Cartesian product of all
dimensions (or a structured subset), then executes each variant as a subprocess, tracking
success and failure across runs.

The sweep format — a plain Python `OPTIONS` dict in a `.py` sweep file — is the real
intellectual contribution here. It supports:

- **Value dims**: explicit list of values, each mapped to CLI flags
- **Branch dims**: mutually exclusive named branches, each of which can introduce
  additional sub-dimensions (structured conditional variation trees)
- **Fixed dims**: flags always appended with no variation
- **Singular dims** (`singular: True`): "resolve once" semantics — only one value of this
  dimension needs to succeed per treatment. Used for hardware-dependent tuning like
  batch size and activation checkpointing mode, where you want to find the fastest
  working config rather than sweep all combos
- **Monotonic dims** (`monotonic: "increasing"/"decreasing"`): if a value fails, skip all
  worse values. Paired with singular dims, this gives you efficient binary-search-like
  probing without writing any search logic
- **Diagonal ordering for singular dims**: singular dims advance together rather than
  producing a full Cartesian product across themselves, which means you don't waste runs
  probing impossible combinations

The execution layer supports:
- Local single-GPU runs
- Multi-GPU parallel execution (via `CUDA_VISIBLE_DEVICES` per job)
- Remote SSH dispatch to multiple workers with automatic GPU discovery
- Artifact sync back from workers via rsync
- Dry-run mode (print commands without executing)
- Per-run log files in an organized output directory
- Live integration with the exp tracking server via `EXP_SERVER` env var

### What's already general

Essentially all of the *sweep logic* is general. The dimension types, the tree expansion
algorithm, the skip logic, the parallel execution machinery — none of this is ML-specific,
let alone ECO-specific. You could use this system to sweep hyperparameters for any
command-line program.

The output layout is fully generic: `{output_dir}/{experiment}/{run_name}/training.log`.
The `EXP_*` environment variables passed to each run are general conventions with no
torchtitan dependency.

The sweep file format (a `.py` file with `OPTIONS`, `BASE_CONFIG`, optionally `EXCLUDE` and
`EXTRA_FLAGS`) is clean and easy to write. The shebang mode (`#!/path/to/run_sweep.py`)
is a nice touch that makes sweep files self-executing.

The SSH remote dispatch is genuinely useful. The code to discover GPUs on workers, build
the slot queue, stream logs back via SSH stdout, and optionally rsync artifacts is all
general-purpose infrastructure that anyone running multi-node research experiments would
want.

### What's not general (coupling to this repo)

**The command builder is hardcoded to `run_config.sh`.**

```python
def build_cmd(base_config, overrides, run_dir, run_name, extra=()):
    return [
        os.path.join(REPO_ROOT, "run_config.sh"), base_config,
        "--job.dump_folder", run_dir, "--job.run_name", run_name,
        "--job.description", run_name, *overrides, *extra,
    ]
```

This is the main coupling point. The flags `--job.dump_folder`, `--job.run_name`,
`--job.description` are specific to torchtitan's `JobConfig`. A general user would want
to define their own command template. This is a one-function fix, but it's the blocker
for making the sweeper usable outside this project.

**The sweep files import from ECO-specific modules.**

`sweeps/paper_repro.py` imports `_common` and `_treatments` — both of which are
ECO/torchtitan-specific. These would stay in this repo as examples; the engine itself
is already separate.

**`REPO_ROOT` hardcoded path logic.**

The engine uses `os.path.dirname(os.path.abspath(__file__))` as `REPO_ROOT` and expects
`sweeps/` and `run_config.sh` relative to it. This is fine as a default but should be
configurable.

**The singular abbreviation dict is hardcoded.**

```python
abbrev = {"local_batch_size": "bs", "ac_mode": "ac", "compile": "comp"}
```

Minor, but this should not be in the engine itself — it belongs in the sweep file or a
config.

### Design assessment

The singular/monotonic/branch dimension system is the most distinctive design choice here.
It solves a real problem: when you're running hyperparameter searches on real hardware, you
don't want to enumerate all batch sizes for all treatments. You want to find the maximum
feasible batch size *once* (because it's hardware-specific) and then run all treatments
at that batch size. The diagonal ordering of singular dims is subtle but correct — it
ensures you're probing all treatments at the same singular values simultaneously rather
than fully resolving one before moving to the next.

The branch dims (mutual exclusion with sub-dimensions) are powerful. They let you write
a sweep where `optimizer=adam` introduces sub-dims for Adam's betas, while
`optimizer=sgd` introduces momentum — without generating nonsensical combinations. This
is more expressive than a flat grid with manual exclusion functions.

The `EXCLUDE` function hook for arbitrary filtering is a useful escape hatch, though it
suggests the branch dim system isn't yet expressive enough for all cases.

### What it needs to become more general

1. **Pluggable command builder.** The most important change. The sweep file should be able
   to define a `COMMAND` template or `CMD_BUILDER` callable. A good default would be a
   simple string template with `{run_dir}`, `{run_name}`, `{overrides}` placeholders. For
   the torchtitan case, you'd set `CMD_BUILDER = torchtitan_cmd_builder` or define it inline.

2. **Configurable output-dir injection flags.** Currently the engine hardcodes
   `--job.dump_folder` and `--job.run_name` as the flags used to tell the subprocess where
   to write its output. These should be configurable per sweep file (or per engine
   invocation), with sensible defaults.

3. **First-class Python API, not just CLI.** Right now `run_sweep` is a CLI tool. You
   should be able to `import run_sweep; run_sweep.run(options, ...)` from Python. This
   would make it much easier to embed in existing project scripts, Jupyter notebooks, or
   CI systems.

4. **Progress/status file.** The sweep writes a `sweep.log` but there's no structured
   status file you can query to see which runs succeeded/failed without parsing the log.
   A `sweep_status.json` updated after each run would let external tooling (dashboards,
   CI) monitor progress.

5. **Retry logic.** Currently a failed run is failed. Some teams want automatic retries
   for transient failures (OOM, network glitch on remote worker). A `max_retries` option
   per run would be useful.

6. **Checkpoint/resume.** If a sweep is interrupted halfway, you currently have to restart
   from scratch or manually skip completed runs. Reading `sweep_status.json` at startup
   to skip already-completed runs would make long sweeps resumable.

---

## Part 2: The Experiment Tracker (`exp_logger.py`, `exp_server.py`, `visualize_experiment.py`)

### What the three files do together

The tracker is a three-layer system, not two:

**`exp_logger.py`** (worker side): When training starts, writes `run_meta.json` and starts
a background thread that drains a queue of metric records. If `EXP_SERVER` is set, records
are POSTed to `exp_server.py`; if the server is unreachable, they fall back to direct
writes to the worker-local `metrics.jsonl`. On close, drains the queue and finalizes the
metadata.

**`exp_server.py`** (aggregator): A zero-dependency stdlib HTTP server. On startup it scans
a root directory for all `run_meta.json` files to rebuild an in-memory index. Accepts
`POST /run/start`, `POST /run/end`, and `POST /metrics` from live training jobs. Serves
`GET /experiments`, `GET /data.json`, and `GET /metric.json` for consumers. Keeps open
file handles per live run for efficient metric appends. This is the data-store / live-data
aggregator for running jobs.

**`visualize_experiment.py`** (consumer): A Python HTTP server that bundles a Plotly-based
single-page application. The SPA is served from this process; it makes fetch() calls to
its own server, which in turn either proxies to `exp_server.py` (if `--exp-source
http://host:port`) or reads local JSONL files directly (if pointed at a local sweep
directory). Features: dual-source data loading, experiment dropdown, metric selector,
color-by-axis, EMA smoothing, value filters, a curves tab, and a sensitivity
(dumbbell-by-dimension effect-size) tab. Dark/light theme. No build system — it's all
one file with inline HTML/JS.

The three-layer design is correct in structure. The bugs and missing capabilities are in
the details of the interactions between layers.

### What's already general

`exp_server.py` and `visualize_experiment.py` are both completely framework-agnostic. The
file layout (`run_meta.json` / `metrics.jsonl`) is a good, general convention. The
dual-source loading in `visualize_experiment.py` — transparent proxy to a remote
`exp_server` or direct file reads — is a strong design: it lets you visualize live runs
against a remote server, and also works offline after training completes, with no code
changes.

The sub-axis detection algorithm (automatically identifying dimensions whose presence
is conditional on a parent dimension value) is genuinely clever and enables smart UI
grouping without any configuration. The sensitivity tab, which ranks all axes by effect
size on the chosen metric, is a good general-purpose analysis tool. The PALETTE + EMA +
colored endpoint-dot design of the curves view is polished.

The logger interface is clean and minimal. The `log(metrics: dict, step: int)` call-site
API is as general as it gets.

### What's not general (coupling to this repo)

**`ExpLogger.__init__` takes `job_config: Any` and makes structural assumptions.**

```python
self.experiment = os.getenv("EXP_EXPERIMENT", job_config.job.description)
self.run_name = job_config.job.run_name or os.path.basename(log_dir)
meta = {"hparams": job_config.to_dict(), ...}
```

The logger assumes `.job.description`, `.job.run_name`, and `.to_dict()`. A general
user's config won't have this shape. This is a code smell — it works but fails
opaquely if the object has the wrong shape.

**The import inside `exp_logger.py` is coupled to torchtitan.**

```python
from torchtitan.tools.logging import logger as _logger
```

This appears in two places inside the file. The logger should have no knowledge of
any particular training framework's logging system.

**The `metrics.py` integration layer is deeply coupled to torchtitan.**

`_build_metric_logger` takes `JobConfig` and `ParallelDims`, handles pipeline-parallel
rank selection, computes log directory paths based on torchtitan conventions, etc. This
is not extractable — but it doesn't need to be. It's the integration glue that lives
in this repo and imports from the general package. Correct separation.

### The sync problem (broken in the current design)

The current fallback logic has a correctness bug in the remote-server case. When the
server is on a different machine:

1. Server goes offline mid-run.
2. Worker falls back to writing directly to its local `metrics.jsonl`.
3. Server comes back online.
4. **The records written during the outage are now only on the worker's disk.** The
   server never learns about them. When you restart the server it scans its own
   filesystem — but the worker's local files are on a different host. If the run ends
   during the outage, the server's record of that run is permanently incomplete.

Worse: **the logger has no reconnect logic at all.** Once `_server_ok` flips to `False`
on the first failure, `_flush_worker` never calls `_post_json` again (line 138: `if
self.server and self._server_ok`). The only thing that can reset `_server_ok` back to
`True` is a successful `_post_json` call — but `_flush_worker` won't attempt one while
it's `False`. The code comment makes this explicit: `# becomes False on first failure
(stays down for run)`. The server could recover 30 seconds into a run, and the logger
would keep writing locally for the remaining hours, with no possibility of recovery until
the sync fix below is implemented.

This is not hypothetical. A sweep with 20 runs on 4 remote workers, each run taking 30
minutes, will almost certainly experience at least one server hiccup. The consequence is
that the visualization shows gaps or truncated curves for affected runs, with no warning.

Even in the single-machine case there is a milder version of this problem: the server
keeps a file handle open (line-buffered append), while the fallback logger opens and
closes `metrics.jsonl` on each write (open → write → close). If both are writing to the
same file path — which they would be locally — the interleaving of writes from two
different file descriptors is undefined behavior. In practice, on Linux with O_APPEND,
individual writes are atomic, but the JSONL lines written by the fallback would not be
reflected in the server's in-memory index and would be invisible to the visualization
until server restart.

**The fix: local-first, always-write, replay-on-reconnect**

The architectural change is: flip the primacy. Currently the server is the primary
write target and local files are a fallback. The fix makes local files the unconditional
primary and the server a secondary target for live visualization.

New `ExpLogger` design:

```
log(metrics, step)
  ├─ always: enqueue record to background thread
  └─ background thread:
       ├─ append to local metrics.jsonl (always, unconditional)
       └─ if server configured:
            ├─ if server was reachable: POST record to server
            └─ if server was down: track _unsent_since (line offset)
                 → on reconnect: replay local file from _unsent_since to server
                 → server can also accept bulk-replay endpoint for this

close()
  ├─ drain queue
  ├─ if server configured and _unsent_since > 0:
  │    replay remaining records from local file to server (best-effort)
  └─ finalize run_meta.json, POST /run/end
```

Key properties of this design:
- **Local file is always complete.** Even if the server dies at step 1 and never comes
  back, the local `metrics.jsonl` has every record. Data loss requires the worker's local
  disk to fail, which is a different failure mode.
- **Server eventually consistent.** When the server comes back, the worker replays the
  missed records in order. The server receives the same stream it would have received
  had the outage not happened.
- **No interleaved writes.** The background thread is the only writer to the local file.
  No file descriptor contention with the server.
- **No open file handle in normal operation.** The background thread holds an open
  file handle to `metrics.jsonl` (opened once at start, closed at `close()`). This
  replaces both the current fallback pattern (open/close per record) and eliminates
  the potential for contention.

For `exp_server.py`, the only change needed is a new endpoint:

```
POST /run/replay   {"experiment": ..., "run_name": ..., "records": [...]}
```

The server appends the records to the run's file, deduplicating by step. This endpoint
is also useful for the case where a run completed before the server was started — you
can manually push the data without restarting.

### The live update problem (missing in the current design)

`visualize_experiment.py` currently has no polling mechanism. The JS `METRIC_CACHE` is
populated once on metric load and never invalidated. New training steps, and new runs
starting mid-sweep, are invisible until the user manually reloads the page. For a sweep
with many runs and steps, `/data.json` loading is slow — potentially longer than a
short run takes, and definitely much longer than a step takes. A naive approach of
reloading `/data.json` on every poll cycle is therefore not viable.

The solution has two parts: a **sweep manifest** written before any run starts, and
**incremental metric polling** that appends new steps without re-fetching history.

**Sweep manifest: `sweep_manifest.json`**

`run_sweep.py` already knows the complete experiment structure before executing any run:
it has called `generate_variations()` and knows every run name, every combo, and every
axis. It should write this to disk immediately, before the first job is submitted:

```json
{
  "experiment": "paper_repro_20260302_1730",
  "axes": {"treatment": ["bf16", "fp8_eco_sr", ...], "lr": [1e-3, 3e-3]},
  "runs": [
    {"name": "paper_repro_tmtbf16_lr0.001", "combo": {"treatment": "bf16", "lr": 0.001}},
    ...
  ],
  "subAxes": {...},
  "metricNames": []
}
```

The axes and run list are complete and fixed — they don't change as the sweep executes.
`metricNames` starts empty and is populated by the server/visualizer as runs produce data.

The visualizer loads the manifest instead of `/data.json` for initial structure. This
means:
- Filter checkboxes and color-by are fully populated before any run has started
- The chart can show all expected runs as "pending" (no data yet) with a visual stub
- The poll loop never needs to discover new axes — the manifest is the complete schema
- Polling only needs to track which runs have started (acquired `run_meta.json`) and
  what new metric data has arrived

The server (`exp_server.py`) serves `GET /manifest.json?name=...` by reading
`{root}/{experiment}/sweep_manifest.json` from disk at request time. No explicit
notification to `exp_server.py` is needed: since `run_sweep.py` uses atomic writes
(tmp file + `os.replace`), the server always reads a consistent snapshot. The `runs`
list grows as jobs are dispatched, and each client request gets the current state. The
server may cache the manifest with a short TTL (e.g., 5 seconds) during active dispatch
to avoid repeated disk reads; once all jobs are dispatched the manifest is immutable and
may be cached indefinitely. The visualizer proxies this endpoint.

When the sweep is run without exp_server (direct file mode), the visualizer reads the
manifest from disk directly. However, the poll loop (run status and incremental metrics)
requires a running `exp_server.py` — see the polling note in item 6 above.

**Incremental metric polling**

Add a `GET /metric_since.json?name=...&metric=...&since_step=N` endpoint to
`exp_server.py`. Returns `{run_name: {steps: [...], values: [...]}}` containing only
records with `step > N`. The client appends these to its existing `METRIC_CACHE`
rather than replacing it. This is O(new_points) not O(total_history).

To discover which runs have started (i.e., have real data now, vs. pending in the
manifest), add `GET /run_status.json?name=...` returning `{run_name: "running"|"completed"|"failed"|"pending"}` for all runs in the manifest. The client polls this to update the visual state of pending runs and to know which run hashes to request metric data for.

**Client-side poll loop**

The poll loop runs every N seconds (configurable via Python CLI arg, passed to the SPA
as a JSON config block in the HTML) and:

1. Calls `/run_status.json` to find newly-started runs. For each newly-started run,
   mark it as active in the UI (remove pending stub, enable metric fetching).
2. For all active runs, calls `/metric_since.json` with each run's `lastKnownStep`.
   Appends new `{steps, values}` to `METRIC_CACHE[metric][run_hash]`.
3. Updates the chart using `Plotly.extendTraces` for line traces (new points appended
   to existing traces) and `Plotly.restyle` for endpoint dot traces (update `x` and `y`
   arrays to the new tip position). This is zero-flicker — no full redraw.
4. Pauses when `document.visibilityState === "hidden"` (tab in background).

The sensitivity tab recomputes from current `METRIC_CACHE` on each render and doesn't
need special incremental logic — it just marks itself dirty and re-renders when the
active tab.

**Polling scope: proxy mode only**

The poll loop calls `/run_status.json` and `/metric_since.json`, which are server
endpoints. When `visualize_experiment.py` is in local-file mode (no `--exp-source`,
reading JSONL directly from disk), these endpoints don't exist and live updates are
unavailable. The manifest loads from disk in local-file mode for initial structure, but
the page must be manually refreshed to see new runs or new steps. This constraint should
be documented clearly. A future enhancement could implement local-file polling using
`inotify`/`watchdog`, but that is out of scope for the initial extraction.

**Caching**

The visualizer's own `_META_CACHE_TTL` applies to `/data.json` (legacy, still served
for backward compat) but should not apply to `/run_status.json` or `/metric_since.json`
— those are the live paths. The manifest is immutable once written and can be cached
indefinitely (or with a very long TTL). In proxy mode, the visualizer should not add
its own caching layer on top of exp_server's for the incremental endpoints.

### Remote server hosting considerations

When `exp_server.py` is on a remote machine, `visualize_experiment.py` runs locally and
proxies to it. This is the design today and it's mostly correct. A few things need
attention:

**The server needs to be accessible from the client machine.** Currently `exp_server.py`
defaults to binding on `0.0.0.0`, which is correct for remote access. `visualize_experiment.py`
defaults to `localhost`-only. Both should document how to configure for remote use.

**The visualizer's `_http_get_json` uses a 30-second timeout** for forwarded requests.
For `/metric.json` on a large experiment (many runs, many steps), reading all the JSONL
files could take longer than this. The timeout should be made configurable, or the
incremental endpoint should be preferred to keep individual requests fast.

**Authentication.** Both servers have `Access-Control-Allow-Origin: *` (or equivalent)
and no auth. For use on a shared or semi-public cluster, a simple shared token in a
request header (checked in the handler) would be sufficient. This is worth adding to
the extraction as an optional flag (`--token SECRET`), not because the data is sensitive,
but because you don't want other users on the cluster accidentally or deliberately
writing to your experiment runs.

**HTTPS.** Currently both servers speak plain HTTP. If the remote server is on a different
host than the training workers, metrics are transmitted unencrypted. In a cluster
environment this is usually fine (traffic stays on the internal network), but for
long-distance remote dispatch (workers in one data center, server in another) it's worth
noting. Adding TLS to a stdlib `http.server` is painful; the pragmatic answer is to SSH
tunnel (`ssh -L 53800:localhost:53800 remote-host`) rather than add TLS to the server.

### Design assessment

The overall philosophy — offline-first, files as source of truth, server as optional cache,
no database — is excellent and genuinely differentiating. Compared to MLflow (requires
database, running server), Aim (SQLite but still requires process, UI that's somewhat
heavy), and W&B (cloud, requires internet): this system works anywhere, loses no data
even if everything crashes, and produces artifacts (JSONL files) that are trivially
analyzed with any other tool.

The three-tier architecture (logger → aggregator server → visualization server) is right.
The visualization server being separate from the aggregator server is particularly good —
it means you can have multiple people running `visualize_experiment.py` against the same
`exp_server.py` without any conflict, and it keeps the aggregator lean and focused.

The sub-axis detection and sensitivity tab are both good ideas that no mainstream tool
has out of the box. Sensitivity analysis (which dimension matters most, measured by
effect size on final loss) is exactly what you want after a sweep, and having it
automatically computed from the run tags without any configuration is the right UX.

The main correctness hole is the sync issue described above. The main completeness hole
is live updates. Both are fixable without changing the file format or the fundamental
architecture.

### What it needs

**Must-do before extraction:**

1. **Fix the sync architecture** (local-first + replay-on-reconnect, as described above).
   This is a correctness issue, not a polish issue.

2. **Decouple `ExpLogger` from `job_config`**. Constructor should accept:
   `ExpLogger(log_dir, experiment, run_name, tags=None, hparams=None, server=None, tag=None)`.

3. **Remove the torchtitan logger import.** Replace with `logging.getLogger(__name__)`.
   Note: the import is inside `__init__` (line 84) and `_post_json` (line 117), not at
   module level. This means importing `ExpLogger` does not fail at import time — the error
   only surfaces at instantiation (the `_logger.info(...)` call in `__init__`) or at the
   first server failure (`_logger.warning(...)` in `_post_json`). Callers can import the
   class without torchtitan installed; they just can't instantiate it until this is fixed.

**Must-do for the live update feature:**

4. **Add `GET /metric_since.json?name=...&metric=...&since_step=N`** to `exp_server.py`
   and to `visualize_experiment.py`'s proxy handler. Returns only new steps, same format
   as `/metric.json`.

5. **Add `GET /run_status.json?name=...`** to `exp_server.py` and proxy. Returns
   `{run_name: "pending"|"running"|"completed"|"failed"}` for all runs known to the
   server for that experiment (served from in-memory state, O(1) per run).

6. **Add poll loop to the JS in `visualize_experiment.py`**. Use `Plotly.extendTraces`
   for appending new data to existing traces. Poll `/run_status.json` to detect runs
   transitioning from pending to active; poll `/metric_since.json` for new data points.
   **Note: the poll loop only works in proxy mode** (when pointed at a live
   `exp_server.py`). In local-file mode, live updates are unavailable — the manifest
   loads from disk for initial structure, but `/run_status.json` and `/metric_since.json`
   require a running server.

7. **Rewrite visualizer initialization to use the manifest.** Load `/manifest.json` for
   the experiment's structural schema (axes, subAxes, runs list). When a run transitions
   from pending to active (detected via `/run_status.json`), update its visual state
   incrementally. Do not re-fetch `/data.json` for run discovery — the manifest plus
   `/run_status.json` replace that path. `/data.json` remains as a fallback for non-sweep
   runs that have no manifest.

**Nice-to-have:**

8. **`POST /run/replay`** — ~~promoted to Priority 1; see item 1 in the Concrete Changes
   section. Do not defer this.~~ (Listed here only for cross-reference.)

9. **Persistent local file handle** in `ExpLogger`: open `metrics.jsonl` once at start,
   hold open, close in `close()`. Current open/close per record in fallback path is
   wasteful and causes a new seek on every write.

10. **Structured hyperparameter schema** in `run_meta.json`: distinguish `tags` (what you
    varied, typed) from `hparams` (full config dump). Currently both exist but tags are
    a flat `dict[str, str]` while hparams is a big nested dict. The tags should be the
    primary key structure the visualization uses; hparams is a searchable appendix.

11. **Tunable poll interval and cache TTLs** via command-line flags.

12. **Optional shared token** for `exp_server.py` and `visualize_experiment.py`
    (`--token SECRET`): check `Authorization: Bearer SECRET` on all requests.

---

## Part 3: The Logging System

### What it does

The logging system has two layers:

**`torchtitan/tools/logging.py`** (46 lines): A module-level Python `logging.Logger`
configured to write to stdout with a `[titan]` prefix and ISO timestamps. A `warn_once`
utility de-duplicates repeated warnings. This is the process-level log (for diagnostic
messages from training code).

**`metrics.py`**: The *metrics* logging system — separate from process logging. Defines
`BaseLogger`, `TensorBoardLogger`, `WandBLogger`, `ExpLogger`, and `LoggerContainer`.
The `MetricsProcessor` class wraps a `LoggerContainer` and adds the training-loop-specific
logic: computing throughput (tokens/second), MFU, device memory stats, formatting the
console output. It handles distributed rank selection (only rank 0 logs by default, unless
`save_for_all_ranks` is set, with special handling for pipeline-parallel loss visibility).

### What's already general

The **logger interface** is minimal and clean:
```python
class BaseLogger:
    def log(self, metrics: dict[str, Any], step: int) -> None: pass
    def close(self) -> None: pass
```

This is as general as it gets. The `LoggerContainer` pattern (fan-out to multiple backends)
is straightforward and correct. The `TensorBoardLogger` and `WandBLogger` implementations
are clean wrappers. The lazy import of `wandb` inside `__init__` is a good practice
(avoids import-time failure if wandb isn't installed).

### What's not general

`MetricsProcessor` is deeply coupled to torchtitan:
- Takes `JobConfig` and `ParallelDims` as constructor arguments
- Computes MFU using torchtitan's `get_peak_flops` utility
- Handles pipeline-parallel rank selection
- Formats console output with torchtitan-specific color utilities
- Tracks `ntokens_since_last_log` and `data_loading_times` in the training-loop style

None of this is extractable as part of the general logging infrastructure — nor should it
be. `MetricsProcessor` is the integration layer between the generic logger backends and
this specific training framework. It belongs in this repo.

### Design assessment

The split between process logging (`logging.py`) and metrics logging (`metrics.py`) is
correct and important. These are fundamentally different things: process logs are for
debugging and operational visibility (what is the code doing?), metrics are time-series
data for analysis (how well is training going?). Mixing them into a single system — as
some frameworks do — makes both worse.

The `BaseLogger` no-op pattern is clean. When logging is disabled, you get a `BaseLogger`
instance that costs nothing. The caller doesn't need to check `if logger_enabled` before
every `log()` call.

The process-level logger is very minimal — just a configured stdlib logger. This is
probably right. The main missing piece is rank-awareness: in a distributed training job,
you often want to suppress logs from all ranks except rank 0. The current code has this
logic in `MetricsProcessor` but not in the process logger itself.

### What it needs to become more general

1. **Rank filtering in the process logger.** Add `init_logger(rank=0, log_rank=0)` so that
   non-root ranks automatically suppress INFO-level output. This is a two-line change but
   makes the logger much more useful in a distributed context.

2. **The `BaseLogger` interface should be published as a formal ABC.** Currently it's just
   a class with pass-through methods. Making it an `ABC` with `@abstractmethod` marks
   makes the contract explicit and helps IDE tooling.

3. **Plugin registration.** If someone wants to add a new backend (MLflow, Neptune, custom
   database), they currently have to modify `metrics.py`. A simple registry —
   `register_logger_backend("mlflow", MLflowLogger)` — would make extension clean.

---

## Part 4: The Config System

### What it does

`job_config.py` defines the full training configuration as a hierarchy of Python dataclasses,
one per section. `config/manager.py` loads configuration via `tyro` (CLI parsing) with TOML
as a base file, supporting `--section.key value` overrides. The config object supports
`to_dict()` for serialization.

### Whether to extract this

The config system is **not a candidate for extraction** into the new repo. `JobConfig` is
completely specific to torchtitan/ECO training: it has sections for `Parallelism`,
`Quantize`, `ECO`, `FaultTolerance`, etc. It's the config for *this* training framework.

What *is* worth noting is the design pattern: hierarchical dataclass configs + tyro + TOML
is a good combination. It gives you type safety, auto-generated `--help`, and a natural
override hierarchy. The `to_dict()` method for serialization into `run_meta.json` is
important — the experiment tracker needs to capture what config was used for each run.

A general experiment tracker should accept `hparams: dict` rather than `job_config:
JobConfig`. The user is responsible for calling `.to_dict()` (or equivalent) on their
own config object before passing it in. This is the right abstraction boundary.

---

## Extraction Plan: Proposed Repository Structure

The three systems above should be extracted into a single new repository — call it
**`runtools`** or **`sweep`** or **`exptrack`** (name TBD). They belong together because
they're designed to work as a pipeline: `run_sweep.py` launches jobs → `ExpLogger` records
results → `exp_server.py` serves them → frontend visualizes them.

Proposed structure for the new repo:

```
runtools/
├── README.md
├── pyproject.toml
├── src/
│   └── runtools/
│       ├── __init__.py
│       ├── sweep/
│       │   ├── __init__.py
│       │   ├── engine.py          # Core: generate_variations, run_sweep, should_skip
│       │   ├── cli.py             # __main__ entry point (current main())
│       │   └── remote.py          # SSH dispatch, GPU discovery
│       └── tracker/
│           ├── __init__.py
│           ├── logger.py          # ExpLogger (decoupled, local-first, replay-on-reconnect)
│           ├── server.py          # exp_server: aggregator + live write target
│           └── visualize.py       # visualize_experiment.py: browser UI + proxy server
├── examples/
│   ├── torchtitan/                # Integration example for torchtitan
│   │   ├── sweep_file.py          # How to write a sweep file for torchtitan
│   │   └── cmd_builder.py         # Torchtitan-specific CMD_BUILDER
│   └── simple/                    # Minimal working example (no ML framework)
│       ├── train.py               # A trivial training script
│       └── my_sweep.py            # A sweep file for it
└── docs/
    ├── sweep_format.md            # OPTIONS dict format docs (already exists)
    ├── tracker_api.md             # Server API reference (endpoints, formats)
    ├── sync_architecture.md       # How local-first + replay-on-reconnect works
    └── quickstart.md
```

---

## Concrete Changes Required Before Extraction

### Priority 1: Correctness (must fix, blocks everything)

**1. Fix the sync architecture in `ExpLogger`.**

The current design (POST to server OR write locally) has a correctness bug when the
server is remote: records written locally during an outage are stranded on the worker's
disk and never reach the server. The fix is local-first with server as secondary:

- Background thread always writes to a persistent local file handle (opened once at
  start, closed in `close()`). This replaces both the open/close-per-record fallback
  pattern and eliminates any contention with the server's own file handles.
- Background thread additionally POSTs to the server when reachable.
- Thread tracks `_unsent_line` (line count in local file successfully sent to server).
- On reconnect (first successful POST after failures): replay local file from
  `_unsent_line` to current end.
- In `close()`: flush queue, then if `_unsent_line < total_lines`, replay remainder
  to server before finalizing.

Add `POST /run/replay` to `exp_server.py`: accepts `{"experiment", "run_name",
"records": [...]}`, appends records to run file, deduplicates by step. Note: `POST
/run/replay` is required for the sync fix and must be implemented as part of Priority 1,
not deferred.

On reconnect (first successful POST after a failure), the background thread immediately
replays the local file from `_unsent_line` to the current end before resuming normal
streaming. The replay is **chunked**: send at most N records per request (N configurable,
default 500) to avoid HTTP timeout on long outages. If a chunk fails, the unsent offset
does not advance and replay retries on the next reconnect or at `close()`. A multi-hour
outage with log_freq=1 step/second could produce ~10,000+ records; a single 10MB POST
would risk timeout even with a generous timeout setting.

Also: the background thread writes a `last_heartbeat` timestamp to `run_meta.json` every
60 seconds. The server uses this instead of the blunt 1-hour cutoff to determine whether
a "running" run is genuinely alive or interrupted. A run is considered interrupted if
`last_heartbeat` is absent and `start_time` is more than 1 hour ago, OR if
`last_heartbeat` is present and more than 5 minutes old (configurable). This prevents
long-running (multi-hour) jobs from being incorrectly marked interrupted on server
restart.

**2. Decouple `ExpLogger` from `job_config`.**

Current signature:
```python
class ExpLogger:
    def __init__(self, log_dir: str, job_config: Any, tag: str | None = None):
        self.experiment = os.getenv("EXP_EXPERIMENT", job_config.job.description)
        self.run_name = job_config.job.run_name or os.path.basename(log_dir)
        meta = {"hparams": job_config.to_dict(), ...}
```

New signature:
```python
class ExpLogger:
    def __init__(
        self,
        log_dir: str,
        *,
        experiment: str,
        run_name: str,
        tags: dict[str, str | int | float | bool] | None = None,
        hparams: dict | None = None,
        server: str | None = None,  # http://host:port
        tag: str | None = None,
    ):
```

The caller (torchtitan's `metrics.py`) constructs `ExpLogger` with values extracted from
`job_config`. The logger itself knows nothing about `JobConfig`. Env var reading
(`EXP_EXPERIMENT`, `EXP_TAGS`, `EXP_SERVER`) moves to the calling code or a factory
function — not baked into `__init__`.

**3. Remove `from torchtitan.tools.logging import logger` from `exp_logger.py`.**

```python
import logging
_logger = logging.getLogger(__name__)
```

**4. Parameterize the command builder in `run_sweep.py`.**

The sweep file should be able to define `CMD_BUILDER`, a callable that produces the
command list for each run. The engine's built-in default remains the current
`run_config.sh` invocation, but it's no longer hardwired. Sweep files that don't define
`CMD_BUILDER` get the default; files that do get full control.

```python
# In sweep file (torchtitan example):
def CMD_BUILDER(base_config, overrides, run_dir, run_name, extra):
    return ["bash", "run_config.sh", base_config,
            "--job.dump_folder", run_dir, "--job.run_name", run_name,
            "--job.description", run_name, *overrides, *extra]
```

### Priority 2: Live updates (required for the stated use case)

**5. Write `sweep_manifest.json` from `run_sweep.py`.**

The manifest has two parts written at different times:

**At sweep launch** (before any job is submitted), write the axis structure:
```json
{
  "experiment": "paper_repro_20260302_1730",
  "axes": {"treatment": ["bf16", "fp8_eco_sr", ...], "lr": [1e-3, 3e-3]},
  "subAxes": {...},
  "runs": [],
  "metricNames": []
}
```
Axes and subAxes are complete and immutable from this point. The `runs` list starts
empty. This lets the visualizer pre-populate filter checkboxes and the color-by selector
before any run has started.

**As each job is actually dispatched** (in `_job_worker`, after skip checks, just before
`subprocess.run`), append the run's entry to the manifest:
```json
{"name": "paper_repro_tmtbf16_lr0.001", "combo": {"treatment": "bf16", "lr": 0.001}}
```
This means singular/monotonic probes that are skipped by the engine never appear in the
manifest. The visualizer sees only dispatched runs. Skipped runs simply don't exist from
the visualizer's perspective — no permanent pending stubs.

**Sub-axis detection** must be extracted from `exp_server.py`'s `_build_experiment_meta`
into a shared utility function — `detect_sub_axes(combos, axes)` — used by both
`run_sweep.py` (to populate `subAxes` in the manifest) and `exp_server.py` (for backward
compat with non-sweep runs using the `/data.json` path). The algorithm operates on the
variation combos, which `run_sweep.py` has available from `generate_variations()`.

**`metricNames`** is populated dynamically by the server when serving `GET
/manifest.json`: if `metricNames` is empty in the on-disk file, the server reads the
first line of `metrics.jsonl` for any available run in that experiment, extracts the
metric keys (excluding `step` and `t`), and merges them into the response without
writing back to the file. `run_sweep.py` does not write `metricNames` — it's always
`[]` in the file. When the manifest is served and no runs have produced data yet
(`metricNames` is still empty after this check), the visualizer's metric selector shows
a loading state rather than an empty/broken list.

**6. Add new endpoints to `exp_server.py`.**

```
GET /manifest.json?name=...
```
Reads `{experiment}/sweep_manifest.json` from disk at request time. Short TTL cache
(5 seconds) during active sweep dispatch since the `runs` list grows; immutable and
freely cacheable once dispatch completes. Returns a structured error if no manifest
exists (direct-logger runs without a sweep).

```
GET /metric_since.json?name=...&metric=...&since_step=N
```
Returns `{run_name: {steps: [...], values: [...]}}` for all steps **strictly greater
than** N (i.e., `step > N`, not `step >= N`). The client passes its last known step as N,
so `> N` returns only new points without re-fetching the boundary. No caching.

```
GET /run_status.json?name=...
```
Returns `{run_name: "pending"|"running"|"completed"|"failed"}` for all runs known to the
server. Served from the in-memory `_run_status` dict (updated by `POST /run/start` and
`POST /run/end`, seeded from disk at startup) — never reads `run_meta.json` files on
this call. This is O(1) per run regardless of experiment size. No caching needed since
the in-memory dict is always current. Pending means the manifest lists the run but no
`/run/start` has arrived.

All three must be proxied in `visualize_experiment.py`'s HTTP handler.

**7. Rewrite visualizer initialization to use the manifest.**

On experiment load: fetch `/manifest.json` first. If present, use it as the structural
source (axes, runs, subAxes). If absent (no sweep manifest, just direct-logger runs),
fall back to `/data.json` as today. Either way, once axes and run list are known, fetch
initial metric data per run.

**8. Add poll loop to the JS in `visualize_experiment.py`.**

Poll every N seconds (N configurable, default 10, passed from Python via an inline JSON
config block in the served HTML). On each cycle:
1. Fetch `/run_status.json` → find newly-started runs, update their visual state
   (pending stub → active trace).
2. Fetch `/metric_since.json` for each active run with `since_step = lastKnownStep[run]`
   → append new `{steps, values}` to `METRIC_CACHE[metric][run_hash]`.
3. Update chart. `METRIC_CACHE` always stores **raw (unsmoothed) values**. Chart traces
   store **smoothed values** (EMA-applied at render time). On poll tick:
   - For each run with new raw points, compute the next smoothed value(s) incrementally:
     `smoothed = alpha * lastSmoothedValue + (1 - alpha) * newRawValue` for each new
     point, using the last element of the trace's current `y` array as `lastSmoothedValue`.
   - Call `Plotly.extendTraces` with the new smoothed `{x, y}` arrays for line traces.
   - Call `Plotly.restyle` to update endpoint dot traces to the new tip position.
   - On smoothing slider change (user action): recompute full EMA array from raw values
     in `METRIC_CACHE` and call `Plotly.react` — this is correct and fast since the raw
     values are always available.
   `Plotly.react` is also used for all user-driven changes (metric switch, filter change,
   experiment switch).
4. Pause when `document.visibilityState === "hidden"`.

**9. Fix cache invalidation and thread safety.**

`/manifest.json` — immutable, can be cached indefinitely.
`/run_status.json` and `/metric_since.json` — no caching in either the visualizer proxy
or exp_server. These are the live paths.
`/data.json` — retain TTL (used for backward compat and non-sweep runs), make it
configurable.

`_meta_cache` in `exp_server.py` is a plain dict accessed from multiple handler threads.
In CPython the GIL makes individual dict reads and writes atomic, so the actual failure
mode is benign: two threads both miss the cache simultaneously, both compute the metadata,
and one's result overwrites the other. This is wasted computation, not corruption. Still,
wrap cache reads and writes in the existing `_lock` for correctness and to avoid
surprises on non-CPython runtimes.

### Priority 3: Important improvements (should do)

**10. Sweep resume / skip-completed.**

At sweep start, check for `{exp_dir}/sweep_status.json`. Skip variations whose run name
appears in the completed set. This makes long sweeps resumable without manual
intervention.

**11. First-class Python API for the sweep engine.**

```python
from runtools.sweep import Sweep, run

sweep = Sweep(options=OPTIONS, cmd_builder=my_cmd_builder)
results = run(sweep, output_dir="outputs/sweeps", gpus=4)
```

**12. Pip-installable package.**

Both `run_sweep.py`, `exp_server.py`, and `visualize_experiment.py` currently live at
the repo root as standalone scripts. They need a proper `pyproject.toml` with entry
points:
```toml
[project.scripts]
runtools-sweep = "runtools.sweep.cli:main"
runtools-server = "runtools.tracker.server:main"
runtools-viz = "runtools.tracker.visualize:main"
```

**13. `sweep_status.json`.**

Write a structured JSON file after each run completes, recording outcome:
```json
{
  "run_name": {"status": "ok", "elapsed": 123.4, "combo": {...}},
  ...
}
```

### Priority 4: Polish (nice to have)

**14. Configurable singular dim abbreviations.**

Move the `abbrev = {"local_batch_size": "bs", ...}` dict out of the engine and into
sweep files, already partially supported by the `"name"` key in dimension specs.

**15. Server: watch mode for new runs discovered off-path.**

If a run's `run_meta.json` appears on disk without a corresponding `POST /run/start`
(e.g., the worker wrote directly during a server outage, then the server was restarted),
the server won't know about it. A lightweight polling loop in `exp_server.py` that
rescans for new `*/*/run_meta.json` files every 60s would pick these up automatically.

**16. Optional shared token for remote deployments.**

`exp_server.py` and `visualize_experiment.py` accept `--token SECRET`. The server checks
`Authorization: Bearer SECRET` on all POST requests (GET requests are read-only and
can remain open, or also gated). This is sufficient for shared-cluster use without
adding TLS complexity.

**17. Configurable timeouts and poll intervals.**

`visualize_experiment.py`'s `_http_get_json` uses a fixed 30s timeout. For large
experiments this can be too short for `/metric.json`. Expose `--request-timeout` and
`--poll-interval` as CLI flags.

---

## What stays in this repo

Not everything should leave. The following stays:

- **`torchtitan/`** — the training framework. This is the user of the infrastructure,
  not the infrastructure itself.
- **`torchtitan/components/metrics.py`** — the integration glue between torchtitan and
  the general logger backends. Stays here, imports from the new package.
- **`sweeps/`** — the ECO-specific sweep files (`paper_repro.py`, `_treatments.py`,
  `_common.py`). These are examples that *use* the sweep engine; they document the
  research, not the tool.
- **`configs/`** — the TOML configuration files for ECO experiments.
- **`run_config.sh`** — the torchrun invocation wrapper. This is the "command" that the
  sweep engine calls, and it's specific to torchtitan.

What needs to be extracted:

| File | Destination | Notes |
|------|-------------|-------|
| `run_sweep.py` | `runtools/sweep/engine.py` + `cli.py` | Parameterize CMD_BUILDER; add manifest write |
| `exp_server.py` | `runtools/tracker/server.py` | Add `/manifest`, `/metric_since`, `/run_status`, `/run/replay` |
| `visualize_experiment.py` | `runtools/tracker/visualize.py` | Manifest-based init; poll loop; extendTraces |
| `torchtitan/components/exp_logger.py` | `runtools/tracker/logger.py` | Local-first rewrite; decouple from JobConfig; replay-on-reconnect |
| `docs/sweep_configuration.md` | `runtools/docs/sweep_format.md` | Already good |
| `torchtitan/components/metrics.py` | stays in torchtitan | `BaseLogger`, `TensorBoardLogger`, `WandBLogger`, and `LoggerContainer` all stay here — they are torchtitan's metric logging infrastructure, not part of the new package. The new package (`runtools`) defines no logger base classes; it only provides `ExpLogger` as a concrete backend. |

---

## Summary Assessment

The infrastructure in this repo is genuinely good. It solves real problems — offline-first
experiment tracking, efficient parameter sweeps with monotonic/singular probing, remote
multi-worker dispatch, a complete browser-based visualization UI — in ways that the
mainstream tools don't. The file-native design is a strong differentiator: no database,
no cloud dependency, no lock-in, results readable with `jq` forever.

The work to make it open-source ready breaks into three distinct categories:

**Correctness (Priority 1):** The sync bug — metrics stranded on worker disks when the
remote server is offline — is a real correctness issue, not a polish issue. It needs to
be fixed before anyone uses this with remote workers. The fix (local-first writes, server
as secondary, immediate replay on reconnect) also makes the code simpler: one consistent
write path instead of two mutually exclusive ones. Replay on reconnect rather than only
at close() means the live visualization recovers mid-run rather than staying gapped for
the entire outage duration.

**Live updates (Priority 2):** The visualization is currently static — you load data once
and it doesn't change. The key architectural addition is the **sweep manifest**: `run_sweep.py`
already knows the complete experiment structure before submitting any jobs, so it writes
`sweep_manifest.json` upfront. The visualizer loads the manifest for initial structure
(axes, run list, combos) and then only needs incremental polling for metric data and run
status — it never needs to re-fetch the full `/data.json`. This avoids the
reload-is-slower-than-a-run problem and makes the poll loop structurally simple.
`Plotly.extendTraces` + `Plotly.restyle` for chart updates gives zero-flicker incremental
rendering.

**Decoupling and packaging (Priority 3):** The torchtitan-specific coupling in `ExpLogger`
and the hardcoded `run_config.sh` in the sweep engine are small changes, but they're what
makes the code usable by anyone who isn't using torchtitan. The `BaseLogger` hierarchy
stays in torchtitan — it's unrelated to the new package. The new package is: sweep engine
+ tracker (logger, server, visualizer).

The sweep engine is already the most complete and sophisticated part of the system. The
visualization is further along than expected — it's a real, polished tool. The logger and
server are correct in design but need the sync fix. The overall sequence: fix sync →
write manifest support → add incremental endpoints → add poll loop → decouple → package
→ release.
