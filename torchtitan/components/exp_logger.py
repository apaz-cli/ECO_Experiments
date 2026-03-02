# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""File-based experiment logger that replaces AimLogger.

Writes run_meta.json and metrics.jsonl co-located with training output.
If EXP_SERVER is set, also POSTs to exp_server.py for live visualization.
Falls back to direct file writes if server is unreachable.

Env vars:
    EXP_EXPERIMENT  experiment name (default: job.description)
    EXP_TAGS        comma-separated key=value pairs
    EXP_SERVER      http://host:port  (if absent: write files directly)
"""

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any


class ExpLogger:
    """Logger that writes JSONL metrics files and optionally POSTs to exp_server.py.

    Non-blocking: log() enqueues immediately and returns. A background daemon
    thread drains the queue and writes/posts. Falls back to direct file writes
    if the server is unreachable, warning once per run.
    """

    def __init__(self, log_dir: str, job_config: Any, tag: str | None = None):
        self.tag = tag
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Config from env vars
        self.experiment = os.getenv("EXP_EXPERIMENT", job_config.job.description)
        self.run_name = job_config.job.run_name or os.path.basename(log_dir)
        self.server = os.getenv("EXP_SERVER", None)  # http://host:port or None
        self._server_ok = True     # becomes False on first failure (stays down for run)
        self._warned_server = False

        # Parse EXP_TAGS=key=value,key2=value2 into a dict
        tags: dict[str, str] = {}
        for pair in os.getenv("EXP_TAGS", "").split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                tags[k.strip()] = v.strip()

        # File paths
        self._meta_path = os.path.join(log_dir, "run_meta.json")
        self._metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self._start_time = time.time()

        meta = {
            "experiment": self.experiment,
            "run_name": self.run_name,
            "tags": tags,
            "hparams": job_config.to_dict(),
            "start_time": self._start_time,
            "end_time": None,
            "status": "running",
        }
        self._write_meta_atomic(meta)

        # POST /run/start to server (best-effort)
        if self.server:
            self._post_json("/run/start", meta)

        # Background flush thread
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._flush_thread = threading.Thread(
            target=self._flush_worker, daemon=True, name="exp-logger-flush"
        )
        self._flush_thread.start()

        from torchtitan.tools.logging import logger as _logger
        _logger.info(
            f"ExpLogger enabled: experiment={self.experiment!r}, "
            f"run={self.run_name!r}, dir={log_dir}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_meta_atomic(self, meta: dict) -> None:
        """Write meta dict atomically via tmp file + os.replace."""
        tmp = self._meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, self._meta_path)

    def _post_json(self, path: str, data: dict) -> bool:
        """POST JSON to server. Returns True on success, False on failure."""
        if not self.server:
            return False
        url = self.server.rstrip("/") + path
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
            self._server_ok = True
            return True
        except (urllib.error.URLError, OSError) as e:
            if not self._warned_server:
                from torchtitan.tools.logging import logger as _logger
                _logger.warning(
                    f"ExpLogger: server {self.server} unreachable ({e}), "
                    "falling back to direct file writes. Data is safe on disk."
                )
                self._warned_server = True
            self._server_ok = False
            return False

    def _append_local(self, record: dict) -> None:
        """Append one JSON line to metrics.jsonl."""
        with open(self._metrics_path, "a", buffering=1) as f:
            f.write(json.dumps(record) + "\n")

    def _flush_worker(self) -> None:
        """Drain the queue, writing/posting each record."""
        while True:
            record = self._queue.get()
            if record is None:  # sentinel from close()
                break
            posted = False
            if self.server and self._server_ok:
                posted = self._post_json(
                    "/metrics",
                    {
                        "run_name": self.run_name,
                        "experiment": self.experiment,
                        "record": record,
                    },
                )
            if not posted:
                self._append_local(record)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Enqueue a metrics record (never blocks the training loop)."""
        record: dict[str, Any] = {"step": step, "t": time.time()}
        for k, v in metrics.items():
            key = k if self.tag is None else f"{self.tag}/{k}"
            record[key] = v
        self._queue.put(record)

    def close(self) -> None:
        """Drain the queue, finalize run_meta.json, POST /run/end."""
        # Signal the flush thread to stop and wait for it to drain
        self._queue.put(None)
        self._flush_thread.join()

        # Finalize run_meta.json
        end_time = time.time()
        try:
            with open(self._meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["end_time"] = end_time
        meta["status"] = "completed"
        self._write_meta_atomic(meta)

        # POST /run/end (best-effort)
        if self.server:
            self._post_json(
                "/run/end",
                {
                    "run_name": self.run_name,
                    "experiment": self.experiment,
                    "end_time": end_time,
                    "status": "completed",
                },
            )
