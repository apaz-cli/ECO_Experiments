"""Adapter that wires mlsweep.logger.MLSweepLogger to torchtitan's JobConfig API.

The underlying mlsweep logger reads MLSWEEP_RUN_NAME from env automatically.
"""
from mlsweep.logger import MLSweepLogger as _MLSweepLogger


class MLSweepLogger:
    """Wraps mlsweep.logger.MLSweepLogger for torchtitan's JobConfig-based metrics API."""

    def __init__(self, log_dir: str, job_config, tag: str | None = None):
        self._logger = _MLSweepLogger(hparams=job_config.to_dict())

    def log(self, metrics: dict, step: int) -> None:
        self._logger.log(metrics, step)

    def close(self) -> None:
        self._logger.close()


__all__ = ["MLSweepLogger"]
