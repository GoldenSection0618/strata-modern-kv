"""GPU memory monitoring via pynvml.

Background thread samples GPU memory at fixed intervals.
Used to detect CPU→GPU cache restore transfers during request execution.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class GPUMonitor:
    """High-frequency GPU memory sampler.

    Usage:
        monitor = GPUMonitor(gpu_id=0, interval_ms=10)
        monitor.start()
        # ... run request ...
        samples = monitor.stop()
        # samples: list of (timestamp_sec, used_bytes)
    """

    def __init__(self, gpu_id: int = 0, interval_ms: int = 10):
        self.gpu_id = gpu_id
        self.interval = interval_ms / 1000.0
        self.samples: list[tuple[float, int]] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start sampling in a background thread."""
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.debug("GPUMonitor started: gpu=%d, interval=%.1fms", self.gpu_id, self.interval * 1000)

    def stop(self) -> list[tuple[float, int]]:
        """Stop sampling and return collected samples."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("GPUMonitor thread did not join cleanly")
            self._thread = None
        logger.debug("GPUMonitor stopped: %d samples collected", len(self.samples))
        return self.samples

    def _sample_loop(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

            while self._running:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append((time.perf_counter(), mem.used))
                time.sleep(self.interval)

            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error("GPUMonitor sampling error: %s", e)
            self._running = False

    @staticmethod
    def analyze_samples(
        samples: list[tuple[float, int]],
        t_start: float,
        t_end: float,
    ) -> dict:
        """Analyze GPU memory samples within a time window.

        Returns:
            dict with:
              - min_used_mb: minimum GPU memory used (MB)
              - max_used_mb: maximum GPU memory used (MB)
              - delta_mb: max - min (MB), approximate transfer volume
              - restore_duration_ms: duration of memory increase (ms)
              - restore_bandwidth_mbs: delta / duration (MB/s), or None
              - n_samples: number of samples in window
        """
        window = [(t, b) for t, b in samples if t_start <= t <= t_end]
        if not window:
            return {
                "min_used_mb": 0,
                "max_used_mb": 0,
                "delta_mb": 0,
                "restore_duration_ms": 0,
                "restore_bandwidth_mbs": None,
                "n_samples": 0,
            }

        used_mb = [b / (1024 * 1024) for _, b in window]
        min_mb = min(used_mb)
        max_mb = max(used_mb)
        delta_mb = max_mb - min_mb

        # Find restore window: from first rising point to peak
        if delta_mb < 1.0:  # Less than 1 MB change — no meaningful restore
            restore_duration_ms = 0.0
            restore_bandwidth = None
        else:
            # Find start (first sample below min + 10% of delta)
            threshold = min_mb + 0.1 * delta_mb
            start_idx = 0
            for i, mb in enumerate(used_mb):
                if mb > threshold:
                    start_idx = max(0, i - 1)
                    break
            peak_idx = used_mb.index(max_mb)
            if peak_idx > start_idx:
                restore_duration_ms = (
                    window[peak_idx][0] - window[start_idx][0]
                ) * 1000
                if restore_duration_ms > 0:
                    restore_bandwidth = (delta_mb / restore_duration_ms) * 1000  # MB/s
                else:
                    restore_bandwidth = None
            else:
                restore_duration_ms = 0.0
                restore_bandwidth = None

        return {
            "min_used_mb": round(min_mb, 2),
            "max_used_mb": round(max_mb, 2),
            "delta_mb": round(delta_mb, 2),
            "restore_duration_ms": round(restore_duration_ms, 2),
            "restore_bandwidth_mbs": (
                round(restore_bandwidth, 2) if restore_bandwidth else None
            ),
            "n_samples": len(window),
        }
