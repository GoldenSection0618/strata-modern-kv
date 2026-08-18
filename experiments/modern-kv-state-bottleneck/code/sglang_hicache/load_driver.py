"""Concurrent HTTP load driver for SGLang Experiment 3.

Drives the *public HTTP server* with concurrent ``/generate`` requests
(rather than calling a synchronous in-process engine from multiple
threads).  Stdlib only: ``concurrent.futures.ThreadPoolExecutor`` +
``urllib``.

Timing separation (per ``00-measurement-conventions.md``):

    TTFT = queueing + service time
    queueing_ms = t_start - t_arrival   (client-side wait for a slot)
    service_ms  = t_first_token - t_start
    ttft_ms     = t_first_token - t_arrival

``t_first_token == t_complete`` because every request uses
``max_new_tokens=1`` (first token is the whole completion).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from sglang_hicache.http_client import SGLangHTTPClient

logger = logging.getLogger(__name__)


@dataclass
class LoadDriverConfig:
    """Configuration for the concurrent HTTP load driver."""

    offered_rate: float          # requests per second
    n_requests: int              # total requests to send
    concurrency_ceiling: int = 64
    arrival_pattern: str = "poisson"  # "poisson" | "constant"
    seed: int = 42
    request_timeout_s: float = 900.0


@dataclass
class HttpRequestRecord:
    """Per-request measurement record (vLLM RequestRecord-compatible keys)."""

    request_id: int
    t_arrival: float
    t_start: float
    t_first_token: float
    t_complete: float
    ok: bool = False
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cached_tokens_details: Optional[dict] = None
    output_token_id: Optional[int] = None

    @property
    def queueing_ms(self) -> float:
        return (self.t_start - self.t_arrival) * 1000

    @property
    def ttft_ms(self) -> float:
        return (self.t_first_token - self.t_arrival) * 1000

    @property
    def service_ms(self) -> float:
        return (self.t_first_token - self.t_start) * 1000

    @property
    def total_ms(self) -> float:
        return (self.t_complete - self.t_arrival) * 1000

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "t_arrival": self.t_arrival,
            "t_start": self.t_start,
            "t_first_token": self.t_first_token,
            "t_complete": self.t_complete,
            "queueing_ms": round(self.queueing_ms, 3),
            "ttft_ms": round(self.ttft_ms, 3),
            "service_ms": round(self.service_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "ok": self.ok,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cached_tokens_details": self.cached_tokens_details,
            "output_token_id": self.output_token_id,
        }


class HttpLoadDriver:
    """Send requests at a target arrival rate with a concurrency ceiling."""

    def __init__(
        self,
        client: SGLangHTTPClient,
        prompts: list[list[int]],
        config: LoadDriverConfig,
    ):
        self.client = client
        self.prompts = prompts
        self.config = config
        self._rng = random.Random(config.seed)
        self._arrival_times = self._generate_arrivals()

    def _generate_arrivals(self) -> list[float]:
        n = self.config.n_requests
        rate = self.config.offered_rate
        if self.config.arrival_pattern == "constant":
            interval = 1.0 / rate if rate > 0 else 0.0
            return [i * interval for i in range(n)]
        arrivals = []
        t = 0.0
        for _ in range(n):
            t += self._rng.expovariate(rate) if rate > 0 else 0.0
            arrivals.append(t)
        return arrivals

    def run(self) -> list[HttpRequestRecord]:
        """Execute the load test; returns per-request records sorted by id.

        Semantics mirror the vLLM path's asyncio driver: every request is a
        concurrent worker that (1) sleeps to its scheduled arrival, (2)
        records ``t_arrival``, (3) contends on the concurrency-ceiling
        semaphore (this wait IS ``queueing_ms = t_start - t_arrival``), and
        (4) performs the HTTP call.  The thread pool is sized to the number
        of requests so it never queues work itself; the semaphore is the
        real concurrency limiter.
        """
        records: list[HttpRequestRecord] = []
        lock = threading.Lock()
        t0 = time.perf_counter()
        permits = threading.BoundedSemaphore(self.config.concurrency_ceiling)
        executor = ThreadPoolExecutor(max_workers=max(len(self._arrival_times), 1))

        def worker(req_id: int, scheduled: float, prompt: list[int]):
            target = t0 + scheduled
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            t_arrival = time.perf_counter()

            acquired = permits.acquire(timeout=self.config.request_timeout_s)
            if not acquired:
                with lock:
                    records.append(
                        HttpRequestRecord(
                            request_id=req_id,
                            t_arrival=t_arrival,
                            t_start=t_arrival,
                            t_first_token=t_arrival,
                            t_complete=t_arrival,
                            ok=False,
                            error="timed out waiting for concurrency slot",
                        )
                    )
                return
            try:
                t_start = time.perf_counter()
                result = self.client.generate(
                    prompt,
                    max_new_tokens=1,
                    request_id=req_id,
                    timeout_s=self.config.request_timeout_s,
                )
                t_complete = time.perf_counter()
                record = HttpRequestRecord(
                    request_id=req_id,
                    t_arrival=t_arrival,
                    t_start=t_start,
                    t_first_token=t_complete,  # max_new_tokens=1: first == complete
                    t_complete=t_complete,
                    ok=result.ok,
                    error=result.error,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cached_tokens=result.cached_tokens,
                    cached_tokens_details=result.cached_tokens_details,
                    output_token_id=result.output_token_id,
                )
            except Exception as e:  # noqa: BLE001 - record per-request failures
                t_complete = time.perf_counter()
                record = HttpRequestRecord(
                    request_id=req_id,
                    t_arrival=t_arrival,
                    t_start=t_start,
                    t_first_token=t_complete,
                    t_complete=t_complete,
                    ok=False,
                    error=str(e)[:300],
                )
            finally:
                permits.release()
            with lock:
                records.append(record)

        futures = [
            executor.submit(worker, i, arrival, self.prompts[i % len(self.prompts)])
            for i, arrival in enumerate(self._arrival_times)
        ]
        for f in futures:
            f.result()
        executor.shutdown(wait=True)

        records.sort(key=lambda r: r.request_id)
        return records
