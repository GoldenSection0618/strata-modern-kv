"""Async request load driver for Experiment 3.

Generates requests at a target arrival rate using a Poisson process,
sends them concurrently to vLLM, and collects per-request timing.

The driver respects a concurrency ceiling to avoid unbounded queueing
inside vLLM's internal scheduler.  When the ceiling is reached, new
requests are held in a client-side queue and released as in-flight
requests complete.  The time spent in this client-side queue is
recorded as `queueing_ms`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """Per-request measurement record."""

    request_id: int
    # Offered arrival time (when the request was generated)
    t_arrival: float
    # Time the request actually started being processed (after queueing)
    t_start: float
    # Time the first token was received
    t_first_token: float
    # Time the request completed (all tokens generated)
    t_complete: float

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
        }


@dataclass
class LoadDriverConfig:
    """Configuration for the async load driver."""

    offered_rate: float          # requests per second
    n_requests: int              # total requests to send
    concurrency_ceiling: int = 64
    # Inter-arrival distribution: "poisson" or "constant"
    arrival_pattern: str = "poisson"
    # Random seed for reproducibility
    seed: int = 42


class AsyncLoadDriver:
    """Drives concurrent requests at a target arrival rate.

    Usage:
        driver = AsyncLoadDriver(llm, sampling_params, prompts, config)
        records = await driver.run()
    """

    def __init__(
        self,
        llm,
        sampling_params,
        prompts: list[dict],
        config: LoadDriverConfig,
    ):
        self.llm = llm
        self.sampling_params = sampling_params
        self.prompts = prompts
        self.config = config
        self._rng = random.Random(config.seed)

        # Pre-compute arrival times
        self._arrival_times = self._generate_arrivals()

    def _generate_arrivals(self) -> list[float]:
        """Generate arrival timestamps (relative to t=0)."""
        n = self.config.n_requests
        rate = self.config.offered_rate

        if self.config.arrival_pattern == "constant":
            interval = 1.0 / rate
            return [i * interval for i in range(n)]
        else:  # poisson
            arrivals = []
            t = 0.0
            for _ in range(n):
                # Exponential inter-arrival with mean = 1/rate
                t += self._rng.expovariate(rate)
                arrivals.append(t)
            return arrivals

    async def run(self) -> list[RequestRecord]:
        """Execute the load test and return per-request records."""
        records: list[RequestRecord] = []
        semaphore = asyncio.Semaphore(self.config.concurrency_ceiling)
        t0 = time.perf_counter()

        async def send_one(req_id: int, arrival_time: float, prompt: dict):
            # Wait until the scheduled arrival time
            target = t0 + arrival_time
            now = time.perf_counter()
            if target > now:
                await asyncio.sleep(target - now)

            t_arrival = time.perf_counter()

            # Acquire concurrency slot
            async with semaphore:
                t_start = time.perf_counter()

                # Use vLLM's async generate (via thread pool for sync LLM)
                loop = asyncio.get_event_loop()
                t_first_token_holder = {"val": None}

                def _generate():
                    outputs = self.llm.generate([prompt], self.sampling_params)
                    return outputs

                # vLLM's LLM.generate is synchronous; run in executor
                # We approximate t_first_token as the completion time
                # since max_tokens=1 means first token = completion.
                outputs = await loop.run_in_executor(None, _generate)

                t_first_token = time.perf_counter()
                t_complete = t_first_token  # max_tokens=1: same

            record = RequestRecord(
                request_id=req_id,
                t_arrival=t_arrival,
                t_start=t_start,
                t_first_token=t_first_token,
                t_complete=t_complete,
            )
            records.append(record)

        # Schedule all requests
        tasks = []
        for i, arrival in enumerate(self._arrival_times):
            prompt = self.prompts[i % len(self.prompts)]
            tasks.append(asyncio.create_task(send_one(i, arrival, prompt)))

        await asyncio.gather(*tasks)

        # Sort by request_id for deterministic output
        records.sort(key=lambda r: r.request_id)
        return records


def summarize_records(records: list[RequestRecord]) -> dict:
    """Compute summary statistics from request records."""
    if not records:
        return {"error": "no records"}

    ttfts = sorted(r.ttft_ms for r in records)
    queueings = sorted(r.queueing_ms for r in records)
    services = sorted(r.service_ms for r in records)
    n = len(ttfts)

    def percentile(sorted_list, p):
        idx = int(len(sorted_list) * p)
        return sorted_list[min(idx, len(sorted_list) - 1)]

    # Active concurrency: max overlap of [t_start, t_first_token] intervals
    events = []
    for r in records:
        events.append((r.t_start, 1))
        events.append((r.t_first_token, -1))
    events.sort()
    max_concurrency = 0
    current = 0
    for _, delta in events:
        current += delta
        max_concurrency = max(max_concurrency, current)

    # Achieved throughput: completed requests / total time span
    t_min = min(r.t_arrival for r in records)
    t_max = max(r.t_complete for r in records)
    duration = t_max - t_min
    achieved_throughput = len(records) / duration if duration > 0 else 0

    return {
        "n_requests": n,
        "offered_rate": len(records) / duration if duration > 0 else 0,
        "achieved_throughput": round(achieved_throughput, 3),
        "active_concurrency_max": max_concurrency,
        "active_concurrency_mean": round(
            sum(services) / max(duration * 1000, 1) * n, 2
        ),
        "ttft_p50_ms": round(percentile(ttfts, 0.50), 3),
        "ttft_p90_ms": round(percentile(ttfts, 0.90), 3),
        "ttft_p99_ms": round(percentile(ttfts, 0.99), 3),
        "ttft_min_ms": round(ttfts[0], 3),
        "ttft_max_ms": round(ttfts[-1], 3),
        "ttft_mean_ms": round(sum(ttfts) / n, 3),
        "queueing_p50_ms": round(percentile(queueings, 0.50), 3),
        "queueing_p90_ms": round(percentile(queueings, 0.90), 3),
        "queueing_p99_ms": round(percentile(queueings, 0.99), 3),
        "queueing_mean_ms": round(sum(queueings) / n, 3),
        "service_p50_ms": round(percentile(services, 0.50), 3),
        "service_p90_ms": round(percentile(services, 0.90), 3),
        "service_mean_ms": round(sum(services) / n, 3),
        "duration_s": round(duration, 3),
    }
