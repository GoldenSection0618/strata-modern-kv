"""vLLM engine wrapper with cache-residency control.

Manages vLLM LLM instance lifecycle and implements the three
cache-residency conditions required by Experiment 1:

  - recompute:    prefix caching disabled, cache reset before each request
  - gpu_hit:      prefix caching enabled, prefix warmed up on GPU
  - cpu_hit:      prefix caching enabled, prefix offloaded to CPU before request
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from transformers import AutoTokenizer

from ..profiling.gpu_monitor import GPUMonitor
from ..profiling.vllm_stats import VLLMStatsCollector
from ..profiling.timing import measure_ttft, TimingResult
from ..workload.token_workload import TokenWorkload, WorkloadSegment

logger = logging.getLogger(__name__)

# vLLM imports (deferred to __init__ to allow module import without vllm)
_vllm_available = False


def _import_vllm():
    global _vllm_available
    try:
        from vllm import LLM, SamplingParams
        _vllm_available = True
        return LLM, SamplingParams
    except ImportError as e:
        logger.error("Cannot import vllm: %s", e)
        raise


class VLLMRunner:
    """Wraps a vLLM LLM instance with experiment-specific residency control.

    One runner = one model × one residency mode.  The engine is initialized
    once and reused across all repetitions at a given context length.
    """

    def __init__(
        self,
        model_path: str,
        model_id: str,
        residency_mode: str,
        context_length: int,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 65536,
        block_size: int = 16,
        kv_cache_metrics: bool = True,
        dtype: str = "bfloat16",
        kv_offloading_size: int | None = None,
        cpu_offload_gb: float = 0.0,
        gpu_id: int = 0,
        gpu_monitor_interval_ms: int = 10,
    ):
        LLM, SamplingParams = _import_vllm()

        self.model_path = model_path
        self.model_id = model_id
        self.residency_mode = residency_mode
        self.context_length = context_length
        self.gpu_id = gpu_id

        # Build engine kwargs based on residency mode
        engine_kwargs = dict(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            block_size=block_size,
            kv_cache_metrics=kv_cache_metrics,
            enable_prefix_caching=(residency_mode != "recompute"),
            cpu_offload_gb=cpu_offload_gb,
        )

        if kv_offloading_size is not None:
            engine_kwargs["kv_offloading_size"] = kv_offloading_size

        # CPU-resident hit: need offloading enabled
        if residency_mode == "cpu_hit":
            if kv_offloading_size is not None:
                engine_kwargs["kv_offloading_size"] = kv_offloading_size
            engine_kwargs["kv_offloading_backend"] = "native"

        logger.info(
            "Initializing vLLM: model=%s, mode=%s, ctx=%d, kwargs=%s",
            model_id, residency_mode, context_length,
            {k: v for k, v in engine_kwargs.items() if k != "model"},
        )

        self.llm = LLM(**engine_kwargs)
        self.sampling_params_cls = SamplingParams

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Stats collector
        self.stats_collector = VLLMStatsCollector(self.llm)

        # GPU monitor
        self.gpu_monitor = GPUMonitor(
            gpu_id=gpu_id, interval_ms=gpu_monitor_interval_ms
        )

        # Record engine metadata
        self.engine_config = {
            "model": model_id,
            "residency_mode": residency_mode,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "block_size": block_size,
            "kv_cache_metrics": kv_cache_metrics,
            "dtype": dtype,
            "kv_offloading_size": kv_offloading_size,
            "cpu_offload_gb": cpu_offload_gb,
            "enable_prefix_caching": residency_mode != "recompute",
        }

    def get_sampling_params(self, max_tokens: int = 1):
        """Create SamplingParams for TTFT measurement."""
        return self.sampling_params_cls(
            max_tokens=max_tokens,
            temperature=0.0,  # deterministic
        )

    def reset_prefix_cache(self):
        """Clear the prefix cache."""
        self.stats_collector.reset_prefix_cache()

    def warmup_prefix(self, prefix_ids: list[int]):
        """Send a prefix-only request to populate the cache.

        Used for gpu_hit and cpu_hit modes.
        """
        sp = self.get_sampling_params(max_tokens=1)
        prompt = {"prompt_token_ids": prefix_ids}
        logger.info("Warmup: sending prefix-only request (%d tokens)", len(prefix_ids))
        self.llm.generate([prompt], sp)

        # Verify cache hit
        stats = self.stats_collector.collect()
        if stats.prefix_hits > 0:
            logger.info("Warmup confirmed: prefix_hits=%d", stats.prefix_hits)
        else:
            logger.warning(
                "Warmup: no prefix cache hit detected (queries=%d, hits=%d)",
                stats.prefix_queries, stats.prefix_hits,
            )

    def evict_prefix_to_cpu(self, prefix_ids: list[int]):
        """Apply GPU memory pressure to offload the prefix to CPU.

        Sends a batch of filler requests with distinct long prefixes
        to fill the GPU KV cache and force eviction of the warmup prefix.

        Used for cpu_hit mode only.
        """
        # Generate filler tokens distinct from the real prefix
        # Use a different starting offset to ensure different content
        filler_len = min(len(prefix_ids), 4096)  # don't need huge fillers
        n_fillers = 8

        logger.info("Eviction: sending %d filler requests (%d tokens each)", n_fillers, filler_len)
        sp = self.get_sampling_params(max_tokens=1)

        prompts = []
        for i in range(n_fillers):
            # Distinct filler: token IDs offset by a large prime * i
            offset = (i + 1) * 7919  # prime to create distinct sequences
            filler_ids = [
                (tid + offset) % 32000 for tid in prefix_ids[:filler_len]
            ]
            prompts.append({"prompt_token_ids": filler_ids})

        # Send all fillers at once
        self.llm.generate(prompts, sp)

        # Check if eviction occurred
        stats = self.stats_collector.collect()
        logger.info(
            "After eviction: usage=%.3f, preempted_reqs=%d, preempted_hits=%d",
            stats.usage, stats.prefix_preempted_requests, stats.prefix_preempted_hits,
        )

    def reset_state(self):
        """Restore residency initial condition before each measured request."""
        if self.residency_mode == "recompute":
            self.reset_prefix_cache()
        elif self.residency_mode == "gpu_hit":
            self.reset_prefix_cache()
            # Warmup will be called separately before measurement
        elif self.residency_mode == "cpu_hit":
            self.reset_prefix_cache()
            # Warmup + eviction will be called separately

    def measure_request(self, segment: WorkloadSegment) -> TimingResult:
        """Send the measured request and collect all instrumentation."""
        sp = self.get_sampling_params(max_tokens=1)
        prompt = segment.to_tokens_prompt()

        result = measure_ttft(
            self.llm,
            prompt,
            sp,
            self.gpu_monitor,
            self.stats_collector,
        )

        return result

    def get_metadata(self) -> dict:
        """Return engine and runtime metadata."""
        try:
            import vllm
            vllm_version = vllm.__version__
        except Exception:
            vllm_version = "unknown"

        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "residency_mode": self.residency_mode,
            "context_length": self.context_length,
            "engine_config": self.engine_config,
            "vllm_version": vllm_version,
            "stats_available": self.stats_collector.is_available(),
        }

    def cleanup(self):
        """Release GPU resources."""
        del self.llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
