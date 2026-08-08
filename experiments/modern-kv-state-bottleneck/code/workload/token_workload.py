"""Exact-token-length workload construction for Experiment 1.

Builds prompt_token_ids with a shared prefix + unique suffix structure.
Uses TokensPrompt for vLLM to bypass tokenization variability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Fallback corpus: a long technical text that tokenizes to 32K+ tokens
# under most tokenizers.  In production, load from base_text_path;
# this fallback ensures the code is runnable without external files.
_FALLBACK_TEXT = (
    "The rapid evolution of large language models has fundamentally changed "
    "the landscape of natural language processing. Modern architectures employ "
    "hybrid attention mechanisms that combine recurrent layers, sliding-window "
    "attention, and full attention to achieve long-context understanding while "
    "managing computational costs. The key-value cache, traditionally used to "
    "store attention tensors for reuse during autoregressive generation, must "
    "now accommodate multiple types of state: attention key-value pairs, "
    "sliding-window local caches with bounded size, and recurrent state "
    "representations that do not grow linearly with sequence length. This "
    "heterogeneity complicates the design of hierarchical caching systems, "
    "which aim to offload less-recently-used state to CPU memory or NVMe "
    "storage and restore it when needed. The cost of restoration, including "
    "CPU-to-GPU transfer latency and potential pipeline stalls, must be "
    "weighed against the cost of recomputation from scratch. Strata and "
    "related work proposed layered context caching to address this tradeoff "
    "for dense-attention models, but the applicability of these techniques "
    "to modern hybrid architectures remains an open question. "
) * 100  # Repeat to reach ~35K+ tokens


@dataclass
class WorkloadSegment:
    """A single prompt segment for one measurement repetition."""

    prefix_ids: list[int]
    suffix_ids: list[int]
    total_tokens: int

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.prefix_ids + self.suffix_ids

    def to_tokens_prompt(self) -> dict:
        """Return vLLM TokensPrompt dict."""
        return {"prompt_token_ids": self.prompt_token_ids}

    def to_prefix_only_prompt(self) -> dict:
        """Return prefix-only prompt for warmup."""
        return {"prompt_token_ids": self.prefix_ids}


class TokenWorkload:
    """Builds exact-token-length workloads with shared-prefix structure.

    For context length L with prefix_ratio r:
      - prefix length = int(L * r)
      - suffix length = L - prefix length
      - prefix is identical across all repetitions at the same context length
      - suffix varies by repetition (different offset into the corpus)
    """

    def __init__(
        self,
        tokenizer,
        total_tokens: int,
        prefix_ratio: float = 0.5,
        n_reps: int = 10,
        base_text: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.total_tokens = total_tokens
        self.prefix_ratio = prefix_ratio
        self.n_reps = n_reps

        self.prefix_len = int(total_tokens * prefix_ratio)
        self.suffix_len = total_tokens - self.prefix_len

        # Tokenize base corpus
        text = base_text or _FALLBACK_TEXT
        all_tokens = tokenizer.encode(text, add_special_tokens=False)
        corpus_len = len(all_tokens)

        if corpus_len < total_tokens + n_reps * self.suffix_len:
            # Tile the corpus to ensure enough tokens
            factor = (total_tokens + n_reps * self.suffix_len) // corpus_len + 2
            all_tokens = all_tokens * factor
            corpus_len = len(all_tokens)
            logger.info(
                "Corpus tiled %dx to %d tokens for total_tokens=%d, n_reps=%d",
                factor, corpus_len, total_tokens, n_reps,
            )

        # Shared prefix: first prefix_len tokens
        self.prefix_ids = all_tokens[: self.prefix_len]

        # Unique suffixes: different offsets for each rep
        self._suffix_cache: list[list[int]] = []
        for rep in range(n_reps):
            offset = self.prefix_len + rep * self.suffix_len
            # Wrap around if needed
            if offset + self.suffix_len > corpus_len:
                offset = offset % corpus_len
                if offset + self.suffix_len > corpus_len:
                    offset = 0
            suffix = all_tokens[offset : offset + self.suffix_len]
            self._suffix_cache.append(suffix)

        logger.info(
            "Workload built: total=%d, prefix=%d, suffix=%d, reps=%d",
            total_tokens, self.prefix_len, self.suffix_len, n_reps,
        )

    def get_segment(self, rep: int) -> WorkloadSegment:
        """Return the workload segment for the given repetition index."""
        suffix = self._suffix_cache[rep % len(self._suffix_cache)]
        return WorkloadSegment(
            prefix_ids=self.prefix_ids,
            suffix_ids=suffix,
            total_tokens=self.prefix_len + len(suffix),
        )

    def get_prefix_ids(self) -> list[int]:
        """Return the shared prefix token IDs."""
        return self.prefix_ids
