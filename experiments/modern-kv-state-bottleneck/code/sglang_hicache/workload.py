"""Exact-token-length workload for the SGLang path.

Mirrors ``workload/token_workload.py`` (shared prefix + unique suffix,
tiled fallback corpus) but tokenizes through the *server-side*
``/v1/tokenize`` endpoint so the runner needs no local tokenizer.
The ``tokenize_fn`` is injectable for unit tests (no network).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


def checkpoint_tokenize_fallback(
    primary_tokenize: Callable[[str], list[int]], model_path: str
) -> Callable[[str], list[int]]:
    """Prefer server tokenization, falling back to the same checkpoint.

    Gemma's tokenizer uses a ``model_max_length`` sentinel larger than int64.
    SGLang includes it in ``/v1/tokenize`` and ORJSON rejects the response even
    though encoding succeeded. The lazy fallback preserves exact token IDs.
    """
    local_tokenizer = None

    def tokenize(text: str) -> list[int]:
        nonlocal local_tokenizer
        try:
            return primary_tokenize(text)
        except Exception as exc:
            logger.warning(
                "Server /v1/tokenize failed; using tokenizer from %s: %s",
                model_path,
                exc,
            )
            if local_tokenizer is None:
                from transformers import AutoTokenizer

                local_tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True, trust_remote_code=True
                )
            return [
                int(token_id)
                for token_id in local_tokenizer.encode(
                    text, add_special_tokens=False
                )
            ]

    return tokenize

# Same fallback corpus as the vLLM path so cross-runtime workloads are
# comparable when no base_text_path is supplied.
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
class SGLangSegment:
    """One prompt: shared prefix + unique suffix (exact token ids)."""

    prefix_ids: list[int]
    suffix_ids: list[int]
    total_tokens: int

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.prefix_ids + self.suffix_ids


class SGLangWorkload:
    """Builds exact-token-length workloads with shared-prefix structure.

    Identical construction rules to ``workload/token_workload.py``:
    ``prefix_len = int(total_tokens * prefix_ratio)``, suffixes taken at
    distinct offsets from the same corpus.
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[int]],
        total_tokens: int,
        prefix_ratio: float = 0.5,
        n_reps: int = 10,
        base_text: str | None = None,
    ):
        self.total_tokens = total_tokens
        self.prefix_ratio = prefix_ratio
        self.n_reps = n_reps

        self.prefix_len = int(total_tokens * prefix_ratio)
        self.suffix_len = total_tokens - self.prefix_len

        text = base_text or _FALLBACK_TEXT
        all_tokens = tokenize_fn(text)
        corpus_len = len(all_tokens)

        if corpus_len < total_tokens + n_reps * self.suffix_len:
            factor = (total_tokens + n_reps * self.suffix_len) // corpus_len + 2
            all_tokens = all_tokens * factor
            corpus_len = len(all_tokens)
            logger.info(
                "Corpus tiled %dx to %d tokens for total_tokens=%d, n_reps=%d",
                factor, corpus_len, total_tokens, n_reps,
            )

        self.prefix_ids = all_tokens[: self.prefix_len]

        self._suffix_cache: list[list[int]] = []
        for rep in range(n_reps):
            offset = self.prefix_len + rep * self.suffix_len
            if offset + self.suffix_len > corpus_len:
                offset = offset % corpus_len
                if offset + self.suffix_len > corpus_len:
                    offset = 0
            self._suffix_cache.append(all_tokens[offset : offset + self.suffix_len])

        logger.info(
            "SGLang workload built: total=%d, prefix=%d, suffix=%d, reps=%d",
            total_tokens, self.prefix_len, self.suffix_len, n_reps,
        )

    def get_segment(self, rep: int) -> SGLangSegment:
        suffix = self._suffix_cache[rep % len(self._suffix_cache)]
        return SGLangSegment(
            prefix_ids=self.prefix_ids,
            suffix_ids=suffix,
            total_tokens=self.prefix_len + len(suffix),
        )

    def get_prefix_ids(self) -> list[int]:
        return self.prefix_ids
