"""Stdlib-only HTTP client for SGLang's public endpoints.

Talks to the public SGLang HTTP surface only:

* ``GET  /health``        — readiness probe
* ``POST /flush_cache``   — reset radix cache (L1 GPU + L2 host)
* ``POST /generate``      — native generation with exact ``input_ids``
* ``GET  /metrics``       — Prometheus text (with ``--enable-metrics``)
* ``GET  /get_model_info``— model id / version metadata

No third-party packages are required (urllib only).  The client always
bypasses the environment proxy so localhost traffic never leaves the
node.

TTFT semantics: SGLang's non-streaming ``/generate`` returns after the
whole request completes.  With ``max_new_tokens=1`` the first token is
the completion, so request latency equals TTFT.  The load driver adds a
client-side arrival timestamp to separate queueing from service time.

The pinned Qwen hybrid native ``/generate`` may omit
``cached_tokens_details``; per-tier counters isolated around each
request are authoritative when request metadata is missing.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from hcv.metrics import PrometheusScrape, parse_prometheus_text

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600.0

#: SGLang native generate response meta_info keys we record.
META_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cached_tokens_details",
    "reasoning_tokens",
)


class SGLangHTTPError(Exception):
    """Raised for non-2xx responses or transport failures."""


@dataclass
class GenerateResult:
    """Timed result of one native /generate call (max_new_tokens=1).

    ``t_first_token == t_complete`` because the non-streaming endpoint
    returns the single output token as the whole response.
    """

    request_id: int
    t_send: float
    t_first_token: float
    t_complete: float
    status: int = 0
    ok: bool = False
    error: str = ""
    text: str = ""
    meta_info: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cached_tokens_details: Optional[dict] = None
    output_token_id: Optional[int] = None

    @property
    def ttft_ms(self) -> float:
        return (self.t_first_token - self.t_send) * 1000

    @property
    def total_ms(self) -> float:
        return (self.t_complete - self.t_send) * 1000

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "ok": self.ok,
            "error": self.error,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cached_tokens_details": self.cached_tokens_details,
            "output_token_id": self.output_token_id,
            "ttft_ms": round(self.ttft_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


def _extract_meta(meta: dict) -> tuple[int, int, int, Optional[dict], Optional[int]]:
    """Extract (prompt_tokens, completion_tokens, cached_tokens,
    cached_tokens_details, output_token_id) from a /generate meta_info."""
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    completion_tokens = int(meta.get("completion_tokens", 0) or 0)
    cached_tokens = int(meta.get("cached_tokens", 0) or 0)
    details = meta.get("cached_tokens_details")
    if details is not None and not isinstance(details, dict):
        details = None

    output_token_id = None
    logprobs = meta.get("output_token_logprobs")
    if logprobs and isinstance(logprobs, list) and logprobs[0]:
        entry = logprobs[0]
        # entry format: (logprob, token_id, text-or-None)
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            try:
                output_token_id = int(entry[1])
            except (TypeError, ValueError):
                output_token_id = None
    return prompt_tokens, completion_tokens, cached_tokens, details, output_token_id


class SGLangHTTPClient:
    """Minimal stdlib HTTP client for a local SGLang server."""

    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # Bypass the cluster proxy for localhost traffic.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # -- low-level ----------------------------------------------------------

    def _request_json(self, method: str, path: str, payload: Optional[dict]) -> tuple[int, dict]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                body = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            body = e.read()
            status = e.code
            try:
                parsed = json.loads(body.decode("utf-8", errors="replace"))
            except ValueError:
                parsed = {"error": body.decode("utf-8", errors="replace")[:500]}
            return status, parsed
        except urllib.error.URLError as e:
            raise SGLangHTTPError(f"transport error {path}: {e}") from e
        try:
            parsed = json.loads(body.decode("utf-8", errors="replace"))
        except ValueError:
            parsed = {}
        return status, parsed

    def _get_text(self, path: str) -> tuple[int, str]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            raise SGLangHTTPError(f"transport error {path}: {e}") from e

    # -- public endpoints ---------------------------------------------------

    def health(self) -> bool:
        """True when the pinned SGLang ``/health`` endpoint returns 200.

        The endpoint's successful response body is empty in the pinned
        4ad990ba7 build, so readiness must not require an ``OK`` string.
        """
        status, _body = self._get_text("/health")
        return status == 200

    def flush_cache(self) -> bool:
        """POST /flush_cache; True on 200."""
        status, _ = self._request_json("POST", "/flush_cache", None)
        return status == 200

    def get_model_info(self) -> dict:
        """GET /get_model_info; returns the JSON payload."""
        status, parsed = self._request_json("GET", "/get_model_info", None)
        if status != 200:
            raise SGLangHTTPError(f"/get_model_info returned {status}")
        return parsed

    def scrape_metrics(self) -> PrometheusScrape:
        """GET /metrics and parse the Prometheus text."""
        status, text = self._get_text("/metrics")
        if status != 200:
            raise SGLangHTTPError(f"/metrics returned {status}")
        return parse_prometheus_text(text)

    def generate(
        self,
        request_id: int,
        input_ids: list[int],
        max_new_tokens: int = 1,
        sampling_params: Optional[dict] = None,
        return_logprob: bool = False,
    ) -> GenerateResult:
        """Native /generate with exact ``input_ids``.

        For text-only traces the runner converts deterministic pseudo-token
        text into integer IDs with a local deterministic encoder (see
        ``hcv.workload``); the IDs are stable across runs so the server
        sees byte-identical prompts.
        """
        t_send = time.time()
        payload: dict = {
            "input_ids": [int(i) for i in input_ids],
            "sampling_params": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": 0.0,
                **(sampling_params or {}),
            },
        }
        if return_logprob:
            payload["return_logprob"] = True
            payload["top_logprobs_num"] = 1
        status, parsed = self._request_json("POST", "/generate", payload)
        t_complete = time.time()

        res = GenerateResult(request_id=request_id, t_send=t_send,
                             t_first_token=t_complete, t_complete=t_complete)
        res.status = status
        if status != 200:
            res.error = str(parsed)[:500]
            return res
        res.ok = True
        res.text = str(parsed.get("text", [""])[0] if isinstance(parsed.get("text"), list) else parsed.get("text", ""))
        meta = parsed.get("meta_info") or {}
        res.meta_info = {k: meta.get(k) for k in META_KEYS}
        prompt_tokens, completion_tokens, cached_tokens, details, out_id = _extract_meta(meta)
        res.prompt_tokens = prompt_tokens
        res.completion_tokens = completion_tokens
        res.cached_tokens = cached_tokens
        res.cached_tokens_details = details
        res.output_token_id = out_id
        return res

    def generate_text(
        self,
        request_id: int,
        text: str,
        max_new_tokens: int = 1,
        sampling_params: Optional[dict] = None,
    ) -> GenerateResult:
        """Native /generate with a raw text prompt (auxiliary path)."""
        t_send = time.time()
        payload: dict = {
            "text": text,
            "sampling_params": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": 0.0,
                **(sampling_params or {}),
            },
        }
        status, parsed = self._request_json("POST", "/generate", payload)
        t_complete = time.time()
        res = GenerateResult(request_id=request_id, t_send=t_send,
                             t_first_token=t_complete, t_complete=t_complete)
        res.status = status
        if status != 200:
            res.error = str(parsed)[:500]
            return res
        res.ok = True
        res.text = str(parsed.get("text", [""])[0] if isinstance(parsed.get("text"), list) else parsed.get("text", ""))
        meta = parsed.get("meta_info") or {}
        res.meta_info = {k: meta.get(k) for k in META_KEYS}
        prompt_tokens, completion_tokens, cached_tokens, details, out_id = _extract_meta(meta)
        res.prompt_tokens = prompt_tokens
        res.completion_tokens = completion_tokens
        res.cached_tokens = cached_tokens
        res.cached_tokens_details = details
        res.output_token_id = out_id
        return res
