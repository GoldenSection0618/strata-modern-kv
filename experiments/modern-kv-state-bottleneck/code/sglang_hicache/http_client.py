"""Stdlib-only HTTP client for SGLang's native endpoints.

Talks to the public SGLang HTTP surface:

* ``GET  /health``       — readiness probe
* ``POST /flush_cache``  — reset radix cache (L1 GPU + L2 host)
* ``POST /generate``     — native generation with exact ``input_ids``
* ``GET  /metrics``      — Prometheus text (with ``--enable-metrics``)

No third-party packages are required (urllib only).  The client always
bypasses the environment proxy so that localhost traffic to the SGLang
server never leaves the node.

TTFT semantics: SGLang's non-streaming ``/generate`` returns after the
whole request completes.  With ``max_new_tokens=1`` the first token IS
the completion, so request latency equals TTFT.  The load driver adds a
client-side arrival timestamp to separate queueing from service time.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

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
        # Never route localhost traffic through the environment proxy.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    # -- low-level ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=timeout_s or self.timeout_s) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise SGLangHTTPError(f"HTTP {method} {url} failed: {e}") from e

    # -- public endpoints ---------------------------------------------------

    def health(self, timeout_s: float = 10.0) -> bool:
        """True when the server answers /health with 2xx."""
        status, _ = self._request("GET", "/health", timeout_s=timeout_s)
        return 200 <= status < 300

    def tokenize(self, text: str, timeout_s: float = 120.0) -> list[int]:
        """Tokenize text with the *server-side* tokenizer (/v1/tokenize).

        Uses ``add_special_tokens=False`` to match the vLLM workload
        construction (``tokenizer.encode(text, add_special_tokens=False)``).
        """
        payload = {"prompt": text, "add_special_tokens": False}
        status, body = self._request("POST", "/v1/tokenize", payload, timeout_s=timeout_s)
        if not (200 <= status < 300):
            raise SGLangHTTPError(
                f"/v1/tokenize returned {status}: {body.decode('utf-8', 'replace')[:300]}"
            )
        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            raise SGLangHTTPError(f"invalid /v1/tokenize response: {e}") from e
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise SGLangHTTPError(
                f"/v1/tokenize response missing tokens list: {str(data)[:200]}"
            )
        return [int(t) for t in tokens]

    def model_info(self, timeout_s: float = 30.0) -> dict:
        """Return the public /model_info payload (model path, arch)."""
        status, body = self._request("GET", "/model_info", timeout_s=timeout_s)
        if not (200 <= status < 300):
            raise SGLangHTTPError(
                f"/model_info returned {status}: {body.decode('utf-8', 'replace')[:300]}"
            )
        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            raise SGLangHTTPError(f"invalid /model_info response: {e}") from e
        return data if isinstance(data, dict) else {}

    def server_info(self, timeout_s: float = 30.0) -> dict:
        """Return the public /server_info payload (resolved server args)."""
        status, body = self._request("GET", "/server_info", timeout_s=timeout_s)
        if not (200 <= status < 300):
            raise SGLangHTTPError(
                f"/server_info returned {status}: {body.decode('utf-8', 'replace')[:300]}"
            )
        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            raise SGLangHTTPError(f"invalid /server_info response: {e}") from e
        return data if isinstance(data, dict) else {}

    def flush_cache(self, timeout_s: float = 60.0) -> bool:
        """Flush L1/L2 after waiting for the scheduler to become idle.

        The pinned SGLang endpoint accepts a ``timeout`` query parameter and
        defers the flush inside the scheduler.  This matters after long
        chunked-prefill requests: HTTP generation may have returned while
        hybrid-cache bookkeeping is still pending even though queue/running
        request counts are already zero.
        """
        path = f"/flush_cache?timeout={float(timeout_s):g}"
        status, body = self._request(
            "POST", path, timeout_s=timeout_s + 10.0
        )
        if not (200 <= status < 300):
            raise SGLangHTTPError(
                f"flush_cache returned {status}: {body.decode('utf-8', 'replace')[:300]}"
            )
        return True

    def fetch_metrics_text(self, timeout_s: float = 30.0) -> str:
        """Return the raw Prometheus text from /metrics."""
        status, body = self._request("GET", "/metrics", timeout_s=timeout_s)
        if not (200 <= status < 300):
            raise SGLangHTTPError(
                f"/metrics returned {status}: {body.decode('utf-8', 'replace')[:300]}"
            )
        return body.decode("utf-8", "replace")

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 1,
        temperature: float = 0.0,
        ignore_eos: bool = True,
        return_logprob: bool = True,
        top_logprobs_num: int = 1,
        request_id: int = 0,
        timeout_s: float | None = None,
    ) -> GenerateResult:
        """Send one native /generate request and return a timed result.

        ``return_logprob=True`` makes the response carry the exact output
        token id in ``meta_info.output_token_logprobs`` (used by the
        prefix-consistency validation).
        """
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "ignore_eos": ignore_eos,
            },
            "stream": False,
            "return_logprob": return_logprob,
            "top_logprobs_num": top_logprobs_num,
        }
        t_send = time.perf_counter()
        status, body = self._request("POST", "/generate", payload, timeout_s=timeout_s)
        t_complete = time.perf_counter()

        result = GenerateResult(
            request_id=request_id,
            t_send=t_send,
            t_first_token=t_complete,
            t_complete=t_complete,
            status=status,
        )
        if not (200 <= status < 300):
            result.error = body.decode("utf-8", "replace")[:500]
            logger.warning("generate request %d failed: status=%d %s",
                           request_id, status, result.error[:200])
            return result

        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            result.error = f"invalid JSON response: {e}"
            return result

        if not isinstance(data, dict):
            result.error = f"unexpected response shape: {type(data).__name__}"
            return result

        if "error" in data:
            result.error = str(data["error"])[:500]
            return result

        result.ok = True
        result.text = str(data.get("text", ""))
        meta = data.get("meta_info") or {}
        if isinstance(meta, dict):
            (
                result.prompt_tokens,
                result.completion_tokens,
                result.cached_tokens,
                result.cached_tokens_details,
                result.output_token_id,
            ) = _extract_meta(meta)
        return result
