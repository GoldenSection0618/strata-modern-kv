"""Focused tests for the public SGLang HTTP client."""

from hcv.http_client import SGLangHTTPClient


def _client_with_health(status: int, body: str) -> SGLangHTTPClient:
    client = SGLangHTTPClient("http://127.0.0.1:1")
    client._get_text = lambda path: (status, body)
    return client


def test_health_accepts_empty_200_body_from_pinned_sglang():
    assert _client_with_health(200, "").health() is True


def test_health_rejects_non_200_even_with_healthy_body():
    assert _client_with_health(503, "healthy").health() is False
