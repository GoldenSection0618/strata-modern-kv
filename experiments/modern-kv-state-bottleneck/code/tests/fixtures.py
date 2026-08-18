"""Shared fixtures for the SGLang-path unit tests.

All fixtures are synthetic and self-contained: no CUDA, no SGLang
installation, no network, no model weights.
"""

from __future__ import annotations

# A realistic Prometheus scrape fragment for the metrics used by the
# cache-stat snapshot (label sets trimmed to what the parser consumes).
PROM_TEXT_BASE = """\
# HELP sglang:prefill_effective_tokens_total Effective prefill tokens
# TYPE sglang:prefill_effective_tokens_total counter
sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="input"} 100
sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="device_hit"} 50
sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="host_hit"} 10
sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="storage_hit"} 0
sglang:load_back_tokens_total{model_name="M",tp_rank="0",pool="kv"} 20
sglang:load_back_tokens_total{model_name="M",tp_rank="0",pool="swa"} 0
sglang:hicache_backup_tokens_total{model_name="M",tp_rank="0",pool="kv"} 15
sglang:load_back_bytes_total{model_name="M",tp_rank="0"} 4194304
sglang:hicache_backup_bytes_total{model_name="M",tp_rank="0"} 3145728
sglang:hicache_host_used_tokens{model_name="M",tp_rank="0"} 500
sglang:hicache_host_total_tokens{model_name="M",tp_rank="0"} 2000
sglang:cached_tokens_total{model_name="M",tp_rank="0",cache_source="device"} 60
sglang:cached_tokens_total{model_name="M",tp_rank="0",cache_source="host"} 10
sglang:cache_hit_rate{model_name="M",tp_rank="0"} 0.33
sglang:kv_used_tokens{model_name="M",tp_rank="0"} 12000
sglang:kv_available_tokens{model_name="M",tp_rank="0"} 3000
sglang:kv_evictable_tokens{model_name="M",tp_rank="0"} 4000
sglang:max_total_num_tokens{model_name="M",tp_rank="0"} 471258
sglang:page_size{model_name="M",tp_rank="0"} 64
sglang:num_running_reqs{model_name="M",tp_rank="0"} 2
sglang:num_queue_reqs{model_name="M",tp_rank="0"} 5
sglang:num_requests_total{model_name="M",tp_rank="0"} 33
sglang:gen_throughput{model_name="M",tp_rank="0"} 7.5
sglang:time_to_first_token_seconds_bucket{model_name="M",tp_rank="0",le="0.1"} 1
sglang:time_to_first_token_seconds_bucket{model_name="M",tp_rank="0",le="1.0"} 2
sglang:time_to_first_token_seconds_bucket{model_name="M",tp_rank="0",le="+Inf"} 2
sglang:time_to_first_token_seconds_sum{model_name="M",tp_rank="0"} 0.25
sglang:time_to_first_token_seconds_count{model_name="M",tp_rank="0"} 2
"""

# The same scrape after a cpu_hit restore: host_hit +30, load_back kv +45,
# host_used -200 (prefix moved out of host into GPU), device_hit +16384.
PROM_TEXT_AFTER_CPU_HIT = PROM_TEXT_BASE.replace(
    'sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="device_hit"} 50',
    'sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="device_hit"} 16434',
).replace(
    'sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="host_hit"} 10',
    'sglang:prefill_effective_tokens_total{model_name="M",engine_type="null",tp_rank="0",pp_rank="0",moe_ep_rank="0",mode="host_hit"} 40',
).replace(
    'sglang:load_back_tokens_total{model_name="M",tp_rank="0",pool="kv"} 20',
    'sglang:load_back_tokens_total{model_name="M",tp_rank="0",pool="kv"} 65',
).replace(
    'sglang:hicache_host_used_tokens{model_name="M",tp_rank="0"} 500',
    'sglang:hicache_host_used_tokens{model_name="M",tp_rank="0"} 300',
)

# Prometheus scrape that is missing several metrics (to verify the
# missing != 0 rule).
PROM_TEXT_SPARSE = """\
sglang:prefill_effective_tokens_total{model_name="M",tp_rank="0",mode="input"} 5
sglang:prefill_effective_tokens_total{model_name="M",tp_rank="0",mode="device_hit"} 3
sglang:max_total_num_tokens{model_name="M",tp_rank="0"} 1000
"""


def host_hit_meta(prefix_len: int = 16384) -> dict:
    """meta_info for a request fully restored from host."""
    return {
        "cached_tokens": prefix_len,
        "cached_tokens_details": {"device": 0, "host": prefix_len},
    }


def device_hit_meta(prefix_len: int = 16384) -> dict:
    """meta_info for a request fully matched from device."""
    return {
        "cached_tokens": prefix_len,
        "cached_tokens_details": {"device": prefix_len, "host": 0},
    }


def no_hit_meta(prefix_len: int = 16384) -> dict:
    """meta_info for a request with no cache match at all."""
    return {"cached_tokens": 0}


def sample_records(n: int = 10, base_ttft: float = 100.0) -> list[dict]:
    """Deterministic synthetic load-driver records."""
    records = []
    for i in range(n):
        arrival = i * 0.1
        start = arrival + 0.01
        complete = start + base_ttft / 1000.0 + (i % 3) * 0.005
        records.append({
            "request_id": i,
            "t_arrival": arrival,
            "t_start": start,
            "t_first_token": complete,
            "t_complete": complete,
            "queueing_ms": (start - arrival) * 1000,
            "ttft_ms": (complete - arrival) * 1000,
            "service_ms": (complete - start) * 1000,
            "total_ms": (complete - arrival) * 1000,
            "ok": True,
            "error": "",
            "cached_tokens": 16384,
            "cached_tokens_details": {"device": 0, "host": 16384},
        })
    return records
