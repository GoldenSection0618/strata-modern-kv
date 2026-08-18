"""Tests for hcv.metrics: parser, missing->None rule, deltas, pool labels."""

from __future__ import annotations

from hcv.metrics import (
    CacheStats,
    diff_snapshots,
    json_dumps_compact,
    metric_value,
    parse_prometheus_text,
    select_values,
    snapshot_from_scrape,
)

SAMPLE = """# HELP sglang:prefill_effective_tokens_total Effective tokens
# TYPE sglang:prefill_effective_tokens_total counter
sglang:prefill_effective_tokens_total{mode="input",model_name="qwen"} 1000
sglang:prefill_effective_tokens_total{mode="device_hit",model_name="qwen"} 200
sglang:prefill_effective_tokens_total{mode="host_hit",model_name="qwen"} 300
sglang:prefill_effective_tokens_total{mode="storage_hit",model_name="qwen"} 0
sglang:load_back_tokens_total{pool="kv",model_name="qwen"} 120
sglang:load_back_tokens_total{pool="mamba",model_name="qwen"} 80
sglang:hicache_backup_tokens_total{pool="kv",model_name="qwen"} 150
sglang:hicache_backup_tokens_total{pool="mamba",model_name="qwen"} 60
sglang:hicache_host_used_tokens 900
sglang:hicache_host_total_tokens 10000
sglang:kv_used_tokens 4000
sglang:kv_available_tokens 6000
sglang:kv_evictable_tokens 100
sglang:max_total_num_tokens 10000
sglang:time_to_first_token_seconds_bucket{le="0.1"} 1
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 2
sglang:time_to_first_token_seconds_sum 0.25
sglang:time_to_first_token_seconds_count 2
sglang:num_running_reqs 3
sglang:num_queue_reqs 0
sglang:num_requests_total 42
"""


def test_parse_basic():
    scrape = parse_prometheus_text(SAMPLE)
    assert scrape.has("sglang:prefill_effective_tokens_total")
    assert metric_value(scrape, "sglang:prefill_effective_tokens_total", mode="input") == 1000
    assert metric_value(scrape, "sglang:prefill_effective_tokens_total", mode="host_hit") == 300


def test_missing_metric_is_none_not_zero():
    scrape = parse_prometheus_text(SAMPLE)
    # a metric that does not exist at all
    assert metric_value(scrape, "sglang:does_not_exist") is None
    # a label value that does not exist for a present family
    assert metric_value(scrape, "sglang:prefill_effective_tokens_total", mode="nope") is None


def test_zero_is_zero_only_when_present():
    scrape = parse_prometheus_text(SAMPLE)
    # storage_hit is present with value 0
    assert metric_value(scrape, "sglang:prefill_effective_tokens_total", mode="storage_hit") == 0.0
    stats = snapshot_from_scrape(scrape)
    assert stats.prefill_storage_hit_tokens == 0.0
    # absent fields stay None
    assert stats.cache_hit_rate is None


def test_histogram_folding():
    scrape = parse_prometheus_text(SAMPLE)
    assert scrape.histograms["sglang:time_to_first_token_seconds"].sum == 0.25
    assert scrape.histograms["sglang:time_to_first_token_seconds"].count == 2


def test_label_escaping():
    text = 'sglang:cache_hit_rate{model_name="a\\"b"} 0.5\n'
    scrape = parse_prometheus_text(text)
    assert metric_value(scrape, "sglang:cache_hit_rate") == 0.5


def test_select_values_multiple():
    scrape = parse_prometheus_text(SAMPLE)
    vals = select_values(scrape, "sglang:load_back_tokens_total")
    assert sorted(vals) == [80.0, 120.0]


def test_snapshot_pool_breakdown():
    scrape = parse_prometheus_text(SAMPLE)
    stats = snapshot_from_scrape(scrape)
    assert stats.load_back_tokens_by_pool == {"kv": 120.0, "mamba": 80.0}
    assert stats.hicache_backup_tokens_by_pool == {"kv": 150.0, "mamba": 60.0}


def test_delta_none_propagation():
    a = CacheStats(timestamp=1.0, prefill_input_tokens=100.0,
                   prefill_host_hit_tokens=50.0)
    b = CacheStats(timestamp=0.0, prefill_input_tokens=90.0,
                   prefill_host_hit_tokens=None)  # missing on one side
    d = diff_snapshots(a, b)
    assert d.get("prefill_input_tokens") == 10.0
    # host_hit missing on one side -> delta None, never a silent zero
    assert d.get("prefill_host_hit_tokens") is None


def test_delta_zero_when_both_present_and_equal():
    a = CacheStats(timestamp=1.0, prefill_input_tokens=100.0)
    b = CacheStats(timestamp=0.0, prefill_input_tokens=100.0)
    d = diff_snapshots(a, b)
    assert d.get("prefill_input_tokens") == 0.0


def test_pool_deltas():
    a = CacheStats(timestamp=1.0,
                   load_back_tokens_by_pool={"kv": 200.0, "mamba": 90.0})
    b = CacheStats(timestamp=0.0,
                   load_back_tokens_by_pool={"kv": 120.0})
    d = diff_snapshots(a, b)
    assert d.pool_deltas["load_back_tokens_total"] == {"kv": 80.0, "mamba": 90.0}


def test_tier_hits_total():
    scrape = parse_prometheus_text(SAMPLE)
    stats = snapshot_from_scrape(scrape)
    assert stats.tier_hits_total() == 200.0 + 300.0 + 0.0


def test_json_compact_deterministic():
    assert json_dumps_compact({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'
    assert json_dumps_compact({"a": 1, "b": 2}) == json_dumps_compact({"b": 2, "a": 1})
