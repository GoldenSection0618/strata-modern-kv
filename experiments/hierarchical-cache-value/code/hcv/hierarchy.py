"""Full-hierarchy capability gate: outcomes, probes, state-group evidence.

Outcomes
--------
``full``                  — every required state group for the model shows
                            restore evidence AND all serial probes pass
                            (recompute / GPU hit / GPU-only eviction
                            negative control / CPU hit) with no
                            restore-to-recompute fallback.
``partial``               — aggregate CPU-tier restore works but per-state-
                            group coverage is not fully observable or only
                            a subset of required groups is validated.
``unsupported``           — the pinned runtime cannot establish or validate
                            the hierarchical path (no host hit after L1
                            eviction, host metrics absent, or probes fail).
``invalid_infrastructure``— environment/infrastructure failure (markers,
                            server start, metrics endpoint, preflight).

Rules encoded here
------------------
* A full-hierarchy claim is never made from aggregate tier counters
  alone: per-state-group restore evidence is required for every state
  group the model needs (Qwen3.5: attention KV + Gated DeltaNet
  recurrent; Gemma 4: local/sliding-window + global attention).
* Serial CPU-hit validation must establish host-hit delta >= prefix
  length and zero device-hit delta; GPU-hit must establish device-hit
  delta >= prefix length and zero host-hit delta.
* A host hit with a simultaneous input-path growth of ~prefix length is
  a silent restore-to-recompute fallback and fails the probe.
* Missing metrics are None/unsupported, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hcv.config import (
    GATE_FULL,
    GATE_INVALID_INFRA,
    GATE_PARTIAL,
    GATE_UNSUPPORTED,
)

#: Known public pool labels -> state group names.
POOL_TO_GROUP = {
    "kv": "attention_kv",
    "attention": "attention_kv",
    "full_attention": "attention_kv",
    "mamba": "gated_delta_recurrent",
    "recurrent": "gated_delta_recurrent",
    "linear": "gated_delta_recurrent",
    "gated_deltanet": "gated_delta_recurrent",
    "swa": "local_sliding_window",
    "local": "local_sliding_window",
    "global": "global_attention",
}


def map_pool_to_group(pool: str) -> str:
    """Map an observed ``pool`` label to a state group name."""
    return POOL_TO_GROUP.get(pool, f"pool:{pool}")


# ---------------------------------------------------------------------------
# Probe records
# ---------------------------------------------------------------------------

PROBE_RECOMPUTE = "recompute"
PROBE_GPU_HIT = "gpu_hit"
PROBE_GPU_ONLY_EVICTION = "gpu_only_eviction_negative_control"
PROBE_CPU_HIT = "cpu_hit"
ALL_PROBES = (PROBE_RECOMPUTE, PROBE_GPU_HIT, PROBE_GPU_ONLY_EVICTION, PROBE_CPU_HIT)


@dataclass
class ProbeResult:
    """Outcome of one serial probe, with isolated before/after deltas."""

    name: str
    prefix_length: int
    ok: bool
    reason: str = ""
    deltas: dict = field(default_factory=dict)
    input_delta: Optional[float] = None
    device_hit_delta: Optional[float] = None
    host_hit_delta: Optional[float] = None
    load_back_delta: Optional[float] = None
    pool_deltas: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prefix_length": self.prefix_length,
            "ok": self.ok,
            "reason": self.reason,
            "deltas": self.deltas,
            "input_delta": self.input_delta,
            "device_hit_delta": self.device_hit_delta,
            "host_hit_delta": self.host_hit_delta,
            "load_back_delta": self.load_back_delta,
            "pool_deltas": self.pool_deltas,
        }


# ---------------------------------------------------------------------------
# State-group evidence
# ---------------------------------------------------------------------------


@dataclass
class StateGroupEvidence:
    """Per-observable-state-group restore evidence."""

    group: str
    observable: bool          # runtime exposed distinguishable evidence
    restore_tokens: Optional[float]      # load_back tokens attributable
    backup_tokens: Optional[float]       # hicache backup tokens attributable
    host_hit_tokens: Optional[float]     # host_hit counter attributable
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "observable": self.observable,
            "restore_tokens": self.restore_tokens,
            "backup_tokens": self.backup_tokens,
            "host_hit_tokens": self.host_hit_tokens,
            "note": self.note,
        }


def build_group_evidence(
    required_groups: list[str],
    cpu_hit_pool_deltas: dict,
    aggregate_host_hit_delta: Optional[float],
    aggregate_load_back_delta: Optional[float],
) -> list[StateGroupEvidence]:
    """Build per-group evidence from the CPU-hit probe's pool deltas.

    ``cpu_hit_pool_deltas`` maps pool label -> load_back tokens delta.
    When the runtime exposes no distinguishable pools, every required
    group is marked not observable and the aggregate numbers are kept
    (used only to label the run partial, never full).
    """
    evidence: list[StateGroupEvidence] = []
    if not cpu_hit_pool_deltas:
        # aggregate-only evidence: per-group coverage unobservable
        for group in required_groups:
            evidence.append(StateGroupEvidence(
                group=group,
                observable=False,
                restore_tokens=aggregate_load_back_delta,
                backup_tokens=None,
                host_hit_tokens=aggregate_host_hit_delta,
                note="aggregate-only; per-group restore not observable from public metrics",
            ))
        return evidence

    # Map observed pools to groups; keep unknown pools as pool:<name>.
    pool_total = sum(v for v in cpu_hit_pool_deltas.values())
    observed_groups: dict[str, float] = {}
    for pool, delta in cpu_hit_pool_deltas.items():
        group = map_pool_to_group(pool)
        # The pinned Gemma runtime labels its full/global-attention pool
        # ``kv``.  Qwen declares the generic attention_kv group instead,
        # while Gemma declares global_attention alongside SWA.
        if (
            pool == "kv"
            and "global_attention" in required_groups
            and "attention_kv" not in required_groups
        ):
            group = "global_attention"
        observed_groups[group] = observed_groups.get(group, 0.0) + delta

    for group in required_groups:
        share = observed_groups.get(group)
        note = "per-pool restore evidence"
        if share is None:
            share = None
            note = "no restore tokens attributable to this group"
        evidence.append(StateGroupEvidence(
            group=group,
            observable=share is not None and share > 0,
            restore_tokens=share,
            backup_tokens=None,
            host_hit_tokens=None,
            note=note,
        ))
    # Keep any extra observable group not in the required list.
    for group, delta in observed_groups.items():
        if group not in required_groups:
            evidence.append(StateGroupEvidence(
                group=group,
                observable=delta > 0,
                restore_tokens=delta,
                backup_tokens=None,
                host_hit_tokens=None,
                note="additional observable state group",
            ))
    return evidence


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Full result of the hierarchy capability gate."""

    status: str
    model: str
    architecture: str
    probe_results: dict = field(default_factory=dict)      # name -> ProbeResult
    state_group_evidence: list = field(default_factory=list)
    concurrent_windows: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    infra_checks: dict = field(default_factory=dict)

    @property
    def reportable(self) -> bool:
        """Only ``full`` (and explicitly gated analyses) are reportable."""
        return self.status == GATE_FULL

    def to_dict(self) -> dict:
        windows = [
            window.to_dict() if hasattr(window, "to_dict") else window
            for window in self.concurrent_windows
        ]
        return {
            "status": self.status,
            "model": self.model,
            "architecture": self.architecture,
            "probe_results": {
                name: p.to_dict() for name, p in self.probe_results.items()
            },
            "state_group_evidence": [e.to_dict() for e in self.state_group_evidence],
            "concurrent_windows": windows,
            "reasons": self.reasons,
            "infra_checks": self.infra_checks,
            "reportable": self.reportable,
        }


# ---------------------------------------------------------------------------
# Pure classification
# ---------------------------------------------------------------------------

_ZERO_OK = 0.0


def _delta_ok(value: Optional[float], minimum: float) -> bool:
    """True when value is observable and >= minimum."""
    return value is not None and value >= minimum


def _zero(value: Optional[float]) -> bool:
    """True when value is observable and exactly 0."""
    return value is not None and value == _ZERO_OK


def classify_gate(
    model: str,
    architecture: str,
    required_groups: list[str],
    probes: dict[str, ProbeResult],
    aggregate_host_hit_delta: Optional[float],
    aggregate_load_back_delta: Optional[float],
    infra_checks: Optional[dict] = None,
    concurrent_windows: Optional[list] = None,
) -> GateResult:
    """Classify the gate outcome from serial probe results (pure).

    Infrastructure failures are handled by the runner before this call
    (``invalid_infrastructure``); this function focuses on the
    mechanism-level classification.
    """
    reasons: list[str] = []
    infra = infra_checks or {}
    if not infra.get("env_ok", True):
        return GateResult(
            status=GATE_INVALID_INFRA, model=model, architecture=architecture,
            probe_results=probes, reasons=[str(infra.get("env_error", "env check failed"))],
            infra_checks=infra,
        )
    if not infra.get("server_ok", True):
        return GateResult(
            status=GATE_INVALID_INFRA, model=model, architecture=architecture,
            probe_results=probes, reasons=[str(infra.get("server_error", "server failed"))],
            infra_checks=infra,
        )
    if not infra.get("metrics_ok", True):
        return GateResult(
            status=GATE_INVALID_INFRA, model=model, architecture=architecture,
            probe_results=probes,
            reasons=["/metrics endpoint unusable; public tier counters unavailable"],
            infra_checks=infra,
        )

    # --- probe-by-probe checks ------------------------------------------
    checks: dict[str, str] = {}

    recompute = probes.get(PROBE_RECOMPUTE)
    if recompute is not None and recompute.ok:
        checks[PROBE_RECOMPUTE] = "ok"
    elif recompute is not None:
        reasons.append(f"recompute probe failed: {recompute.reason}")
        checks[PROBE_RECOMPUTE] = "failed"
    else:
        reasons.append("recompute probe missing")
        checks[PROBE_RECOMPUTE] = "missing"

    gpu_hit = probes.get(PROBE_GPU_HIT)
    if gpu_hit is not None and gpu_hit.ok:
        checks[PROBE_GPU_HIT] = "ok"
    elif gpu_hit is not None:
        reasons.append(f"gpu_hit probe failed: {gpu_hit.reason}")
        checks[PROBE_GPU_HIT] = "failed"
    else:
        reasons.append("gpu_hit probe missing")
        checks[PROBE_GPU_HIT] = "missing"

    neg = probes.get(PROBE_GPU_ONLY_EVICTION)
    if neg is not None and neg.ok:
        checks[PROBE_GPU_ONLY_EVICTION] = "ok"
    elif neg is not None:
        reasons.append(f"gpu_only_eviction negative control failed: {neg.reason}")
        checks[PROBE_GPU_ONLY_EVICTION] = "failed"
    else:
        reasons.append("gpu_only_eviction probe missing")
        checks[PROBE_GPU_ONLY_EVICTION] = "missing"

    cpu_hit = probes.get(PROBE_CPU_HIT)
    if cpu_hit is not None and cpu_hit.ok:
        checks[PROBE_CPU_HIT] = "ok"
    elif cpu_hit is not None:
        reasons.append(f"cpu_hit probe failed: {cpu_hit.reason}")
        checks[PROBE_CPU_HIT] = "failed"
    else:
        reasons.append("cpu_hit probe missing")
        checks[PROBE_CPU_HIT] = "missing"

    # --- classification ---------------------------------------------------
    if checks.get(PROBE_CPU_HIT) != "ok":
        if checks.get(PROBE_CPU_HIT) == "missing":
            status = GATE_UNSUPPORTED
            reasons.append("cpu_hit probe did not run; hierarchical path not exercised")
        elif aggregate_host_hit_delta is None or aggregate_host_hit_delta <= 0:
            # No host-hit evidence at all: the CPU tier never served a hit.
            status = GATE_UNSUPPORTED
        else:
            # Some host-hit evidence exists but the serial CPU-hit probe
            # did not pass (e.g. insufficient restore or a fallback).
            status = GATE_PARTIAL
        if not reasons:
            reasons.append("CPU-tier restore could not be validated")
        group_evidence = build_group_evidence(
            required_groups, {}, aggregate_host_hit_delta, aggregate_load_back_delta
        )
        return GateResult(
            status=status, model=model, architecture=architecture,
            probe_results=probes, state_group_evidence=group_evidence,
            concurrent_windows=concurrent_windows or [],
            reasons=reasons, infra_checks=infra,
        )

    # CPU hit works at aggregate level; decide full vs partial by
    # per-state-group coverage AND all four probes passing.
    group_evidence = build_group_evidence(
        required_groups,
        cpu_hit.pool_deltas if cpu_hit else {},
        aggregate_host_hit_delta,
        aggregate_load_back_delta,
    )
    all_observable = all(e.observable for e in group_evidence if e.group in required_groups)
    all_probes_ok = all(
        checks.get(p) == "ok"
        for p in (PROBE_RECOMPUTE, PROBE_GPU_HIT, PROBE_GPU_ONLY_EVICTION, PROBE_CPU_HIT)
    )
    if all_observable and all_probes_ok:
        status = GATE_FULL
    else:
        status = GATE_PARTIAL
        if not all_observable:
            reasons.append(
                "aggregate CPU-tier restore verified but per-state-group "
                "restore coverage is not fully observable"
            )
        elif not all_probes_ok:
            reasons.append(
                "CPU-tier restore verified but not all gate probes passed "
                "(see probe_results)"
            )
    return GateResult(
        status=status, model=model, architecture=architecture,
        probe_results=probes, state_group_evidence=group_evidence,
        concurrent_windows=concurrent_windows or [],
        reasons=reasons, infra_checks=infra,
    )
