# Current Status

Last updated: 2026-08-12 (A100 smoke completion)

## Scope of the completed work

The implementation, dry-run checks, and one-repeat A100 smoke measurements for Experiments 1--4 are complete. These runs validate the execution path, configuration semantics, full-hierarchy capability gate, and result-processing pipeline. They are **not** the formal repeated measurements from which performance conclusions or confidence intervals should be reported.

All GPU work used `smtg5002` and the pinned user-directory environment:

```text
/share01/hpc/humxlab_intern/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211
SGLang 0.5.6.post3.dev8468+g4ad990ba7
commit 4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63
PyTorch 2.11.0+cu129
```

The runtime baseline is `direct` I/O, `page_first_direct` host layout, `write_through` policy, page size 64, and `hicache-ratio=3`. Docker, system CUDA/driver changes, and login-node builds were not used.

## Capability and smoke evidence

| Item | Status | Evidence / interpretation |
| --- | --- | --- |
| Qwen3.5-9B full hierarchy gate | complete | Required attention-KV and Gated DeltaNet state groups restored correctly; formal ceiling-1 and ceiling-4 windows supplied the admission-tier and H-to-D load-back evidence. |
| Experiment 1 | complete smoke | Four GPU-only/hierarchical and cold/warm cells completed (jobs `1294852`--`1294855`). |
| Experiment 2 | complete smoke | Six Low/Medium/High by GPU-only/hierarchical cells completed (jobs `1295128`--`1295133`). The calibrated Qwen budget fractions are Low `0.85`, Medium `0.65`, and High `0.60`; the protected-prefix admission floor is `0.55`. |
| Experiment 3 | complete smoke | Eight configured reuse-level cells completed with fixed high-pressure preparation. Reuse is evaluated over eligible request slots, while the reported realized fraction remains normalized by all slots. |
| Experiment 4 (Gemma) | complete smoke | Final full-gate run `1295215` completed. V0 uses budget `0.85`; V1/V2 use `0.75` plus capacity-aware filler. The hierarchy cells for V1/V2 recorded CPU hits and H-to-D restore; V0 is the intended low-pressure control and did not require a CPU hit. |
| Processing and unit tests | complete | The final analysis job `1295222` completed; the current suite passed `92/92` tests before the final smoke submission. |

The earlier Exp4 run `1295190` is superseded: it exposed that the Qwen budget fractions were below Gemma's SWA admission floor. It must not be used as a negative result. The final Exp4 configuration uses Gemma-specific matched budgets and a model-aware state-group mapping (`kv` maps to Gemma `global_attention`, rather than Qwen `attention_kv`).

## Measurement semantics retained for formal runs

- A direct serial probe can lose a particular host copy after its resident GPU copy is reused or retired. For the hierarchical Experiment 2 windows, the authoritative evidence is therefore the concurrent formal window's CPU-hit and H-to-D load-back counters, with the serial probe retained as a diagnostic rather than a hard invalidation rule.
- GPU-only and hierarchical cells keep the same model, GPU cache budget, serving settings, workload, and pressure construction. CPU hierarchy is the paired difference.
- Exp2's Low point is deliberately not given artificial filler pressure. Medium and High use calibrated/fixed filler only to establish the intended capacity regime.
- Gemma has distinct global-attention KV and sliding-window-attention state groups. Its full gate requires the actual Gemma group names, not Qwen naming assumptions.

## Next step: formal measurements

The next execution phase is repeated formal measurement, not further environment debugging:

1. Freeze the smoke-validated code and configurations from this commit.
2. Run the prescribed repeated Exp1--Exp3 paired cells, keeping the same capability gate and fixed-pressure preparation.
3. Run the selected Gemma Exp4 representative configurations with its matched-budget policy, rather than transplanting Qwen fractions.
4. Process only completed, valid repeated cells into formal summary tables and confidence intervals; retain the smoke artifacts as execution-path evidence.

Result artifacts remain under `results/` and are intentionally not versioned. See [`README.md`](../README.md) for scope and [`code/README.md`](../code/README.md) for invocation and validation details.
