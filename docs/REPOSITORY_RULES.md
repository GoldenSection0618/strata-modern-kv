# Repository Rules

Cluster execution is additionally governed by the canonical rules in
`/share01/hpc/humxlab_intern/yanglihan/dl-stack/docs/01-集群使用规则.md` and
environment guidance in `02-环境配置.md`. Those global rules take precedence
for login-node resource limits, Slurm submission, environment installation,
agent delegation, GPU selection, and cleanup safety; this repository records
only project-specific additions.

## 1. Purpose

This repository is an experimental research codebase. The main requirements are:

- preserve a traceable experimental history;
- make reported results reproducible;
- keep generated artifacts and large model files out of Git history;
- separate implementation changes from experimental conclusions;
- pin volatile model/runtime behavior whenever it affects a reported result.

## 2. Protected history

### `main`

`main` is the canonical project history.

Rules:

- force pushes to `main` are prohibited;
- published commits must not be rewritten or removed to make history look cleaner;
- use a new commit to correct an earlier mistake;
- do not delete or recreate `main` as part of normal development.

If a local branch diverges from `main`, resolve it by rebasing or merging locally and then performing a normal push. Do not bypass repository protection by changing GitHub rules for convenience.

## 3. Branches

Use short-lived branches for changes that are experimental, invasive, or not ready to become part of the canonical implementation.

Recommended naming:

```text
exp/<topic>
feat/<topic>
fix/<topic>
docs/<topic>
```

Examples:

```text
exp/page-size-sweep
feat/hierarchical-cache
fix/io-accounting
docs/experiment-plan
```

Force pushing a personal development branch is allowed only when necessary. Prefer `--force-with-lease` over `--force`. Never force push a shared branch.

## 4. Commits

A commit should represent one coherent change.

Recommended prefixes:

```text
feat:  new system functionality
exp:   experiment or benchmark changes
fix:   bug fix
docs:  documentation
refactor: internal restructuring
chore: repository maintenance
```

Examples:

```text
exp: add long-context reuse workload
fix: correct CPU-GPU traffic accounting
docs: document scheduler ablation plan
```

Do not rewrite published commits merely to obtain prettier commit history.

## 5. Experimental reproducibility

Every experiment that contributes to a reported result should make it possible to recover:

- exact model identifier and model revision;
- hardware platform;
- GPU driver, CUDA/runtime, and serving-engine version or commit;
- model precision or quantization mode;
- workload definition and exact token lengths;
- relevant runtime configuration and feature flags;
- cache-residency mode and cache/state policy;
- random seed when applicable;
- raw measurement source;
- processing or plotting procedure.

A plotted number should be traceable back to raw measurements rather than being copied manually into plotting code.

Features whose behavior is explicitly experimental or rapidly changing in the serving runtime must be validated on the pinned version before their measurements are interpreted as model behavior.

## 6. Experimental integrity

Do not silently discard runs because they disagree with the expected conclusion.

When results are unstable or anomalous:

1. preserve the raw result;
2. identify whether the run is invalid for a documented reason;
3. record that reason in the run metadata;
4. rerun when appropriate;
5. report meaningful variance or instability.

Negative results are part of the project evidence when they materially affect a claim.

A configuration change made after inspecting results must be recorded. Do not silently tune thresholds, cache policies, or load points separately for different models in a way that changes the comparison question.

## 7. Generated and large files

Do not commit:

- model weights;
- Hugging Face caches;
- Python virtual environments;
- compiled binaries and build directories;
- raw profiler dumps;
- large benchmark logs;
- generated figures that can be reproduced automatically, unless a specific result snapshot needs to be preserved;
- datasets that have an external canonical source or redistribution restriction.

Store paths, download instructions, metadata, checksums, and small manifests instead.

## 8. Results organization

Keep the distinction between three levels of data:

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

Processing scripts should not overwrite raw measurements.

When a result is used in a report or paper, preserve enough metadata to identify the exact experiment configuration and raw runs that produced it.

## 9. Measurement semantics

Do not replace measured runtime state footprint with a theoretical formula when the serving engine uses hybrid cache groups, allocator padding, checkpointing, or other implementation-specific layouts.

Do not add raw transfer duration to computation time when asynchronous transfer overlaps with compute. A latency decomposition must distinguish transfer activity from non-overlapped I/O stall.

Cross-model normalized metrics must retain the underlying absolute measurements so normalization cannot hide large differences in real workload or capacity.

## 10. Configuration over hidden state

Experimental parameters should live in version-controlled configuration or command-line arguments whenever practical.

Avoid relying on:

- manually edited constants that are not recorded;
- shell history as the only record of an experiment;
- undocumented environment variables;
- machine-local paths embedded in source code;
- implicit cache contents left by a previous run.

Machine-specific paths and credentials must stay outside Git.

## 11. Secrets and credentials

Never commit:

- API keys;
- access tokens;
- SSH private keys;
- cluster credentials;
- private dataset credentials;
- `.env` files containing secrets.

If a secret is committed accidentally, removing it in a later commit is not sufficient. Treat it as compromised and rotate it.

## 12. Documentation changes

When an experiment design materially changes, update the corresponding documentation in the same development cycle.

Keep `docs/EXPERIMENT_PLAN.md`, `docs/TECHNICAL_BASELINE.md`, and experiment-specific documentation synchronized with:

- the questions being tested;
- major experiment groups;
- comparison baselines;
- measurement semantics;
- interpretation boundaries;
- volatile runtime assumptions.

The documentation describes the intended evidence. The code and raw measurements determine what was actually executed.
