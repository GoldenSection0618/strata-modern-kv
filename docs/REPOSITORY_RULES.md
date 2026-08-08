# Repository Rules

## 1. Purpose

This repository is an experimental research codebase. The main requirements are:

- preserve a traceable experimental history;
- make reported results reproducible;
- keep generated artifacts and large model files out of Git history;
- separate implementation changes from experimental conclusions.

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

- model and model revision;
- hardware platform;
- software and dependency versions;
- workload definition;
- relevant runtime configuration;
- random seed when applicable;
- raw measurement source;
- processing or plotting procedure.

A plotted number should be traceable back to raw measurements rather than being copied manually into plotting code.

## 6. Experimental integrity

Do not silently discard runs because they disagree with the expected conclusion.

When results are unstable or anomalous:

1. preserve the raw result;
2. identify whether the run is invalid for a documented reason;
3. rerun when appropriate;
4. report meaningful variance or instability.

Negative results are part of the project evidence when they materially affect a claim.

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

Store paths, download instructions, metadata, and small manifests instead.

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

When a result is used in a report or paper, preserve enough metadata to identify the exact experiment configuration that produced it.

## 9. Configuration over hidden state

Experimental parameters should live in version-controlled configuration or command-line arguments whenever practical.

Avoid relying on:

- manually edited constants that are not recorded;
- shell history as the only record of an experiment;
- undocumented environment variables;
- machine-local paths embedded in source code.

Machine-specific paths and credentials must stay outside Git.

## 10. Secrets and credentials

Never commit:

- API keys;
- access tokens;
- SSH private keys;
- cluster credentials;
- private dataset credentials;
- `.env` files containing secrets.

If a secret is committed accidentally, removing it in a later commit is not sufficient. Treat it as compromised and rotate it.

## 11. Documentation changes

When an experiment design materially changes, update the corresponding documentation in the same development cycle.

In particular, keep `docs/EXPERIMENT_PLAN.md` synchronized with:

- the questions being tested;
- major experiment groups;
- comparison baselines;
- interpretation boundaries.

The documentation describes the intended evidence. The code and raw measurements determine what was actually executed.
