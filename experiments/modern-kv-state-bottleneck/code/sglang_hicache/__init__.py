"""SGLang / HiCache execution path for Experiments 1-3.

This package drives the public SGLang server through its HTTP and
Prometheus boundaries only.  It never imports SGLang internals and never
copies SGLang code into this repository; it talks to a separately
launched ``python -m sglang.launch_server`` process.

The package is deliberately named ``sglang_hicache`` (not ``sglang``) so
that a child ``python -m sglang.launch_server`` started from ``code/``
resolves the *upstream* SGLang package and never shadows it with this
experiment package.

Backend selection
-----------------
The vLLM path (``runners.vllm_runner`` / ``run_exp1.py`` etc.) remains
the legacy reference implementation.  These modules are the explicit
SGLang execution path:

* ``run_exp1.py`` / ``run_exp2.py`` / ``run_exp3.py`` — entry points
* ``server_lifecycle.py`` — child-process lifecycle for a full Slurm job
* ``http_client.py`` — stdlib HTTP client for the native /generate API
* ``metrics.py`` — typed before/after Prometheus cache-stat snapshots
* ``residency.py`` — recompute / gpu_hit / cpu_hit preparation + evidence
* ``validation.py`` — validation gate shared by Exp1-3
* ``summary.py`` — pure percentile / load summaries (vLLM-compatible keys)
* ``load_driver.py`` — concurrent HTTP load driver for Exp3

All modules are import-safe without CUDA, SGLang, network, or model
weights; the SGLang server is always a separate child process.
"""
