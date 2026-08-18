"""Hierarchical Cache Value experiment group.

Local experiment package for ``experiments/hierarchical-cache-value``.

The package name ``hcv`` is deliberately distinct from upstream
``sglang`` so that ``python -m sglang.launch_server`` (the installed
upstream module) can never be shadowed by this repository's code.  All
interaction with the serving runtime goes through the public SGLang
HTTP surface (``/health``, ``/generate``, ``/flush_cache``,
``/metrics``, ``/get_model_info``); no SGLang internals are imported or
copied.

The frozen hierarchy-only baseline for this group is ``direct`` I/O,
``page_first_direct`` host layout, ``write_through`` policy, page size
64, public metrics, and the canonical conda prefix
``sglang-hicache-cu129-torch211`` (installed commit
``4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63``).
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Installed SGLang commit pinned for every run in this group.
PINNED_SGLANG_COMMIT = "4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63"
#: Installed SGLang version string (from provenance.json).
PINNED_SGLANG_VERSION = "0.5.6.post3.dev8468+g4ad990ba7"
