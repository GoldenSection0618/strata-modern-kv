"""Bounded SGLang server lifecycle for one Slurm job.

Launches ``python -m sglang.launch_server`` as a child process, waits
for readiness on ``/health`` with a bounded timeout, keeps stdout/stderr
files, and shuts the server down gracefully on normal completion.

Hang policy: a process that does not become ready, or does not exit
after SIGTERM, is left for the Slurm job to clean up — the cluster
convention is that a stuck job must be killed with ``scancel`` at the
job level (Slurm cgroup cleanup removes the whole process tree).  This
module therefore never blocks forever and never force-kills after a
bounded grace period without logging the exact state.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_READY_TIMEOUT_S = 1800.0   # model load on A100 can take several minutes
READY_POLL_INTERVAL_S = 2.0
TERM_GRACE_S = 30.0

#: Upstream SGLang launch module.  This must stay exactly
#: ``sglang.launch_server`` (the installed upstream package); the local
#: experiment package is ``hcv`` precisely so that
#: ``python -m sglang.launch_server`` from ``code/`` resolves upstream
#: SGLang and is never shadowed by a local ``sglang`` package.
LAUNCH_MODULE = "sglang.launch_server"


def pick_free_port() -> int:
    """Ask the OS for a free ephemeral port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerLaunchSpec:
    """Everything needed to launch one SGLang server process."""

    argv: list[str]
    stdout_path: str
    stderr_path: str
    cwd: str | None = None
    env: dict = field(default_factory=dict)
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S


class SGLangServerProcess:
    """Manage one SGLang server child process."""

    def __init__(
        self,
        argv: list[str],
        stdout_path: str,
        stderr_path: str,
        cwd: str | None = None,
        env: dict | None = None,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        python_executable: str | None = None,
    ):
        self.argv = list(argv)
        self.stdout_path = str(stdout_path)
        self.stderr_path = str(stderr_path)
        self.cwd = cwd
        self.env = env or {}
        self.ready_timeout_s = ready_timeout_s
        self.python_executable = python_executable or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self.ready = False
        self._started_at: float = 0.0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Launch the server child process with logs to files."""
        stdout_dir = Path(self.stdout_path).parent
        stderr_dir = Path(self.stderr_path).parent
        stdout_dir.mkdir(parents=True, exist_ok=True)
        stderr_dir.mkdir(parents=True, exist_ok=True)

        argv = [self.python_executable, "-m", LAUNCH_MODULE, *self.argv]

        env = os.environ.copy()
        env.update(self.env)

        logger.info("Launching SGLang server: %s", " ".join(argv[:6]) + (" ..." if len(argv) > 6 else ""))
        logger.info("  stdout -> %s", self.stdout_path)
        logger.info("  stderr -> %s", self.stderr_path)

        with open(self.stdout_path, "ab") as out, open(self.stderr_path, "ab") as err:
            self.proc = subprocess.Popen(
                argv,
                stdout=out,
                stderr=err,
                cwd=self.cwd,
                env=env,
                start_new_session=True,
            )
        self._started_at = time.monotonic()
        logger.info("SGLang server pid=%d", self.proc.pid)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, health_check) -> bool:
        """Poll readiness until timeout.  ``health_check`` is a callable
        returning bool (e.g. ``client.health``)."""
        deadline = time.monotonic() + self.ready_timeout_s
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            if not self.is_alive():
                tail = self._tail_stderr()
                raise RuntimeError(
                    "SGLang server exited before becoming ready "
                    f"(rc={self.proc.poll()}). stderr tail:\n{tail}"
                )
            try:
                if health_check():
                    self.ready = True
                    logger.info("SGLang server ready (pid=%d)", self.proc.pid)
                    return True
            except Exception as e:  # noqa: BLE001 - readiness probing
                last_error = str(e)
            time.sleep(READY_POLL_INTERVAL_S)

        tail = self._tail_stderr()
        raise TimeoutError(
            f"SGLang server not ready within {self.ready_timeout_s:.0f}s "
            f"(last health error: {last_error}). stderr tail:\n{tail}"
        )

    def stop(self, term_grace_s: float = TERM_GRACE_S) -> int:
        """Best-effort graceful shutdown: SIGTERM, bounded wait, report.

        If the process does not exit within ``term_grace_s``, logs the
        state and returns — the Slurm job must be ``scancel``-ed so the
        cgroup cleanup removes the whole tree.
        """
        if self.proc is None:
            return -1
        if not self.is_alive():
            rc = self.proc.poll()
            logger.info("SGLang server already exited (rc=%s)", rc)
            self.proc = None
            return int(rc or 0)

        logger.info("Sending SIGTERM to SGLang server pid=%d", self.proc.pid)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as e:
            logger.warning("Cannot signal SGLang server: %s", e)

        try:
            rc = self.proc.wait(timeout=term_grace_s)
        except subprocess.TimeoutExpired:
            logger.error(
                "SGLang server pid=%d did not exit within %.0fs after SIGTERM. "
                "Job must be cleaned up with scancel (Slurm cgroup cleanup).",
                self.proc.pid, term_grace_s,
            )
            self.ready = False
            return -2
        logger.info("SGLang server exited cleanly (rc=%s)", rc)
        self.proc = None
        self.ready = False
        return int(rc or 0)

    def _tail_stderr(self, n_lines: int = 40) -> str:
        try:
            lines = Path(self.stderr_path).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return "(stderr unavailable)"
        return "\n".join(lines[-n_lines:])

    # -- helpers ------------------------------------------------------------

    def log_files(self) -> dict:
        return {
            "stdout": self.stdout_path,
            "stderr": self.stderr_path,
        }
