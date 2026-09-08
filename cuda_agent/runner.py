"""Subprocess harness: build, accuracy and bench against a real checkout.

The commands are env-configurable rather than hard-coded. sparkinfer's scored
target moves (Qwen3.8, DSpark, and next spark2_5), and each target has its own
bench entry point; wiring one in should be a config change, not a code change.

Everything runs with a timeout and captured output. A hung compile must cost the
round one iteration, not the whole budget.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from cuda_agent.harness import AccuracyResult, BenchResult, BuildResult

DEFAULT_BUILD = "cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j"
DEFAULT_BENCH = "bench/scripts/bench.sh --tokens 128"
DEFAULT_ACCURACY = "bench/scripts/accuracy.sh"

# sparkinfer's bench emits a structured summary line, which is what we prefer:
#   SWEEP_JSON {"0":{"decode_tps":95.7000,"prefill_pp":6942.0000},"4096":{...}}
# and, per context, a human line where the throughput comes BEFORE the context:
#   decode tg    : 95.70 tok/s  (n=128, ctx=4096, bs=1)
_SWEEP_JSON_RE = re.compile(r"^SWEEP_JSON\s+(\{.*\})\s*$", re.MULTILINE)
DEFAULT_TPS_RE = r"decode tg[^:\n]*:\s*(?P<tps>\d+\.?\d*)\s*tok/s[^\n]*?ctx=(?P<ctx>\d+)"

# accuracy_compare.py prints a machine line:
#   <LABEL> top1=0.960000 kl=0.012000 ppl_spark=... ppl_llama=...
DEFAULT_TOP1_RE = r"top1[=\s:]+(?P<v>\d+\.?\d*)"
DEFAULT_KL_RE = r"\bkl[=\s:]+(?P<v>\d+\.?\d*)"

TOP1_BAR = 0.90
KL_BAR = 0.20


def _ctx_label(raw: str) -> str:
    n = int(raw)
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


@dataclass
class SubprocessHarness:
    repo: Path
    build_cmd: str = ""
    bench_cmd: str = ""
    accuracy_cmd: str = ""
    build_timeout_s: float = 2400.0
    bench_timeout_s: float = 1800.0
    quick_bench_cmd: str = ""
    env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.build_cmd = self.build_cmd or os.environ.get("CUDA_AGENT_BUILD_CMD", DEFAULT_BUILD)
        self.bench_cmd = self.bench_cmd or os.environ.get("CUDA_AGENT_BENCH_CMD", DEFAULT_BENCH)
        self.accuracy_cmd = self.accuracy_cmd or os.environ.get("CUDA_AGENT_ACCURACY_CMD", DEFAULT_ACCURACY)
        # A cheap single-context sweep for the inner loop. The full gated sweep
        # is slow enough that running it every iteration would leave a 3-hour
        # round with a handful of tries.
        self.quick_bench_cmd = self.quick_bench_cmd or os.environ.get(
            "CUDA_AGENT_QUICK_BENCH_CMD", self.bench_cmd
        )

    # -- gates ----------------------------------------------------------------

    def build(self, path: Path) -> BuildResult:
        t0 = time.monotonic()
        proc = self._run(self.build_cmd, path, self.build_timeout_s)
        return BuildResult(
            ok=proc.returncode == 0,
            log=(proc.stdout or "") + (proc.stderr or ""),
            elapsed_s=time.monotonic() - t0,
        )

    def accuracy(self, path: Path) -> AccuracyResult:
        proc = self._run(self.accuracy_cmd, path, self.bench_timeout_s)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            return AccuracyResult(ok=False, detail=_tail(out))

        top1 = _search_float(os.environ.get("CUDA_AGENT_TOP1_RE", DEFAULT_TOP1_RE), out)
        kl = _search_float(os.environ.get("CUDA_AGENT_KL_RE", DEFAULT_KL_RE), out)
        if top1 is None or kl is None:
            # An unparseable accuracy run is a failure, never a pass. Treating
            # "no numbers found" as OK is how a broken gate silently stops
            # gating.
            return AccuracyResult(ok=False, detail=f"could not parse accuracy output: {_tail(out, 6)}")

        ok = top1 >= TOP1_BAR and kl <= KL_BAR
        return AccuracyResult(
            ok=ok, top1=top1, kl=kl,
            detail="" if ok else f"top1={top1:.3f} kl={kl:.3f} outside bars",
        )

    def bench(self, path: Path, *, quick: bool = False) -> BenchResult:
        cmd = self.quick_bench_cmd if quick else self.bench_cmd
        proc = self._run(cmd, path, self.bench_timeout_s)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            return BenchResult(ok=False, detail=_tail(out))

        tps = parse_tps(out)
        if not tps:
            return BenchResult(ok=False, detail=f"no tok/s found in output: {_tail(out, 6)}")
        return BenchResult(ok=True, tps=tps)

    # -- process --------------------------------------------------------------

    def _run(self, cmd: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess:
        env = {**os.environ, **(self.env or {})}
        try:
            return subprocess.run(
                cmd, shell=True, cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                shlex.split(cmd), returncode=124,
                stdout=(exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"timed out after {timeout:.0f}s",
            )


def parse_tps(out: str) -> dict[str, float]:
    """Extract {context label: decode tok/s} from bench output.

    Prefers the structured SWEEP_JSON line over scraping the human-readable
    ones: it is emitted by the bench itself, carries full precision, and cannot
    drift when someone reformats a printf.
    """
    sweep = _parse_sweep_json(out)
    if sweep:
        return sweep

    pattern = os.environ.get("CUDA_AGENT_TPS_RE", DEFAULT_TPS_RE)
    tps: dict[str, float] = {}
    # Not DOTALL: each measurement is one line, and letting `.` cross newlines
    # would pair one context's throughput with another context's label.
    for m in re.finditer(pattern, out, re.IGNORECASE):
        try:
            tps[_ctx_label(m.group("ctx"))] = float(m.group("tps"))
        except (IndexError, ValueError):
            continue
    return tps


def _parse_sweep_json(out: str) -> dict[str, float]:
    """Decode throughput from the bench's own JSON summary, if it emitted one."""
    matches = _SWEEP_JSON_RE.findall(out or "")
    if not matches:
        return {}
    try:
        rows = json.loads(matches[-1])          # last line wins: it is the final sweep
    except json.JSONDecodeError:
        return {}

    tps: dict[str, float] = {}
    for ctx, row in rows.items():
        if not isinstance(row, dict):
            continue
        value = row.get("decode_tps")
        if isinstance(value, (int, float)) and value > 0:
            tps[_ctx_label(str(ctx))] = float(value)
    return tps


def _search_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group("v"))
    except (IndexError, ValueError):
        return None


def _tail(log: str, lines: int = 20) -> str:
    rows = [r for r in (log or "").splitlines() if r.strip()]
    return "\n".join(rows[-lines:])
