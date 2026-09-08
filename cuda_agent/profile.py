"""Nsight Compute profiling, and target selection driven by it.

Without this the agent walks kernels round-robin: with 43 kernel files and a
3B-active proposer, most of the round is spent looking at code that was never on
the critical path. With it, the agent starts at the kernel that owns the time and
is told *why* it is slow, which is the one piece of context that turns a generic
"make this faster" into a specific technique.

Profiling is expensive (ncu replays each launch), so it runs once at the start of
a round and again only after the frontier has moved enough to invalidate it.
When ncu is missing or profiling permission is denied -- common on shared boxes,
where it needs CAP_SYS_ADMIN or NVreg_RestrictProfilingToAdminUsers=0 -- the
agent degrades to round-robin rather than failing the round.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cuda_agent.source import Target, list_symbols

# Deliberately small. Every metric costs replay passes, and this set is enough to
# separate the four cases that lead to different techniques: memory-bound,
# compute-bound, occupancy-limited, and latency/dependency-stalled.
METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "l1tex__t_sector_hit_rate.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
)

_STALL_RE = re.compile(r"warps_issue_stalled_(?P<reason>\w+?)_per_issue_active")

# What each stall reason usually means, and which families are worth trying. The
# mapping is a prior, not a rule -- the bandit still has to earn its beliefs from
# measured outcomes.
STALL_GUIDANCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "long_scoreboard": (
        "waiting on global/local memory loads",
        ("vector_width", "memory_hint", "smem_layout"),
    ),
    "short_scoreboard": (
        "waiting on shared memory or MIO",
        ("smem_layout", "vector_width"),
    ),
    "barrier": (
        "warps idling at __syncthreads",
        ("tile_shape", "smem_layout", "launch_config"),
    ),
    "mio_throttle": (
        "MIO instruction queue saturated (often shared-memory bank conflicts)",
        ("smem_layout", "vector_width"),
    ),
    "lg_throttle": (
        "local/global instruction queue saturated",
        ("vector_width", "memory_hint"),
    ),
    "math_pipe_throttle": (
        "the math pipe is the bottleneck",
        ("tile_shape", "unroll", "redundant_work"),
    ),
    "no_instruction": (
        "instruction fetch stalls (large loop body or poor locality)",
        ("unroll", "tile_shape"),
    ),
    "wait": (
        "fixed-latency dependency stalls",
        ("unroll", "split_k", "launch_config"),
    ),
}

MEMORY_BOUND = "memory-bound"
COMPUTE_BOUND = "compute-bound"
OCCUPANCY_LIMITED = "occupancy-limited"
LATENCY_BOUND = "latency-bound"

VERDICT_FAMILIES: dict[str, tuple[str, ...]] = {
    MEMORY_BOUND: ("vector_width", "memory_hint", "smem_layout", "fusion"),
    COMPUTE_BOUND: ("tile_shape", "unroll", "redundant_work"),
    OCCUPANCY_LIMITED: ("launch_config", "unroll", "tile_shape"),
    LATENCY_BOUND: ("unroll", "split_k", "launch_config", "fusion"),
}


@dataclass
class KernelProfile:
    name: str
    time_ns: float = 0.0
    launches: int = 0
    compute_pct: float = 0.0        # SM throughput, % of peak
    memory_pct: float = 0.0         # memory throughput, % of peak
    occupancy_pct: float = 0.0      # achieved occupancy
    registers: float = 0.0
    l1_hit_pct: float = 0.0
    stalls: dict[str, float] = field(default_factory=dict)
    time_share: float = 0.0         # fraction of total profiled time

    @property
    def base_name(self) -> str:
        """Identifier without template arguments or the parameter list."""
        head = self.name.split("(")[0]
        head = re.sub(r"<[^<>]*(?:<[^<>]*>[^<>]*)*>", "", head)
        return head.strip().split("::")[-1].split()[-1]

    @property
    def top_stall(self) -> tuple[str, float]:
        if not self.stalls:
            return ("", 0.0)
        reason = max(self.stalls, key=lambda k: self.stalls[k])
        return (reason, self.stalls[reason])

    @property
    def verdict(self) -> str:
        """Which of the four bottleneck regimes this kernel is in."""
        if self.memory_pct >= 60 and self.memory_pct > self.compute_pct:
            return MEMORY_BOUND
        if self.compute_pct >= 60:
            return COMPUTE_BOUND
        # Neither pipe is saturated. Low occupancy means there is not enough work
        # in flight to hide latency; otherwise the warps are there but stalled.
        if self.occupancy_pct and self.occupancy_pct < 40:
            return OCCUPANCY_LIMITED
        return LATENCY_BOUND

    def suggested_families(self) -> tuple[str, ...]:
        """Families worth trying first, from the verdict and the top stall."""
        out: list[str] = list(VERDICT_FAMILIES.get(self.verdict, ()))
        reason, _ = self.top_stall
        for fam in STALL_GUIDANCE.get(reason, ("", ()))[1]:
            if fam not in out:
                out.append(fam)
        return tuple(out)

    def hint(self) -> str:
        """Compact profiler summary for the proposer prompt."""
        lines = [
            f"{self.base_name} is {self.verdict}. "
            f"{self.time_share * 100:.0f}% of profiled GPU time over {self.launches} launches."
        ]
        lines.append(
            f"  SM throughput {self.compute_pct:.0f}% of peak, "
            f"memory throughput {self.memory_pct:.0f}% of peak, "
            f"achieved occupancy {self.occupancy_pct:.0f}%."
        )
        if self.registers:
            lines.append(f"  {self.registers:.0f} registers per thread.")
        reason, share = self.top_stall
        if reason:
            meaning = STALL_GUIDANCE.get(reason, ("", ()))[0]
            detail = f" -- {meaning}" if meaning else ""
            lines.append(f"  Dominant warp stall: {reason} ({share:.2f} per issue-active){detail}.")
        return "\n".join(lines)


def parse_ncu_csv(text: str) -> list[KernelProfile]:
    """Parse `ncu --csv --page raw` long-format output into per-kernel records.

    Rows are (kernel launch, metric) pairs, so metrics are aggregated per kernel:
    durations summed, everything else averaged across launches.
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []

    name_col = _column(reader.fieldnames, "Kernel Name")
    metric_col = _column(reader.fieldnames, "Metric Name")
    value_col = _column(reader.fieldnames, "Metric Value")
    id_col = _column(reader.fieldnames, "ID")
    if not (name_col and metric_col and value_col):
        return []

    acc: dict[str, dict[str, list[float]]] = {}
    launches: dict[str, set[str]] = {}

    for row in reader:
        name = (row.get(name_col) or "").strip()
        metric = (row.get(metric_col) or "").strip()
        value = _number(row.get(value_col))
        if not name or not metric or value is None:
            continue
        acc.setdefault(name, {}).setdefault(metric, []).append(value)
        launches.setdefault(name, set()).add((row.get(id_col) or "").strip() if id_col else metric)

    profiles: list[KernelProfile] = []
    for name, metrics in acc.items():
        prof = KernelProfile(name=name, launches=len(launches.get(name, ())) or 1)
        for metric, values in metrics.items():
            mean = sum(values) / len(values)
            if metric.startswith("gpu__time_duration"):
                prof.time_ns = sum(values)
            elif metric.startswith("sm__throughput"):
                prof.compute_pct = mean
            elif metric.startswith("gpu__compute_memory_throughput"):
                prof.memory_pct = mean
            elif metric.startswith("sm__warps_active"):
                prof.occupancy_pct = mean
            elif metric.startswith("launch__registers_per_thread"):
                prof.registers = mean
            elif metric.startswith("l1tex__t_sector_hit_rate"):
                prof.l1_hit_pct = mean
            else:
                stall = _STALL_RE.search(metric)
                if stall:
                    prof.stalls[stall.group("reason")] = mean
        profiles.append(prof)

    total = sum(p.time_ns for p in profiles)
    for p in profiles:
        p.time_share = (p.time_ns / total) if total else 0.0
    profiles.sort(key=lambda p: p.time_ns, reverse=True)
    return profiles


def _column(fieldnames: list[str], want: str) -> str | None:
    for f in fieldnames:
        if (f or "").strip().lower() == want.lower():
            return f
    return None


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = str(raw).strip().replace(",", "").replace("%", "")
    if not cleaned or cleaned.lower() in ("n/a", "nan", "-"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class Profiler:
    """Runs ncu over the benchmark and maps hot kernels back to source."""

    def __init__(
        self,
        repo: Path,
        bench_cmd: str,
        *,
        ncu_bin: str = "",
        launch_count: int = 12,
        timeout_s: float = 900.0,
    ) -> None:
        self.repo = repo
        self.bench_cmd = bench_cmd
        self.ncu_bin = ncu_bin or os.environ.get("CUDA_AGENT_NCU", "ncu")
        self.launch_count = launch_count
        self.timeout_s = timeout_s
        self.last_error = ""

    def available(self) -> bool:
        return shutil.which(self.ncu_bin) is not None

    def collect(self, path: Path) -> list[KernelProfile]:
        """Profile the benchmark in `path`. Returns [] when profiling is unusable."""
        if not self.available():
            self.last_error = f"{self.ncu_bin} not found"
            return []

        cmd = [
            self.ncu_bin,
            "--csv",
            "--page", "raw",
            "--target-processes", "all",
            "--kernel-name-base", "demangled",
            "--launch-count", str(self.launch_count),
            "--metrics", ",".join(METRICS),
            "--", "bash", "-lc", self.bench_cmd,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.repo), capture_output=True, text=True, timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired:
            self.last_error = f"ncu timed out after {self.timeout_s:.0f}s"
            return []

        out = proc.stdout or ""
        if proc.returncode != 0 and "Kernel Name" not in out:
            # ncu reports its own errors on STDOUT, while the profiled program's
            # chatter lands on stderr. Scanning only stderr found the workload's
            # progress messages and reported one of those as "the ncu error",
            # which is worse than useless -- it hides a diagnosable cause behind
            # a meaningless line. Scan both.
            both = f"{out}\n{proc.stderr or ''}"
            if "ERR_NVGPUCTRPERM" in both or "does not have permission" in both:
                self.last_error = (
                    "profiling permission denied (ERR_NVGPUCTRPERM) -- GPU performance "
                    "counters are admin-gated on this host; targeting falls back to round-robin"
                )
            elif "not supported" in both.lower():
                self.last_error = "ncu: profiling not supported on this device"
            else:
                # Prefer ncu's own ==ERROR==/==WARNING== lines over a blind tail.
                marked = [ln for ln in both.splitlines() if ln.startswith(("==ERROR==", "==WARNING=="))]
                detail = " | ".join(marked[-3:]) if marked else both.strip()[-300:]
                self.last_error = f"ncu failed: {detail}"
            return []

        profiles = parse_ncu_csv(out)
        if not profiles:
            self.last_error = "ncu produced no parseable rows"
        return profiles


def build_symbol_index(repo: Path, roots: tuple[str, ...] = ("kernels",)) -> dict[str, str]:
    """Map kernel identifier -> repo-relative source path."""
    index: dict[str, str] = {}
    for root in roots:
        base = repo / root
        if not base.is_dir():
            continue
        for cu in sorted(base.rglob("*.cu")):
            rel = str(cu.relative_to(repo))
            for sym in list_symbols(repo, rel, limit=200):
                index.setdefault(sym, rel)
    return index


def rank_targets(
    profiles: list[KernelProfile],
    index: dict[str, str],
    *,
    limit: int = 12,
    min_share: float = 0.01,
) -> list[tuple[Target, KernelProfile]]:
    """Hot kernels that map to editable source, hottest first.

    A kernel with no source match is dropped rather than guessed at: it is
    usually a cuBLAS/CUTLASS call the agent cannot edit anyway, and proposing
    edits against a file that does not contain it wastes the round.
    """
    ranked: list[tuple[Target, KernelProfile]] = []
    for prof in profiles:
        if prof.time_share < min_share:
            continue
        path = index.get(prof.base_name)
        if not path:
            continue
        ranked.append((Target(path, prof.base_name), prof))
        if len(ranked) >= limit:
            break
    return ranked
