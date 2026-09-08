"""The measurement oracle.

The agent never decides whether a change was good -- this does, and it decides on
numbers. The gate order is deliberate and matches sparkinfer's own eval loop:
build, then accuracy, then speed. Accuracy first because a speed win that erodes
parity with the reference is not a win, and because six silent correctness bugs
during Qwen3.8 bring-up all left throughput untouched.

`Harness` is a protocol so the loop can be exercised against a fake without a
GPU. The subprocess implementation binds to sparkinfer's bench scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cuda_agent.ledger import ACCEPTED, ACCURACY_FAILED, BUILD_FAILED, NOT_FASTER

# Below this, a difference is measurement noise rather than a speedup. Matches
# the 2% significance floor the sparkinfer eval loop uses.
SIGNIFICANCE = 0.02

# Any guarded context regressing past this fraction of the frontier fails, even
# if another context improved.
REGRESSION_TOL = 0.98


@dataclass
class BuildResult:
    ok: bool
    log: str = ""
    elapsed_s: float = 0.0


@dataclass
class AccuracyResult:
    ok: bool
    top1: float = 0.0
    kl: float = 0.0
    detail: str = ""


@dataclass
class BenchResult:
    ok: bool
    # context label -> decode tok/s, e.g. {"4k": 93.6, "32k": 81.3}
    tps: dict[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass
class Evaluation:
    verdict: str
    detail: str = ""
    speedup: float = 0.0
    tps: dict[str, float] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verdict == ACCEPTED


class Harness(Protocol):
    def build(self, path: Path) -> BuildResult: ...
    def accuracy(self, path: Path) -> AccuracyResult: ...
    def bench(self, path: Path, *, quick: bool = False) -> BenchResult: ...


def score(baseline: dict[str, float], candidate: dict[str, float]) -> tuple[float, str]:
    """Best relative gain across contexts, or a regression verdict.

    Returns (gain, detail). A negative gain means some guarded context regressed
    past tolerance -- a gain elsewhere does not excuse it.
    """
    if not baseline or not candidate:
        return 0.0, "no measurements"

    shared = [k for k in baseline if k in candidate and baseline[k] > 0]
    if not shared:
        return 0.0, "no comparable contexts"

    regressed = [k for k in shared if candidate[k] / baseline[k] < REGRESSION_TOL]
    if regressed:
        worst = min(regressed, key=lambda k: candidate[k] / baseline[k])
        pct = (candidate[worst] / baseline[worst] - 1) * 100
        return -1.0, f"regression at {worst}: {pct:.1f}%"

    best = max(shared, key=lambda k: candidate[k] / baseline[k])
    gain = candidate[best] / baseline[best] - 1
    return gain, f"{best}: {baseline[best]:.1f} -> {candidate[best]:.1f} tok/s ({gain * 100:+.1f}%)"


def evaluate(
    harness: Harness,
    path: Path,
    baseline: dict[str, float],
    *,
    quick: bool = False,
) -> Evaluation:
    """Run the full gate on a candidate worktree."""
    build = harness.build(path)
    if not build.ok:
        return Evaluation(BUILD_FAILED, _tail(build.log))

    acc = harness.accuracy(path)
    if not acc.ok:
        return Evaluation(
            ACCURACY_FAILED,
            acc.detail or f"top1={acc.top1:.3f} kl={acc.kl:.3f}",
        )

    bench = harness.bench(path, quick=quick)
    if not bench.ok:
        return Evaluation(NOT_FASTER, bench.detail or "benchmark failed")

    gain, detail = score(baseline, bench.tps)
    if gain < 0:
        return Evaluation(NOT_FASTER, detail, speedup=gain, tps=bench.tps)
    if gain < SIGNIFICANCE:
        return Evaluation(NOT_FASTER, detail, speedup=gain, tps=bench.tps)
    return Evaluation(ACCEPTED, detail, speedup=gain, tps=bench.tps)


def _tail(log: str, lines: int = 20) -> str:
    """Last few lines of a build log -- the compiler error, not the whole run.

    Also what gets fed back to the proposer, which cannot afford the full log in
    its context.
    """
    rows = [r for r in (log or "").splitlines() if r.strip()]
    return "\n".join(rows[-lines:])
