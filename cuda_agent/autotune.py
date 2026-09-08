"""Deterministic constant sweep.

A large share of CUDA wins are parameter choices, not insight: a tile that is one
step too narrow, an unroll factor tuned for a different architecture, a block size
inherited from a kernel that no longer looks like this one. Those are found by
*trying values*, and trying values costs no tokens and needs no cleverness.

So this runs first. It also gives the round a floor -- whatever the proposer does
afterwards has to beat what a machine found by counting. Every candidate goes
through the same build/accuracy/speed gate as an LLM proposal; nothing here is
trusted because it came from a sweep.

Search is coordinate descent, not a grid. Knobs interact, but a grid over even ten
knobs is thousands of builds and the round is three hours long.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cuda_agent.edits import Edit, apply_all
from cuda_agent.harness import Harness, evaluate
from cuda_agent.ledger import ACCEPTED, Attempt, Ledger
from cuda_agent.worktree import Frontier

FAMILY = "autotune"

# Knob kinds, most-likely-to-pay first. Tile and block geometry moves throughput;
# an unroll factor usually moves it less; an arbitrary integer constant is mostly
# noise and is only swept if budget remains.
PRIORITY = {"tile": 0, "launch_bounds": 1, "unroll": 2, "define": 3, "constexpr": 4}

# Names that mark a constant as geometry rather than an arbitrary magic number.
_GEOMETRY = re.compile(
    r"TILE|BLOCK|WARP|SPLIT|CHUNK|STAGE|PAD|VEC|VPT|WIDTH|ROWS|COLS|NWARP|NTHREAD|BN|BM|BK",
    re.IGNORECASE,
)

# Capacity and bounds constants -- buffer sizes, device counts, array limits.
# These are NOT tuning knobs: shrinking one can pass a short accuracy check and
# still overflow under a longer context or a wider batch, which is precisely the
# class of change that looks like a speedup and is really a latent bug. Skipped
# unless the name also reads as geometry (ARGMAX_ROWS_MAX is a tile bound).
_CAPACITY = re.compile(r"MAX|MIN|LIMIT|CAPACITY|COUNT|SLOTS|RESERVE|BUDGET", re.IGNORECASE)

_PATTERNS: tuple[tuple[str, str], ...] = (
    # static constexpr int TILE_N = 64;
    ("constexpr",
     r"(?P<pre>(?:static\s+)?constexpr\s+(?:int|unsigned|size_t|long)\s+(?P<name>\w+)\s*=\s*)"
     r"(?P<val>\d+)(?P<post>\s*;)"),
    # const int NWARPS = 8;
    ("constexpr",
     r"(?P<pre>const\s+int\s+(?P<name>\w+)\s*=\s*)(?P<val>\d+)(?P<post>\s*;)"),
    # #define BLOCK_SIZE 256
    ("define", r"(?P<pre>#define\s+(?P<name>\w+)\s+)(?P<val>\d+)(?P<post>\s*$)"),
    # #pragma unroll 4
    ("unroll", r"(?P<pre>#pragma\s+unroll\s+)(?P<val>\d+)(?P<post>\s*$)"),
    # __launch_bounds__(256, 4)
    ("launch_bounds", r"(?P<pre>__launch_bounds__\(\s*)(?P<val>\d+)(?P<post>\s*[,)])"),
)


@dataclass(frozen=True)
class Knob:
    path: str
    kind: str
    name: str
    value: int
    anchor: str          # exact text as it appears in the file, matched once
    pre: str
    post: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name or self.kind}@{self.value}"

    def edit(self, new_value: int) -> Edit:
        return Edit(path=self.path, search=self.anchor, replace=f"{self.pre}{new_value}{self.post}")

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.kind, 9)


def discover_knobs(repo: Path, path: str) -> list[Knob]:
    """Tunable integer constants in a file, in priority order.

    A constant whose anchor text appears more than once is skipped rather than
    guessed at -- the edit format requires an unambiguous anchor, and expanding
    context here would just move the ambiguity somewhere harder to see.
    """
    text = (repo / path).read_text(encoding="utf-8", errors="surrogateescape")
    knobs: list[Knob] = []
    seen: set[str] = set()

    for kind, pattern in _PATTERNS:
        for m in re.finditer(pattern, text, re.MULTILINE):
            anchor = m.group(0)
            if text.count(anchor) != 1 or anchor in seen:
                continue
            name = (m.groupdict().get("name") or "").strip()
            value = int(m.group("val"))
            if value <= 0:
                continue
            is_geometry = bool(_GEOMETRY.search(name))
            if name and _CAPACITY.search(name) and not is_geometry:
                continue
            # A named constant that reads as geometry is a tile knob whatever the
            # declaration looked like; anything else keeps its syntactic kind and
            # sorts to the back of the queue.
            effective = "tile" if (kind == "constexpr" and is_geometry) else kind
            seen.add(anchor)
            knobs.append(
                Knob(path=path, kind=effective, name=name, value=value,
                     anchor=anchor, pre=m.group("pre"), post=m.group("post"))
            )

    knobs.sort(key=lambda k: (k.priority, k.path, k.name))
    return knobs


def candidate_values(knob: Knob, *, width: int = 2) -> list[int]:
    """Plausible alternatives for one knob, nearest-first.

    Deliberately narrow. Each value costs a build and a benchmark, so the sweep
    tries the neighbours that usually matter rather than everything legal.
    """
    v = knob.value

    if knob.kind == "unroll":
        return [c for c in (1, 2, 4, 8, 16) if c != v][:width * 2]

    if knob.kind == "launch_bounds":
        return [c for c in (128, 256, 512, 1024) if c != v][:width * 2]

    out: list[int] = []
    if _is_pow2(v):
        # Walk outward from the current value: halve and double alternately, so
        # the cheapest-to-verify neighbours come first.
        for step in range(1, width + 1):
            for cand in (v >> step, v << step):
                if 1 <= cand <= 4096 and cand != v:
                    out.append(cand)
    else:
        for cand in (v // 2, v * 2, v - 1, v + 1):
            if 1 <= cand <= 4096 and cand != v:
                out.append(cand)

    deduped: list[int] = []
    for c in out:
        if c not in deduped:
            deduped.append(c)
    return deduped[: width * 2]


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class Autotuner:
    """Coordinate descent over discovered knobs, gated by the real harness."""

    def __init__(
        self,
        frontier: Frontier,
        harness: Harness,
        ledger: Ledger,
        *,
        allowed: tuple[str, ...],
        denied: tuple[str, ...],
        quick: bool = True,
        memory=None,
    ) -> None:
        self.frontier = frontier
        self.harness = harness
        self.ledger = ledger
        self.allowed = allowed
        self.denied = denied
        # Cross-round memory. Re-measuring a value yesterday already ruled out is
        # a wasted build, and the sweep budget is the scarcest thing it has.
        self.memory = memory
        self.skipped = 0
        # The inner sweep uses the fast single-context bench; anything it accepts
        # is re-measured by the arena's cold re-run anyway.
        self.quick = quick

    def run(
        self,
        paths: list[str],
        baseline: dict[str, float],
        *,
        max_evals: int,
        should_stop=lambda: False,
    ) -> tuple[dict[str, float], int]:
        """Sweep until the evaluation budget or the wall clock runs out.

        Returns the (possibly improved) baseline and the number of evaluations
        spent, so the caller can hand the remainder to the proposer.
        """
        evals = 0

        for path in paths:
            for knob in discover_knobs(self.frontier.repo, path):
                if evals >= max_evals or should_stop():
                    return baseline, evals

                already = self.memory.swept(knob.key) if self.memory else set()
                for value in candidate_values(knob):
                    if evals >= max_evals or should_stop():
                        return baseline, evals
                    if value in already:
                        self.skipped += 1
                        continue

                    evals += 1
                    accepted, baseline = self._try(knob, value, baseline)
                    if accepted:
                        # This knob paid; move to the next one rather than
                        # pushing further in a direction already banked.
                        break

        return baseline, evals

    def _try(self, knob: Knob, value: int, baseline: dict[str, float]) -> tuple[bool, dict[str, float]]:
        cand = self.frontier.checkout()
        idx = self.ledger.next_idx()
        try:
            try:
                apply_all(cand.path, [knob.edit(value)], allowed=self.allowed, denied=self.denied)
            except Exception as exc:
                self.ledger.record(Attempt(idx, knob.key, FAMILY, "malformed", detail=str(exc)))
                return False, baseline

            ev = evaluate(self.harness, cand.path, baseline, quick=self.quick)
            if self.memory:
                self.memory.note_swept(knob.key, value)
            self.ledger.record(Attempt(
                idx, knob.key, FAMILY, ev.verdict,
                detail=f"{knob.name or knob.kind} {knob.value} -> {value}: {ev.detail}",
                paths=[knob.path], speedup=ev.speedup,
            ))
            if ev.verdict == ACCEPTED:
                self.frontier.promote(
                    cand, f"perf({knob.path}): {knob.name or knob.kind} {knob.value} -> {value}"
                )
                return True, dict(ev.tps) or baseline
            return False, baseline
        finally:
            self.frontier.discard(cand)
