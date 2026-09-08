"""Optimization hypothesis families, and a bandit that picks between them.

Free-form "look at this kernel and think of something" wastes a 3B-active
proposer. Instead the search is structured into named families, each with its own
prompt hint, and a UCB1 bandit learns which families pay off on which kernel.

This is also the only source of diversity available: the gateway ignores sampling
parameters and decodes greedily, so two identical prompts return the identical
answer. Variation has to come from *what* is asked, not from temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A gain of this size counts as a full-credit success for the bandit; anything
# larger saturates. Roughly the boundary between sparkinfer's L and XL tiers.
REWARD_SATURATION = 0.10


@dataclass(frozen=True)
class Family:
    name: str
    hint: str


FAMILIES: tuple[Family, ...] = (
    Family(
        "tile_shape",
        "Change the tile or block-tile dimensions so each block does more work per "
        "load, or so the tile matches the problem's shape (heads, head_dim, batch) "
        "with less waste at the edges.",
    ),
    Family(
        "launch_config",
        "Change threads-per-block, the grid shape, or __launch_bounds__ so occupancy "
        "or register pressure improves. Small launches with idle SMs are the usual win.",
    ),
    Family(
        "vector_width",
        "Widen loads and stores to float4/int4 (16-byte) accesses, or fix an access "
        "pattern that prevents vectorisation, so the memory pipe issues fewer, wider "
        "transactions.",
    ),
    Family(
        "smem_layout",
        "Pad or swizzle a shared-memory array to remove bank conflicts, or restructure "
        "the staging so loads are coalesced.",
    ),
    Family(
        "unroll",
        "Add, remove or retune #pragma unroll so the inner loop trades register "
        "pressure against issue overhead more favourably.",
    ),
    Family(
        "fusion",
        "Fold an elementwise epilogue into the producing kernel, or merge two adjacent "
        "launches, removing a round trip through global memory and one launch overhead.",
    ),
    Family(
        "memory_hint",
        "Apply cache-control hints (__ldg, .cg/.cs/.lu cache modifiers, cp.async "
        "prefetch, EVICT_FIRST/EVICT_LAST) so streaming data does not evict data that "
        "is reused.",
    ),
    Family(
        "split_k",
        "Split a long reduction across more blocks and combine the partials, trading a "
        "second pass for parallelism when the grid is too small to fill the GPU.",
    ),
    Family(
        "redundant_work",
        "Hoist a loop invariant, skip work on masked or out-of-range elements, or drop "
        "a recomputation that a cheap cache would serve.",
    ),
)

BY_NAME = {f.name: f for f in FAMILIES}


@dataclass
class _Arm:
    pulls: int = 0
    reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward / self.pulls if self.pulls else 0.0


class FamilyBandit:
    """UCB1 over families, scored independently per target.

    Per-target because the answer genuinely differs: a GEMV kernel and a
    flash-decode kernel reward different families, and pooling them would average
    away the signal the agent is trying to learn within a single round.
    """

    def __init__(self, families: tuple[Family, ...] = FAMILIES, *, c: float = 1.4) -> None:
        self.families = families
        self.c = c
        self._arms: dict[tuple[str, str], _Arm] = {}
        self._total: dict[str, int] = {}
        self._preference: dict[str, tuple[str, ...]] = {}

    def set_preference(self, target: str, families: tuple[str, ...]) -> None:
        """Order the unexplored arms for a target (profiler-supplied prior).

        Only the *exploration order* moves. Once an arm has been pulled its value
        comes from measurement, so a wrong prior costs a few early iterations
        rather than biasing the whole round.
        """
        self._preference[target] = tuple(f for f in families if f in BY_NAME)

    def _arm(self, target: str, family: str) -> _Arm:
        return self._arms.setdefault((target, family), _Arm())

    def select(self, target: str) -> Family:
        """Pick the next family for this target: unexplored arms first, then UCB1."""
        unexplored = [f for f in self.families if self._arm(target, f.name).pulls == 0]
        if unexplored:
            preferred = self._preference.get(target, ())
            for name in preferred:
                for f in unexplored:
                    if f.name == name:
                        return f
            return unexplored[0]

        total = max(1, self._total.get(target, 0))
        log_total = math.log(total)

        def score(f: Family) -> float:
            arm = self._arm(target, f.name)
            return arm.mean + self.c * math.sqrt(log_total / arm.pulls)

        return max(self.families, key=score)

    def update(self, target: str, family: str, *, accepted: bool, speedup: float = 0.0) -> float:
        """Record an outcome and return the reward that was credited."""
        reward = 0.0
        if accepted and speedup > 0:
            reward = min(1.0, speedup / REWARD_SATURATION)
        arm = self._arm(target, family)
        arm.pulls += 1
        arm.reward += reward
        self._total[target] = self._total.get(target, 0) + 1
        return reward

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        """Per-target arm statistics, for the run report."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (target, family), arm in self._arms.items():
            if arm.pulls:
                out.setdefault(target, {})[family] = {"pulls": arm.pulls, "mean": round(arm.mean, 4)}
        return out
