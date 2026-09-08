"""Run journal: every attempt, what it changed, and what it measured.

Two jobs, and the second is the important one:

  1. An auditable record of the round, written as it happens so a killed agent
     still leaves evidence.
  2. The proposer's memory. A weak model in a loop will otherwise re-propose the
     same failed idea until the budget runs out, so the last few verdicts for a
     target are fed straight back into its prompt.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Verdicts. Anything that is not ACCEPTED left the frontier untouched.
MALFORMED = "malformed"          # the model did not emit an applicable edit
BUILD_FAILED = "build-failed"
ACCURACY_FAILED = "accuracy-failed"
NOT_FASTER = "not-faster"
KNOWN_BAD = "skipped-known-bad"   # remembered from an earlier round; never built
ACCEPTED = "accepted"


@dataclass
class Attempt:
    idx: int
    target: str
    family: str
    verdict: str
    detail: str = ""
    paths: list[str] = field(default_factory=list)
    speedup: float = 0.0            # fraction over the frontier, best context
    elapsed_s: float = 0.0
    completion_tokens: int = 0
    served_uid: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.verdict == ACCEPTED


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts: list[Attempt] = []

    def record(self, attempt: Attempt) -> Attempt:
        self.attempts.append(attempt)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(attempt)) + "\n")
        return attempt

    def next_idx(self) -> int:
        return len(self.attempts) + 1

    # -- prompt context -------------------------------------------------------

    def recent_for(self, target: str, limit: int = 6) -> list[Attempt]:
        return [a for a in self.attempts if a.target == target][-limit:]

    def history_block(self, target: str, limit: int = 6) -> str:
        """Render prior attempts on this target for the proposer prompt.

        Kept terse on purpose -- prompt tokens spent here are tokens not spent on
        source, and the model only needs to know what has already been ruled out.
        """
        rows = self.recent_for(target, limit)
        if not rows:
            return "No previous attempts on this target."
        lines = []
        for a in rows:
            if a.verdict == ACCEPTED:
                outcome = f"ACCEPTED (+{a.speedup * 100:.1f}%)"
            else:
                outcome = a.verdict.upper()
            detail = f" -- {a.detail}" if a.detail else ""
            lines.append(f"  [{a.family}] {outcome}{detail}")
        return "Previous attempts on this target:\n" + "\n".join(lines)

    # -- reporting ------------------------------------------------------------

    def summary(self) -> dict:
        by_verdict: dict[str, int] = {}
        for a in self.attempts:
            by_verdict[a.verdict] = by_verdict.get(a.verdict, 0) + 1
        wins = [a for a in self.attempts if a.ok]
        by_family: dict[str, dict[str, int]] = {}
        for a in self.attempts:
            row = by_family.setdefault(a.family, {"tried": 0, "accepted": 0})
            row["tried"] += 1
            row["accepted"] += int(a.ok)
        return {
            "by_family": by_family,
            "attempts": len(self.attempts),
            "by_verdict": by_verdict,
            "accepted": len(wins),
            "best_speedup": max((a.speedup for a in wins), default=0.0),
            "completion_tokens": sum(a.completion_tokens for a in self.attempts),
        }
