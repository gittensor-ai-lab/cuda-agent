"""Memory that survives the round.

A daily competition rewards an agent that remembers yesterday. Three things are
worth carrying, and they pay off in different currencies:

  * **Which families paid on which kernel.** Cheap to store, and with only ~20
    proposer iterations in a round, starting in the right part of the search
    space is most of the game.
  * **Edits already tried and rejected.** The expensive one. Re-proposing a
    change that failed to build last week costs a full build-and-benchmark cycle
    -- several minutes of a three-hour budget -- for a result already known.
  * **Knob values already swept.** Same argument, applied to the sweep.

Negative memory is dangerous if it is too durable: the code moves, and a change
that failed against last week's kernel may be exactly right against today's. So a
rejection is only trusted while its anchor text still exists unchanged in the
file, and it expires by age regardless. Suppressing a good idea forever is a
worse failure than re-running one build.

As with the profiler's priors, remembered family statistics reorder *exploration*
only. Value still comes from measurement in the current round -- a kernel that
was compute-bound yesterday may not be today.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 1

# A rejection is distrusted after this many rounds even if its anchor survives.
DEFAULT_MAX_AGE_ROUNDS = 7


def edit_key(path: str, search: str, replace: str) -> str:
    """Content-addressed identity for an edit, independent of commit."""
    h = hashlib.sha256()
    h.update(path.encode("utf-8", "surrogateescape"))
    h.update(b"\0")
    h.update(search.encode("utf-8", "surrogateescape"))
    h.update(b"\0")
    h.update(replace.encode("utf-8", "surrogateescape"))
    return h.hexdigest()[:32]


@dataclass
class _FamilyStat:
    tried: int = 0
    accepted: int = 0
    reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward / self.tried if self.tried else 0.0


@dataclass
class Memory:
    path: Path
    max_age_rounds: int = DEFAULT_MAX_AGE_ROUNDS

    round_no: int = 0
    families: dict[str, dict[str, _FamilyStat]] = field(default_factory=dict)
    # edit_key -> {"path", "anchor", "verdict", "detail", "round"}
    rejected: dict[str, dict] = field(default_factory=dict)
    # "path::name" -> [values already tried]
    knobs: dict[str, list[int]] = field(default_factory=dict)
    wins: list[dict] = field(default_factory=list)

    # -- persistence ----------------------------------------------------------

    @classmethod
    def load(cls, path: Path, *, max_age_rounds: int = DEFAULT_MAX_AGE_ROUNDS) -> "Memory":
        mem = cls(path=path, max_age_rounds=max_age_rounds)
        if not path.exists():
            return mem
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt store must not cost the round. Start clean.
            return mem
        if raw.get("schema") != SCHEMA:
            return mem

        mem.round_no = int(raw.get("round_no", 0))
        for kernel, fams in (raw.get("families") or {}).items():
            mem.families[kernel] = {
                name: _FamilyStat(**{k: v for k, v in stat.items() if k in ("tried", "accepted", "reward")})
                for name, stat in fams.items()
            }
        mem.rejected = dict(raw.get("rejected") or {})
        mem.knobs = {k: list(v) for k, v in (raw.get("knobs") or {}).items()}
        mem.wins = list(raw.get("wins") or [])
        return mem

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "round_no": self.round_no,
            "families": {
                kernel: {name: {"tried": s.tried, "accepted": s.accepted, "reward": round(s.reward, 4)}
                         for name, s in fams.items()}
                for kernel, fams in self.families.items()
            },
            "rejected": self.rejected,
            "knobs": self.knobs,
            "wins": self.wins[-200:],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)                 # atomic: a killed agent cannot truncate it

    def start_round(self) -> int:
        self.round_no += 1
        return self.round_no

    # -- family priors --------------------------------------------------------

    def preferred_families(self, kernel: str) -> tuple[str, ...]:
        """Families for this kernel, best-performing first."""
        stats = self.families.get(kernel) or {}
        ranked = sorted(stats.items(), key=lambda kv: (kv[1].mean, kv[1].accepted), reverse=True)
        return tuple(name for name, stat in ranked if stat.tried)

    def seed(self, bandit, kernels: list[str] | None = None) -> int:
        """Order each kernel's unexplored arms by what paid in earlier rounds."""
        seeded = 0
        for kernel in (kernels if kernels is not None else list(self.families)):
            preferred = self.preferred_families(kernel)
            if preferred:
                bandit.set_preference(kernel, preferred)
                seeded += 1
        return seeded

    def note_family(self, kernel: str, family: str, *, accepted: bool, reward: float) -> None:
        stat = self.families.setdefault(kernel, {}).setdefault(family, _FamilyStat())
        stat.tried += 1
        stat.accepted += int(accepted)
        stat.reward += reward

    # -- negative memory ------------------------------------------------------

    def known_bad(self, repo: Path, path: str, search: str, replace: str) -> str | None:
        """Why this exact edit was rejected before, or None if it is worth trying.

        Trusted only while the anchor still exists in the file and the record is
        recent: code moves, and an idea that failed against an older version of a
        kernel may be right against this one.
        """
        record = self.rejected.get(edit_key(path, search, replace))
        if not record:
            return None
        if self.round_no - int(record.get("round", 0)) > self.max_age_rounds:
            return None

        anchor = record.get("anchor") or search
        try:
            current = (repo / path).read_text(encoding="utf-8", errors="surrogateescape")
        except OSError:
            return None
        if current.count(anchor) != 1:
            return None                        # the code moved; the memory no longer applies
        return str(record.get("verdict") or "rejected")

    def note_rejected(self, path: str, search: str, replace: str, *, verdict: str, detail: str = "") -> None:
        self.rejected[edit_key(path, search, replace)] = {
            "path": path,
            "anchor": search,
            "verdict": verdict,
            "detail": detail[:200],
            "round": self.round_no,
            "ts": int(time.time()),
        }

    def note_win(self, path: str, family: str, detail: str, speedup: float) -> None:
        self.wins.append({
            "path": path, "family": family, "detail": detail[:200],
            "speedup": round(speedup, 4), "round": self.round_no,
        })

    # -- sweep memory ---------------------------------------------------------

    def swept(self, knob_key: str) -> set[int]:
        return set(self.knobs.get(knob_key, ()))

    def note_swept(self, knob_key: str, value: int) -> None:
        values = self.knobs.setdefault(knob_key, [])
        if value not in values:
            values.append(value)

    # -- reporting ------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "round": self.round_no,
            "kernels_known": len(self.families),
            "rejections_remembered": len(self.rejected),
            "knobs_swept": sum(len(v) for v in self.knobs.values()),
            "wins_recorded": len(self.wins),
        }
