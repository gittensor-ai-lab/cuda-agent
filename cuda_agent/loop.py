"""The optimiser loop.

The loop is deterministic Python and the model is a subroutine inside it. The
model proposes an edit; the harness builds, checks correctness, and measures; the
loop decides. The model never gets to keep its own change, and never sees a
number it produced itself treated as a result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from cuda_agent.backend import Gateway, GatewayExhausted
from cuda_agent.config import Settings
from cuda_agent.edits import EditError, apply_all, looks_like_attempted_edit, parse_edits
from cuda_agent.families import FamilyBandit
from cuda_agent.harness import Harness, evaluate
from cuda_agent.ledger import ACCEPTED, KNOWN_BAD, MALFORMED, Attempt, Ledger
from cuda_agent.profile import Profiler, build_symbol_index, rank_targets
from cuda_agent.prompts import build_proposal, build_repair
from cuda_agent.source import Target, read_target
from cuda_agent.worktree import Frontier


@dataclass
class RunReport:
    base_sha: str = ""
    frontier_sha: str = ""
    baseline: dict[str, float] = field(default_factory=dict)
    final: dict[str, float] = field(default_factory=dict)
    total_speedup: float = 0.0
    attempts: int = 0
    accepted: int = 0
    autotune_evals: int = 0
    by_family: dict = field(default_factory=dict)
    profiled: bool = False
    profile_error: str = ""
    skipped_known_bad: int = 0
    memory: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    rate_limited: int = 0
    served_uids: dict[str, int] = field(default_factory=dict)
    bandit: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)


class Optimizer:
    def __init__(
        self,
        settings: Settings,
        gateway: Gateway,
        harness: Harness,
        frontier: Frontier,
        targets: list[Target],
        *,
        ledger: Ledger | None = None,
        bandit: FamilyBandit | None = None,
        autotuner=None,
        autotune_evals: int = 0,
        autotune_fraction: float = 0.4,
        profiler: Profiler | None = None,
        reprofile_after: int = 3,
        memory=None,
        clock=time.monotonic,
    ) -> None:
        if not targets:
            raise ValueError("no targets to optimise")
        self.settings = settings
        self.gateway = gateway
        self.harness = harness
        self.frontier = frontier
        self.targets = targets
        self.ledger = ledger or Ledger(settings.out_dir / "ledger.jsonl")
        self.bandit = bandit or FamilyBandit()
        self.autotuner = autotuner
        self.autotune_evals = autotune_evals
        self.autotune_fraction = autotune_fraction
        self.profiler = profiler
        self.reprofile_after = reprofile_after
        self.memory = memory
        self.skipped_known_bad = 0
        self.clock = clock
        self._cursor = 0
        self.autotune_spent = 0
        self.profile_error = ""
        self._hints: dict[str, str] = {}
        self._accepted_since_profile = 0

    # -- budget ---------------------------------------------------------------

    def _out_of_budget(self, started: float) -> bool:
        if self.clock() - started >= self.settings.wall_clock_s:
            return True
        cap = self.settings.max_completion_tokens
        return bool(cap) and self.gateway.usage.completion_tokens >= cap

    def _next_target(self) -> Target:
        target = self.targets[self._cursor % len(self.targets)]
        self._cursor += 1
        return target

    def _reprofile(self, path: Path) -> None:
        """Re-rank targets from a fresh profile, and seed the bandit from it.

        Called at the start of the round and again after the frontier has moved
        a few times: fixing the top kernel moves the bottleneck, and continuing
        to hammer it is how an agent spends an hour on a kernel that stopped
        being the problem.
        """
        if not self.profiler:
            return
        profiles = self.profiler.collect(path)
        if not profiles:
            self.profile_error = self.profiler.last_error
            return

        ranked = rank_targets(profiles, build_symbol_index(self.frontier.repo))
        if not ranked:
            self.profile_error = "no hot kernel mapped to editable source"
            return

        # Weight the rotation by share of GPU time: a kernel owning 40% of the
        # time should be visited more often than one owning 3%.
        weighted: list[Target] = []
        for target, prof in ranked:
            self._hints[target.key] = prof.hint()
            # This round's profile is fresher evidence than last round's outcomes,
            # so it leads; remembered winners follow rather than being discarded.
            families = list(prof.suggested_families())
            if self.memory:
                for name in self.memory.preferred_families(target.key):
                    if name not in families:
                        families.append(name)
            self.bandit.set_preference(target.key, tuple(families))
            weighted.extend([target] * max(1, round(prof.time_share * 10)))

        self.targets = weighted or [t for t, _ in ranked]
        self._cursor = 0
        self._accepted_since_profile = 0
        self.profile_error = ""

    # -- main loop ------------------------------------------------------------

    def run(self) -> RunReport:
        started = self.clock()

        base = self.harness.bench(self.frontier.repo, quick=False)
        if not base.ok or not base.tps:
            raise RuntimeError(f"could not establish a baseline: {base.detail}")
        baseline = dict(base.tps)
        first = dict(baseline)

        # Deterministic sweep first. It costs no tokens, it does not depend on the
        # model being clever, and whatever it banks raises the bar the proposer
        # then has to beat -- every later candidate is measured against the swept
        # frontier, so the two phases cannot double-count the same gain.
        if self.autotuner and self.autotune_evals > 0:
            deadline = started + self.settings.wall_clock_s * self.autotune_fraction
            paths = list(dict.fromkeys(t.path for t in self.targets))
            baseline, self.autotune_spent = self.autotuner.run(
                paths,
                baseline,
                max_evals=self.autotune_evals,
                should_stop=lambda: self.clock() >= deadline or self._out_of_budget(started),
            )

        if self.memory:
            self.memory.seed(self.bandit, [t.key for t in self.targets])

        if self.profiler:
            self._reprofile(self.frontier.repo)

        while not self._out_of_budget(started):
            if (self.profiler and self.reprofile_after
                    and self._accepted_since_profile >= self.reprofile_after):
                self._reprofile(self.frontier.repo)

            target = self._next_target()
            family = self.bandit.select(target.key)
            idx = self.ledger.next_idx()
            t0 = self.clock()

            try:
                edits, tokens, uid, err = self._propose(target, family, baseline)
            except GatewayExhausted as exc:
                # Sustained 429 is a capacity problem, not an agent problem. Stop
                # cleanly and report it rather than burning the clock on retries.
                self.ledger.record(Attempt(idx, target.key, family.name, MALFORMED,
                                           detail=f"gateway exhausted: {exc}",
                                           elapsed_s=self.clock() - t0))
                break

            if not edits:
                self.bandit.update(target.key, family.name, accepted=False)
                self.ledger.record(Attempt(idx, target.key, family.name, MALFORMED,
                                           detail=err, completion_tokens=tokens,
                                           served_uid=uid, elapsed_s=self.clock() - t0))
                continue

            remembered = self._recall(edits)
            if remembered:
                self.skipped_known_bad += 1
                self.bandit.update(target.key, family.name, accepted=False)
                self.ledger.record(Attempt(idx, target.key, family.name, KNOWN_BAD,
                                           detail=remembered, completion_tokens=tokens,
                                           served_uid=uid, elapsed_s=self.clock() - t0))
                continue

            cand = self.frontier.checkout()
            try:
                try:
                    paths = apply_all(
                        cand.path, edits,
                        allowed=self.settings.allowed_paths,
                        denied=self.settings.denied_paths,
                    )
                except EditError as exc:
                    self.bandit.update(target.key, family.name, accepted=False)
                    self.ledger.record(Attempt(idx, target.key, family.name, MALFORMED,
                                               detail=str(exc), completion_tokens=tokens,
                                               served_uid=uid, elapsed_s=self.clock() - t0))
                    continue

                ev = evaluate(self.harness, cand.path, baseline)
                reward = self.bandit.update(target.key, family.name,
                                            accepted=ev.accepted, speedup=ev.speedup)
                if self.memory:
                    self.memory.note_family(target.key, family.name,
                                            accepted=ev.accepted, reward=reward)
                    if ev.accepted:
                        self.memory.note_win(target.path, family.name, ev.detail, ev.speedup)
                    else:
                        for e in edits:
                            self.memory.note_rejected(e.path, e.search, e.replace,
                                                      verdict=ev.verdict, detail=ev.detail)
                self.ledger.record(Attempt(idx, target.key, family.name, ev.verdict,
                                           detail=ev.detail, paths=paths,
                                           speedup=ev.speedup, completion_tokens=tokens,
                                           served_uid=uid, elapsed_s=self.clock() - t0))

                if ev.accepted:
                    self.frontier.promote(
                        cand, f"perf({target.path}): {family.name} ({ev.detail})"
                    )
                    # The frontier moved, so every later candidate is measured
                    # against the new best, not the round's starting point. This
                    # is what makes each accepted gain marginal rather than
                    # double-counted.
                    baseline = dict(ev.tps) or baseline
                    self._accepted_since_profile += 1
            finally:
                self.frontier.discard(cand)

        return self._report(started, first, baseline)

    def _recall(self, edits) -> str:
        """Why an earlier round already rejected this exact edit, if it did."""
        if not self.memory:
            return ""
        for e in edits:
            verdict = self.memory.known_bad(self.frontier.repo, e.path, e.search, e.replace)
            if verdict:
                return f"already {verdict} in an earlier round"
        return ""

    # -- proposal -------------------------------------------------------------

    def _propose(self, target: Target, family, baseline: dict[str, float]):
        """Ask for one edit, with a single cheap repair attempt.

        Returns (edits, completion_tokens, served_uid, error_detail).
        """
        source = read_target(self.frontier.repo, target)
        messages = build_proposal(
            target=target,
            family=family,
            source=source,
            history=self.ledger.history_block(target.key),
            baseline=baseline,
            profile=self._hints.get(target.key, ""),
        )
        completion = self.gateway.complete(messages)
        tokens = completion.usage.completion_tokens
        edits = parse_edits(completion.text)
        if edits:
            return edits, tokens, completion.served_uid, ""

        # No parseable edit. If it *tried* and the block was malformed or the
        # answer ran into the token cap, one correction is much cheaper than a
        # wasted iteration; if it answered in prose, the family probably does not
        # apply and the bandit should learn that.
        if not looks_like_attempted_edit(completion.text) and not completion.truncated:
            return [], tokens, completion.served_uid, "no edit emitted (prose answer)"

        reason = "answer truncated at the token cap" if completion.truncated else "malformed edit block"
        repair = self.gateway.complete(build_repair(completion.text, reason))
        tokens += repair.usage.completion_tokens
        edits = parse_edits(repair.text)
        return edits, tokens, repair.served_uid or completion.served_uid, "" if edits else reason

    # -- report ---------------------------------------------------------------

    def _report(self, started: float, first: dict[str, float], final: dict[str, float]) -> RunReport:
        summary = self.ledger.summary()
        gains = [c / first[k] for k, c in final.items() if k in first and first[k] > 0]
        usage = self.gateway.usage
        return RunReport(
            base_sha=self.frontier.base_sha,
            frontier_sha=self.frontier.sha,
            baseline=first,
            final=final,
            total_speedup=(max(gains) - 1) if gains else 0.0,
            attempts=summary["attempts"],
            accepted=summary["accepted"],
            autotune_evals=self.autotune_spent,
            profiled=bool(self._hints),
            profile_error=self.profile_error,
            skipped_known_bad=self.skipped_known_bad,
            memory=self.memory.summary() if self.memory else {},
            by_family=summary.get("by_family", {}),
            elapsed_s=self.clock() - started,
            completion_tokens=usage.completion_tokens,
            est_cost_usd=round(usage.est_cost_usd, 4),
            rate_limited=usage.rate_limited,
            served_uids=dict(usage.served_uids),
            bandit=self.bandit.snapshot(),
            ledger=summary,
        )
