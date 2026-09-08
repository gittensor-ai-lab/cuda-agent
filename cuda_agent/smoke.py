"""Preflight checks: exercise every layer against reality, cheapest first.

This is a diagnostic, not a test suite. Unit tests prove the code agrees with
its fixtures; smoke proves it agrees with *this box, this gateway, this repo*.
The two failures that motivated it -- a bench regex that matched the wrong field
order, and an accuracy gate that could not tell "no numbers found" from "passed"
-- were both invisible to fixtures and instant on contact with real output.

So every check reports what it actually saw. On failure it prints the raw tail of
the output that confused it, because the question is always "what did the program
print that I did not expect", and a bare FAIL cannot answer that.

Checks run in dependency order and skip rather than cascade: no point running the
sweep when the build never worked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

PASS = "pass"
FAIL = "fail"
SKIP = "skip"
WARN = "warn"

_ICON = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip", WARN: "warn"}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    evidence: str = ""              # raw output, shown on failure
    elapsed_s: float = 0.0
    data: dict = field(default_factory=dict)

    def line(self) -> str:
        secs = f"{self.elapsed_s:6.1f}s" if self.elapsed_s >= 0.05 else "       "
        return f"[{_ICON[self.status]}] {self.name:<24} {secs}  {self.detail}"


def _tail(text: str, lines: int = 12) -> str:
    rows = [r for r in (text or "").splitlines() if r.strip()]
    return "\n".join(rows[-lines:])


class Smoke:
    """Runs the checks and accumulates results."""

    def __init__(self, settings, *, harness=None, gateway=None, profiler=None) -> None:
        self.settings = settings
        self.harness = harness
        self.gateway = gateway
        self.profiler = profiler
        self.results: list[Check] = []
        self._ok: set[str] = set()

    # -- plumbing -------------------------------------------------------------

    def _run(self, name: str, fn: Callable[[], Check], *, requires: tuple[str, ...] = ()) -> Check:
        missing = [r for r in requires if r not in self._ok]
        if missing:
            check = Check(name, SKIP, f"needs {', '.join(missing)}")
        else:
            started = time.monotonic()
            try:
                check = fn()
            except Exception as exc:                     # a broken check is a failed check
                check = Check(name, FAIL, f"{type(exc).__name__}: {exc}")
            check.name = name
            check.elapsed_s = time.monotonic() - started
        self.results.append(check)
        if check.status in (PASS, WARN):
            self._ok.add(name)
        return check

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.results if c.status == FAIL]

    def report(self) -> dict:
        return {
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail,
                 "elapsed_s": round(c.elapsed_s, 2), **({"data": c.data} if c.data else {})}
                for c in self.results
            ],
            "passed": sum(1 for c in self.results if c.status == PASS),
            "failed": len(self.failed),
            "skipped": sum(1 for c in self.results if c.status == SKIP),
        }

    # -- checks: no network, no GPU -------------------------------------------

    def check_config(self) -> Check:
        s = self.settings
        if not s.api_key:
            return Check("", FAIL, "GITTENSOR_API_KEY is not set")
        if not s.repo.exists():
            return Check("", FAIL, f"repo not found: {s.repo}")
        if not (s.repo / ".git").exists():
            return Check("", FAIL, f"{s.repo} is not a git checkout (worktrees need one)")

        import subprocess
        resolved = subprocess.run(
            ["git", "-C", str(s.repo), "rev-parse", "--verify", s.base_ref],
            capture_output=True, text=True,
        )
        if resolved.returncode != 0:
            # A missing ref means the checkout does not contain the target at
            # all -- the whole round would optimise the wrong tree.
            return Check("", FAIL,
                         f"base ref {s.base_ref!r} not found in {s.repo} "
                         f"(fetch the competition branch first)")
        sha = resolved.stdout.strip()[:8]
        return Check("", PASS,
                     f"repo={s.repo} ref={s.base_ref}@{sha} model={s.model} max_tokens={s.max_tokens}",
                     data={"base_ref": s.base_ref, "base_sha": sha})

    def check_targets(self) -> Check:
        from cuda_agent.source import list_symbols, read_target, Target

        cu_files = sorted((self.settings.repo / "kernels").rglob("*.cu"))
        if not cu_files:
            return Check("", FAIL, f"no .cu files under {self.settings.repo}/kernels")

        targets: list[Target] = []
        for cu in cu_files:
            rel = str(cu.relative_to(self.settings.repo))
            for sym in list_symbols(self.settings.repo, rel, limit=4):
                targets.append(Target(rel, sym))
        if not targets:
            return Check("", FAIL, f"{len(cu_files)} .cu files but no kernels parsed out of them")

        # Extraction must actually narrow: falling back to a whole file silently
        # would blow the prompt budget on the first proposal.
        sample = read_target(self.settings.repo, targets[0])
        return Check("", PASS,
                     f"{len(targets)} targets across {len(cu_files)} files; "
                     f"first extract {len(sample)}B ({targets[0].symbol})",
                     data={"targets": len(targets), "files": len(cu_files)})

    def check_knobs(self) -> Check:
        from cuda_agent.autotune import discover_knobs

        total, kinds = 0, {}
        for cu in sorted((self.settings.repo / "kernels").rglob("*.cu")):
            for knob in discover_knobs(self.settings.repo, str(cu.relative_to(self.settings.repo))):
                total += 1
                kinds[knob.kind] = kinds.get(knob.kind, 0) + 1
        if not total:
            return Check("", WARN, "no tunable constants found -- the sweep will do nothing")
        return Check("", PASS, f"{total} knobs {kinds}", data={"knobs": total, "kinds": kinds})

    # -- checks: gateway ------------------------------------------------------

    def check_gateway_auth(self) -> Check:
        """One minimal call. Confirms the key, the model name and the response shape."""
        completion = self.gateway.complete(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=16,
        )
        u = completion.usage
        shape = []
        if not completion.text.strip():
            shape.append("empty content")
        if not completion.finish_reason:
            shape.append("no finish_reason")
        if not completion.served_uid:
            shape.append("no gittensor.served_uid")
        if u.completion_tokens == 0:
            shape.append("no usage accounting")

        detail = (f"replied {completion.text.strip()[:40]!r} "
                  f"({u.prompt_tokens}+{u.completion_tokens} tok, uid={completion.served_uid or '?'})")
        status = WARN if shape else PASS
        if shape:
            detail += f" -- missing: {', '.join(shape)}"
        return Check("", status, detail,
                     data={"served_uid": completion.served_uid,
                           "finish_reason": completion.finish_reason})

    def check_gateway_context(self) -> Check:
        """A real proposal prompt on a real kernel -- the size the agent will send.

        This is the check that answers the question unit tests cannot: does the
        served context window fit one kernel function plus its history block?
        """
        from cuda_agent.edits import looks_like_attempted_edit, parse_edits
        from cuda_agent.families import FAMILIES
        from cuda_agent.prompts import build_proposal
        from cuda_agent.source import Target, list_symbols

        cu = sorted((self.settings.repo / "kernels").rglob("*.cu"))[0]
        rel = str(cu.relative_to(self.settings.repo))
        symbols = list_symbols(self.settings.repo, rel, limit=1)
        target = Target(rel, symbols[0] if symbols else "")

        from cuda_agent.source import read_target
        messages = build_proposal(
            target=target, family=FAMILIES[0],
            source=read_target(self.settings.repo, target),
            history="No previous attempts on this target.",
            baseline={"4k": 100.0},
        )
        chars = sum(len(m["content"]) for m in messages)

        completion = self.gateway.complete(messages)
        edits = parse_edits(completion.text)
        u = completion.usage

        detail = (f"prompt {chars}B -> {u.prompt_tokens} tok; "
                  f"replied {u.completion_tokens} tok, finish={completion.finish_reason or '?'}")
        data = {"prompt_chars": chars, "prompt_tokens": u.prompt_tokens,
                "edits": len(edits), "truncated": completion.truncated}

        if edits:
            return Check("", PASS, f"{detail}; parsed {len(edits)} edit(s)", data=data)
        if completion.truncated:
            return Check("", WARN, f"{detail}; answer hit the token cap mid-edit",
                         evidence=_tail(completion.text), data=data)
        if looks_like_attempted_edit(completion.text):
            return Check("", WARN, f"{detail}; emitted a malformed edit block",
                         evidence=_tail(completion.text), data=data)
        # Not a failure: the model may reasonably decline. But it is the single
        # most useful thing to eyeball before a real round.
        return Check("", WARN, f"{detail}; answered in prose, no edit",
                     evidence=_tail(completion.text, 8), data=data)

    # -- checks: GPU ----------------------------------------------------------

    def check_build(self) -> Check:
        result = self.harness.build(self.settings.repo)
        if not result.ok:
            return Check("", FAIL, f"build failed ({self.harness.build_cmd})",
                         evidence=_tail(result.log))
        return Check("", PASS, f"built in {result.elapsed_s:.0f}s")

    def check_bench(self) -> Check:
        """The check that catches a parser/format mismatch, which is instant here."""
        result = self.harness.bench(self.settings.repo, quick=True)
        if not result.ok:
            return Check("", FAIL,
                         f"bench produced nothing parseable ({self.harness.quick_bench_cmd})",
                         evidence=result.detail)
        rows = ", ".join(f"{k}={v:.1f}" for k, v in sorted(result.tps.items()))
        return Check("", PASS, f"decode {rows} tok/s", data={"tps": result.tps})

    def check_accuracy(self) -> Check:
        result = self.harness.accuracy(self.settings.repo)
        if not result.ok:
            return Check("", FAIL, f"accuracy gate did not pass ({self.harness.accuracy_cmd})",
                         evidence=result.detail)
        return Check("", PASS, f"top1={result.top1:.3f} kl={result.kl:.4f}",
                     data={"top1": result.top1, "kl": result.kl})

    def check_profile(self) -> Check:
        if not self.profiler:
            return Check("", SKIP, "profiling disabled")
        if not self.profiler.available():
            return Check("", WARN, f"{self.profiler.ncu_bin} not found -- targeting will be round-robin")
        profiles = self.profiler.collect(self.settings.repo)
        if not profiles:
            # Usually admin-restricted counters: a host configuration problem, and
            # the round can still run, so this is a warning rather than a failure.
            return Check("", WARN, self.profiler.last_error or "no kernels profiled")

        from cuda_agent.profile import build_symbol_index, rank_targets
        ranked = rank_targets(profiles, build_symbol_index(self.settings.repo))
        if not ranked:
            return Check("", WARN,
                         f"{len(profiles)} kernels profiled but none mapped to editable source",
                         evidence="\n".join(p.base_name for p in profiles[:8]))
        top, prof = ranked[0]
        return Check("", PASS,
                     f"{len(ranked)} hot kernels; top {prof.base_name} "
                     f"({prof.time_share * 100:.0f}%, {prof.verdict})",
                     data={"hot": len(ranked), "top": prof.base_name, "verdict": prof.verdict})

    # -- checks: one full iteration of each phase -----------------------------

    def check_sweep_iteration(self, baseline: dict) -> Check:
        """One deterministic sweep candidate through the whole gate. No tokens."""
        from cuda_agent.autotune import Autotuner, discover_knobs
        from cuda_agent.ledger import Ledger
        from cuda_agent.worktree import Frontier

        paths = [
            str(cu.relative_to(self.settings.repo))
            for cu in sorted((self.settings.repo / "kernels").rglob("*.cu"))
        ]
        first = next(
            (p for p in paths if discover_knobs(self.settings.repo, p)), None
        )
        if not first:
            return Check("", SKIP, "no knobs to sweep")

        frontier = Frontier(self.settings.repo, self.settings.worktree_root)
        ledger = Ledger(self.settings.out_dir / "smoke-ledger.jsonl")
        tuner = Autotuner(frontier, self.harness, ledger,
                          allowed=self.settings.allowed_paths,
                          denied=self.settings.denied_paths)
        try:
            _, evals = tuner.run([first], baseline, max_evals=1)
        finally:
            frontier.cleanup()

        if not evals:
            return Check("", SKIP, "sweep produced no candidate")
        attempt = ledger.attempts[-1]
        return Check("", PASS,
                     f"1 candidate through build+accuracy+bench -> {attempt.verdict}",
                     data={"verdict": attempt.verdict, "detail": attempt.detail})


def run(settings, *, harness, gateway, profiler=None, skip_gpu: bool = False) -> Smoke:
    """Run every check in dependency order."""
    smoke = Smoke(settings, harness=harness, gateway=gateway, profiler=profiler)

    smoke._run("config", smoke.check_config)
    smoke._run("targets", smoke.check_targets, requires=("config",))
    smoke._run("knobs", smoke.check_knobs, requires=("config",))
    smoke._run("gateway-auth", smoke.check_gateway_auth, requires=("config",))
    smoke._run("gateway-context", smoke.check_gateway_context,
               requires=("config", "targets", "gateway-auth"))

    if skip_gpu:
        for name in ("build", "bench", "accuracy", "profile", "sweep-iteration"):
            smoke.results.append(Check(name, SKIP, "--smoke-no-gpu"))
        return smoke

    smoke._run("build", smoke.check_build, requires=("config",))
    bench = smoke._run("bench", smoke.check_bench, requires=("build",))
    smoke._run("accuracy", smoke.check_accuracy, requires=("build",))
    smoke._run("profile", smoke.check_profile, requires=("build",))
    smoke._run("sweep-iteration",
               lambda: smoke.check_sweep_iteration(bench.data.get("tps", {})),
               requires=("bench", "accuracy", "knobs"))
    return smoke


def render(smoke: Smoke) -> str:
    """Human-readable report, with raw evidence for anything that failed."""
    lines = [c.line() for c in smoke.results]
    r = smoke.report()
    lines.append("")
    lines.append(f"{r['passed']} passed, {r['failed']} failed, {r['skipped']} skipped")

    for check in smoke.results:
        if check.evidence and check.status in (FAIL, WARN):
            lines += ["", f"--- {check.name}: what it actually printed ---", check.evidence]
    return "\n".join(lines)
