"""Arena entrypoint.

Contract with the arena runner:

  in   $CUDA_AGENT_REPO      a checkout at the round's pinned base commit
       $GITTENSOR_API_KEY    scoped, metered, per-round
  out  $CUDA_AGENT_OUT/patch.diff     the submission
       $CUDA_AGENT_OUT/report.json    evidence (never the score)
       $CUDA_AGENT_OUT/ledger.jsonl   every attempt, written as it happens

The patch is written on every exit path, including SIGTERM at the budget
deadline, so a killed agent still submits its best verified state.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict
from pathlib import Path

from cuda_agent.autotune import Autotuner
from cuda_agent.backend import Gateway
from cuda_agent.config import get_settings
from cuda_agent.ledger import Ledger
from cuda_agent.loop import Optimizer, RunReport
from cuda_agent.memory import Memory
from cuda_agent.profile import Profiler
from cuda_agent.runner import SubprocessHarness
from cuda_agent import smoke as smoke_mod
from cuda_agent.source import Target, list_symbols
from cuda_agent.worktree import Frontier


def load_targets(repo: Path, spec: Path | None) -> list[Target]:
    """Targets from a JSON file, or discovered from the kernel tree."""
    if spec and spec.exists():
        raw = json.loads(spec.read_text())
        return [Target(t["path"], t.get("symbol", "")) for t in raw]

    targets: list[Target] = []
    for cu in sorted((repo / "kernels").rglob("*.cu")):
        rel = str(cu.relative_to(repo))
        for sym in list_symbols(repo, rel, limit=4):
            targets.append(Target(rel, sym))
    return targets


def write_outputs(out: Path, frontier: Frontier, report: RunReport | None, error: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        (out / "patch.diff").write_text(frontier.diff(), encoding="utf-8")
    except Exception as exc:                       # never let reporting mask the patch
        (out / "patch.diff").write_text("", encoding="utf-8")
        error = error or f"could not render diff: {exc}"

    payload = asdict(report) if report else {}
    if error:
        payload["error"] = error
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cuda-agent")
    parser.add_argument("--targets", type=Path, help="JSON list of {path, symbol}")
    parser.add_argument("--wall-clock", type=float, help="override the self-imposed budget (s)")
    parser.add_argument("--base-ref", help="branch/commit to optimise (e.g. feat/spark-x25-4b)")
    parser.add_argument("--dry-run", action="store_true", help="resolve targets and exit")
    parser.add_argument("--autotune-evals", type=int, default=40,
                        help="max deterministic sweep evaluations before the proposer runs (0 disables)")
    parser.add_argument("--autotune-fraction", type=float, default=0.4,
                        help="fraction of the wall clock the sweep may use")
    parser.add_argument("--list-knobs", action="store_true",
                        help="list the constants the sweep would try, and exit")
    parser.add_argument("--smoke", action="store_true",
                        help="run preflight checks against this box and gateway, then exit")
    parser.add_argument("--smoke-no-gpu", action="store_true",
                        help="preflight without build/bench/profile (config + gateway only)")
    parser.add_argument("--no-memory", action="store_true",
                        help="start cold; do not read or write cross-round memory")
    parser.add_argument("--memory", type=Path, help="override the cross-round memory path")
    parser.add_argument("--no-profile", action="store_true",
                        help="skip ncu profiling and walk targets round-robin")
    parser.add_argument("--profile-launches", type=int, default=12,
                        help="kernel launches ncu replays per profiling pass")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.wall_clock:
        settings.wall_clock_s = args.wall_clock
    if args.base_ref:
        settings.base_ref = args.base_ref
    settings.validate()

    targets = load_targets(settings.repo, args.targets)
    if not targets:
        print("no targets found", file=sys.stderr)
        return 2
    if args.dry_run:
        for t in targets:
            print(t.key)
        return 0

    if args.list_knobs:
        from cuda_agent.autotune import candidate_values, discover_knobs
        for path in dict.fromkeys(t.path for t in targets):
            for knob in discover_knobs(settings.repo, path):
                print(f"{knob.key:64s} {knob.kind:14s} -> {candidate_values(knob)}")
        return 0

    if args.smoke or args.smoke_no_gpu:
        gateway = Gateway(settings)
        harness = SubprocessHarness(repo=settings.repo)
        profiler = None if args.no_profile else Profiler(
            settings.repo, harness.quick_bench_cmd, launch_count=args.profile_launches
        )
        try:
            result = smoke_mod.run(settings, harness=harness, gateway=gateway,
                                   profiler=profiler, skip_gpu=args.smoke_no_gpu)
        finally:
            gateway.close()
        print(smoke_mod.render(result))
        (settings.out_dir).mkdir(parents=True, exist_ok=True)
        (settings.out_dir / "smoke.json").write_text(
            json.dumps(result.report(), indent=2), encoding="utf-8"
        )
        return 1 if result.failed else 0

    frontier = Frontier(settings.repo, settings.worktree_root, base_ref=settings.base_ref)
    report: RunReport | None = None

    memory = None
    if not args.no_memory:
        memory = Memory.load(args.memory or settings.memory_path)
        round_no = memory.start_round()
        print(f"round {round_no}: {json.dumps(memory.summary())}", file=sys.stderr)

    # The arena sends SIGTERM at the deadline with a short grace period. Whatever
    # has been verified so far is already committed on the frontier branch, so
    # the handler only has to render it.
    def on_term(signum, frame):                    # noqa: ARG001
        # Memory is saved here too: a round killed at the deadline still learned
        # something, and throwing that away would make the deadline the one place
        # the agent forgets everything.
        if memory:
            try:
                memory.save()
            except OSError:
                pass
        write_outputs(settings.out_dir, frontier, report, error="terminated at deadline")
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)

    gateway = Gateway(settings)
    harness = SubprocessHarness(repo=settings.repo)
    ledger = Ledger(settings.out_dir / "ledger.jsonl")
    autotuner = None
    if args.autotune_evals > 0:
        # Shares the ledger and the frontier with the proposer, so a gain the
        # sweep banks is the baseline the proposer is then measured against.
        autotuner = Autotuner(
            frontier, harness, ledger,
            allowed=settings.allowed_paths, denied=settings.denied_paths,
            memory=memory,
        )
    profiler = None
    if not args.no_profile:
        # Profiling is best-effort: on a box where counters are admin-restricted
        # the collect() returns nothing and the loop falls back to round-robin,
        # with the reason recorded in the report rather than failing the round.
        profiler = Profiler(
            settings.repo, harness.quick_bench_cmd, launch_count=args.profile_launches
        )
        if not profiler.available():
            print(f"note: {profiler.ncu_bin} not found; targeting round-robin", file=sys.stderr)

    optimizer = Optimizer(
        settings, gateway, harness, frontier, targets,
        ledger=ledger,
        autotuner=autotuner,
        autotune_evals=args.autotune_evals,
        autotune_fraction=args.autotune_fraction,
        profiler=profiler,
        memory=memory,
    )

    error = ""
    try:
        report = optimizer.run()
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:                       # a crashed agent still submits
        error = f"{type(exc).__name__}: {exc}"
    finally:
        gateway.close()
        if memory:
            try:
                memory.save()
            except OSError as exc:
                print(f"warning: could not persist memory: {exc}", file=sys.stderr)
        write_outputs(settings.out_dir, frontier, report, error)
        frontier.cleanup()

    if report:
        print(json.dumps({
            "accepted": report.accepted,
            "attempts": report.attempts,
            "autotune_evals": report.autotune_evals,
            "by_family": report.by_family,
            "profiled": report.profiled,
            "profile_error": report.profile_error,
            "skipped_known_bad": report.skipped_known_bad,
            "memory": report.memory,
            "speedup": round(report.total_speedup, 4),
            "tokens": report.completion_tokens,
            "est_cost_usd": report.est_cost_usd,
            "rate_limited": report.rate_limited,
        }, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
