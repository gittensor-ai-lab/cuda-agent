"""End-to-end loop test against a fake gateway and a simulated GPU.

The fake harness models a real tuning knob: a tile constant in the source maps to
a throughput, one value fails to build, and one is fast but wrong. That is enough
to check the only property that really matters -- the loop keeps measured wins
and discards everything else, whatever the model claimed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from cuda_agent.backend import Completion, Usage
from cuda_agent.config import Settings
from cuda_agent.families import Family, FamilyBandit
from cuda_agent.harness import AccuracyResult, BenchResult, BuildResult
from cuda_agent.ledger import ACCEPTED, BUILD_FAILED, MALFORMED, NOT_FASTER, Ledger
from cuda_agent.loop import Optimizer
from cuda_agent.source import Target
from cuda_agent.worktree import Frontier

KERNEL = """\
// toy kernel
__global__ void hot_kernel(const float* a, float* b, int n) {
    constexpr int TILE_N = 64;
    for (int i = 0; i < n; i += TILE_N) b[i] = a[i];
}
"""

# tile value -> (builds, correct, tok/s)
BEHAVIOUR = {
    64: (True, True, 100.0),
    128: (True, True, 120.0),      # the real win
    256: (False, True, 0.0),       # does not compile
    512: (True, False, 999.0),     # fast and wrong -- must be rejected
    32: (True, True, 90.0),        # slower
}


def edit_block(old: int, new: int) -> str:
    return (
        '<edit path="kernels/hot.cu">\n'
        "<<<<<<< SEARCH\n"
        f"    constexpr int TILE_N = {old};\n"
        "=======\n"
        f"    constexpr int TILE_N = {new};\n"
        ">>>>>>> REPLACE\n"
        "</edit>\n"
    )


class FakeGateway:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.usage = Usage()
        self.calls = 0

    def complete(self, messages, *, max_tokens=None) -> Completion:
        self.calls += 1
        text = self.answers.pop(0) if self.answers else "nothing to do here"
        self.usage.completion_tokens += 50
        self.usage.requests += 1
        return Completion(text=text, usage=Usage(completion_tokens=50), finish_reason="stop")


class FakeHarness:
    """Reads the tile constant out of the worktree and looks up its behaviour."""

    def __init__(self) -> None:
        self.builds = 0

    @staticmethod
    def _tile(path: Path) -> int:
        text = (path / "kernels" / "hot.cu").read_text()
        return int(re.search(r"TILE_N = (\d+)", text).group(1))

    def build(self, path: Path) -> BuildResult:
        self.builds += 1
        ok, _, _ = BEHAVIOUR[self._tile(path)]
        return BuildResult(ok=ok, log="error: too much shared memory" if not ok else "")

    def accuracy(self, path: Path) -> AccuracyResult:
        _, correct, _ = BEHAVIOUR[self._tile(path)]
        return AccuracyResult(ok=correct, top1=1.0 if correct else 0.4, kl=0.0 if correct else 0.9,
                              detail="" if correct else "top1=0.400 below bar")

    def bench(self, path: Path, *, quick: bool = False) -> BenchResult:
        _, _, tps = BEHAVIOUR[self._tile(path)]
        return BenchResult(ok=True, tps={"4k": tps})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "hot.cu").write_text(KERNEL)
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "-m", "base"],
        check=True,
    )
    return root


def make_optimizer(repo: Path, tmp_path: Path, answers: list[str], *, budget: float = 1e9):
    settings = Settings(
        api_key="test",
        repo=repo,
        out_dir=tmp_path / "out",
        worktree_root=tmp_path / "wt",
        wall_clock_s=budget,
    )
    gateway = FakeGateway(answers)
    harness = FakeHarness()
    frontier = Frontier(repo, settings.worktree_root)
    ticks = iter(range(10_000))
    opt = Optimizer(
        settings, gateway, harness, frontier,
        targets=[Target("kernels/hot.cu", "hot_kernel")],
        ledger=Ledger(settings.out_dir / "ledger.jsonl"),
        bandit=FamilyBandit(families=(Family("tile_shape", "widen the tile"),)),
        clock=lambda: next(ticks),
    )
    return opt, gateway, harness, frontier


def test_accepts_a_measured_win_and_advances_the_frontier(repo, tmp_path):
    opt, _, _, frontier = make_optimizer(repo, tmp_path, [edit_block(64, 128)], budget=3)
    report = opt.run()

    assert report.accepted == 1
    assert opt.ledger.attempts[0].verdict == ACCEPTED
    assert report.total_speedup == pytest.approx(0.20, abs=1e-6)
    assert frontier.sha != frontier.base_sha
    assert "TILE_N = 128" in frontier.diff()


def test_rejects_a_build_failure(repo, tmp_path):
    opt, _, _, frontier = make_optimizer(repo, tmp_path, [edit_block(64, 256)], budget=3)
    report = opt.run()

    assert report.accepted == 0
    assert opt.ledger.attempts[0].verdict == BUILD_FAILED
    assert frontier.sha == frontier.base_sha


def test_rejects_fast_but_wrong(repo, tmp_path):
    """The whole point of the accuracy gate: 999 tok/s must not win."""
    opt, _, _, frontier = make_optimizer(repo, tmp_path, [edit_block(64, 512)], budget=3)
    report = opt.run()

    assert report.accepted == 0
    assert opt.ledger.attempts[0].verdict == "accuracy-failed"
    assert frontier.sha == frontier.base_sha


def test_rejects_a_regression(repo, tmp_path):
    opt, _, _, frontier = make_optimizer(repo, tmp_path, [edit_block(64, 32)], budget=3)
    report = opt.run()

    assert opt.ledger.attempts[0].verdict == NOT_FASTER
    assert frontier.sha == frontier.base_sha


def test_prose_answer_is_recorded_as_malformed_without_a_build(repo, tmp_path):
    opt, _, harness, _ = make_optimizer(repo, tmp_path, ["You could try a wider tile."], budget=3)
    opt.run()

    assert opt.ledger.attempts[0].verdict == MALFORMED
    assert harness.builds == 0, "a prose answer must never reach the compiler"


def test_truncated_edit_gets_one_repair_attempt(repo, tmp_path):
    truncated = '<edit path="kernels/hot.cu">\n<<<<<<< SEARCH\n    constexpr int TILE_N = 64;\n'
    opt, gateway, _, frontier = make_optimizer(
        repo, tmp_path, [truncated, edit_block(64, 128)], budget=3
    )
    report = opt.run()

    assert gateway.calls == 2, "the malformed answer should be retried exactly once"
    assert report.accepted == 1
    assert "TILE_N = 128" in frontier.diff()


def test_stale_candidate_cannot_be_promoted(repo, tmp_path):
    """Two candidates cut from different baselines must not silently stack."""
    from cuda_agent.worktree import GitError

    frontier = Frontier(repo, tmp_path / "wt")
    stale = frontier.checkout()
    (stale.path / "kernels" / "hot.cu").write_text(KERNEL.replace("64", "128"))

    fresh = frontier.checkout()
    (fresh.path / "kernels" / "hot.cu").write_text(KERNEL.replace("64", "256"))
    frontier.promote(fresh, "first win")

    with pytest.raises(GitError, match="re-evaluate it against the current frontier"):
        frontier.promote(stale, "stale win")


def test_wall_clock_budget_stops_the_loop(repo, tmp_path):
    answers = [edit_block(64, 128)] + ["prose"] * 50
    opt, _, _, _ = make_optimizer(repo, tmp_path, answers, budget=5)
    report = opt.run()

    assert report.attempts <= 5


# --- sweep + proposer composition ------------------------------------------

def test_sweep_runs_first_and_the_proposer_builds_on_it(repo, tmp_path):
    """The two phases must compose without double-counting the same gain.

    The sweep finds TILE_N 64 -> 128 (100 -> 120 tok/s) with no model call. The
    proposer then finds 128 -> 256... which does not build. The reported total
    must be the sweep's 20%, measured once.
    """
    from cuda_agent.autotune import Autotuner
    from cuda_agent.ledger import BUILD_FAILED

    opt, gateway, harness, frontier = make_optimizer(
        repo, tmp_path, [edit_block(128, 256)], budget=40
    )
    opt.autotuner = Autotuner(
        frontier, harness, opt.ledger,
        allowed=opt.settings.allowed_paths, denied=opt.settings.denied_paths,
    )
    opt.autotune_evals = 4
    opt.autotune_fraction = 0.5

    report = opt.run()

    families = {a.family for a in opt.ledger.attempts}
    assert "autotune" in families, "the sweep must run"
    assert report.autotune_evals > 0

    # The sweep banked the win without the model.
    swept = [a for a in opt.ledger.attempts if a.family == "autotune" and a.verdict == ACCEPTED]
    assert len(swept) == 1
    assert "TILE_N = 128" in frontier.diff()

    # The proposer's follow-up was measured against the *swept* frontier.
    proposed = [a for a in opt.ledger.attempts if a.family != "autotune"]
    assert proposed and proposed[0].verdict == BUILD_FAILED

    assert report.total_speedup == pytest.approx(0.20, abs=1e-6)
    assert report.by_family["autotune"]["accepted"] == 1


def test_sweep_is_skipped_when_no_autotuner_is_configured(repo, tmp_path):
    opt, _, _, _ = make_optimizer(repo, tmp_path, [edit_block(64, 128)], budget=3)
    report = opt.run()
    assert report.autotune_evals == 0
    assert "autotune" not in {a.family for a in opt.ledger.attempts}


# --- profile-directed targeting --------------------------------------------

class FakeProfiler:
    """Stands in for ncu: reports one hot, memory-bound kernel."""

    def __init__(self, profiles=None, error=""):
        from cuda_agent.profile import KernelProfile
        self.calls = 0
        self.last_error = error
        self._profiles = profiles if profiles is not None else [
            KernelProfile(
                name="hot_kernel(const float*, float*, int)",
                time_ns=1000, time_share=0.95, launches=4,
                memory_pct=88, compute_pct=15, occupancy_pct=70,
                stalls={"long_scoreboard": 5.1},
            )
        ]

    def collect(self, path):
        self.calls += 1
        return self._profiles


def test_profile_directs_targeting_and_reaches_the_prompt(repo, tmp_path):
    opt, gateway, _, _ = make_optimizer(repo, tmp_path, [edit_block(64, 128)], budget=3)
    opt.profiler = FakeProfiler()

    captured: list = []
    original = gateway.complete

    def spy(messages, **kw):
        captured.append(messages)
        return original(messages, **kw)

    gateway.complete = spy
    report = opt.run()

    assert report.profiled is True
    assert opt.profiler.calls == 1
    # The hot kernel became the target, replacing the configured round-robin list.
    assert all(t.symbol == "hot_kernel" for t in opt.targets)
    # ...and the bottleneck diagnosis reached the proposer.
    prompt = captured[0][1]["content"]
    assert "memory-bound" in prompt
    assert "long_scoreboard" in prompt


def test_profiler_seeds_the_bandit_toward_memory_families(repo, tmp_path):
    opt, _, _, _ = make_optimizer(repo, tmp_path, ["prose"], budget=2)
    opt.bandit = __import__("cuda_agent.families", fromlist=["FamilyBandit"]).FamilyBandit()
    opt.profiler = FakeProfiler()
    opt.run()

    # A memory-bound kernel stalled on long_scoreboard should be probed with a
    # memory-oriented family first, not whatever happens to be listed first.
    first = opt.ledger.attempts[0].family
    assert first in ("vector_width", "memory_hint", "smem_layout", "fusion")


def test_missing_profiler_degrades_to_round_robin(repo, tmp_path):
    opt, _, _, frontier = make_optimizer(repo, tmp_path, [edit_block(64, 128)], budget=3)
    opt.profiler = FakeProfiler(profiles=[], error="profiling permission denied")
    report = opt.run()

    assert report.profiled is False
    assert report.profile_error == "profiling permission denied"
    # The round still ran and still banked the win.
    assert report.accepted == 1
    assert "TILE_N = 128" in frontier.diff()


def test_reprofiles_after_the_frontier_moves(repo, tmp_path):
    """Fixing the top kernel moves the bottleneck; the ranking must follow."""
    answers = [edit_block(64, 128), "prose", "prose"]
    opt, _, _, _ = make_optimizer(repo, tmp_path, answers, budget=6)
    opt.profiler = FakeProfiler()
    opt.reprofile_after = 1
    opt.run()

    assert opt.profiler.calls >= 2, "an accepted change should trigger a re-profile"


# --- cross-round memory ----------------------------------------------------

def test_known_bad_edit_is_skipped_without_a_build(repo, tmp_path):
    """The expensive win: a change rejected last round costs no build this round."""
    from cuda_agent.ledger import KNOWN_BAD
    from cuda_agent.memory import Memory

    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_rejected("kernels/hot.cu",
                      "    constexpr int TILE_N = 64;",
                      "    constexpr int TILE_N = 256;",
                      verdict="build-failed", detail="out of shared memory")
    mem.start_round()

    opt, _, harness, frontier = make_optimizer(repo, tmp_path, [edit_block(64, 256)], budget=3)
    opt.memory = mem
    report = opt.run()

    assert opt.ledger.attempts[0].verdict == KNOWN_BAD
    assert harness.builds == 0, "the candidate must never reach the compiler"
    assert report.skipped_known_bad == 1
    assert frontier.sha == frontier.base_sha


def test_a_win_and_its_family_are_remembered(repo, tmp_path):
    from cuda_agent.memory import Memory

    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    opt, _, _, _ = make_optimizer(repo, tmp_path, [edit_block(64, 128)], budget=3)
    opt.memory = mem
    report = opt.run()
    mem.save()

    assert report.accepted == 1
    carried = Memory.load(tmp_path / "mem.json")
    assert carried.wins and carried.wins[0]["path"] == "kernels/hot.cu"
    kernel = "kernels/hot.cu::hot_kernel"
    assert carried.families[kernel]["tile_shape"].accepted == 1
    assert carried.preferred_families(kernel)[0] == "tile_shape"


def test_a_rejection_is_remembered_for_the_next_round(repo, tmp_path):
    from cuda_agent.memory import Memory

    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    opt, _, _, _ = make_optimizer(repo, tmp_path, [edit_block(64, 32)], budget=3)
    opt.memory = mem
    opt.run()
    mem.save()

    carried = Memory.load(tmp_path / "mem.json")
    carried.start_round()
    assert carried.known_bad(
        repo, "kernels/hot.cu",
        "    constexpr int TILE_N = 64;", "    constexpr int TILE_N = 32;",
    ) == "not-faster"


def test_profile_evidence_leads_memory_follows(repo, tmp_path):
    """Today's profile outranks yesterday's outcomes, but does not erase them."""
    from cuda_agent.memory import Memory

    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_family("kernels/hot.cu::hot_kernel", "split_k", accepted=True, reward=1.0)

    opt, _, _, _ = make_optimizer(repo, tmp_path, ["prose"], budget=2)
    from cuda_agent.families import FamilyBandit
    opt.bandit = FamilyBandit()
    opt.memory = mem
    opt.profiler = FakeProfiler()          # memory-bound verdict
    opt.run()

    prefs = opt.bandit._preference["kernels/hot.cu::hot_kernel"]
    assert prefs[0] in ("vector_width", "memory_hint", "smem_layout", "fusion")
    assert "split_k" in prefs, "the remembered winner is appended, not dropped"


# --- in-place candidates (build incrementality) ----------------------------

def test_in_place_keeps_the_build_directory(repo, tmp_path):
    """A fresh worktree has no build/, so every candidate would pay a full
    rebuild -- ~20 min vs ~78 s incremental, measured on an RTX 5090."""
    build = repo / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text("cached")

    frontier = Frontier(repo, tmp_path / "wt")
    cand = frontier.checkout()
    assert cand.path == repo, "the candidate is the repo itself"
    assert (cand.path / "build" / "CMakeCache.txt").exists()

    (repo / "kernels" / "hot.cu").write_text(KERNEL.replace("64", "128"))
    frontier.discard(cand)
    assert (build / "CMakeCache.txt").exists(), "rollback must not wipe the build dir"


def test_in_place_rollback_restores_the_source(repo, tmp_path):
    frontier = Frontier(repo, tmp_path / "wt")
    cand = frontier.checkout()
    (repo / "kernels" / "hot.cu").write_text(KERNEL.replace("64", "999"))
    frontier.discard(cand)
    assert "TILE_N = 64" in (repo / "kernels" / "hot.cu").read_text()


def test_in_place_rollback_is_scoped_to_editable_paths(repo, tmp_path):
    """A bare `git checkout -- .` would delete a downloaded model or build log."""
    stray = repo / "models"
    stray.mkdir()
    (stray / "weights.gguf").write_text("expensive to re-download")

    frontier = Frontier(repo, tmp_path / "wt")
    cand = frontier.checkout()
    (repo / "kernels" / "hot.cu").write_text(KERNEL.replace("64", "128"))
    frontier.discard(cand)

    assert (stray / "weights.gguf").exists(), "rollback must not touch untracked data"


def test_in_place_promote_advances_the_branch_and_diff(repo, tmp_path):
    frontier = Frontier(repo, tmp_path / "wt")
    cand = frontier.checkout()
    (repo / "kernels" / "hot.cu").write_text(KERNEL.replace("64", "128"))
    sha = frontier.promote(cand, "perf: widen tile")
    assert sha != frontier.base_sha
    assert "TILE_N = 128" in frontier.diff()


def test_worktree_mode_still_available(repo, tmp_path):
    frontier = Frontier(repo, tmp_path / "wt", in_place=False)
    cand = frontier.checkout()
    assert cand.path != repo
    frontier.discard(cand)
