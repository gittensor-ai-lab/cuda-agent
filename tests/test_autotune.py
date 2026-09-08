from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cuda_agent.autotune import Autotuner, Knob, candidate_values, discover_knobs
from cuda_agent.harness import AccuracyResult, BenchResult, BuildResult
from cuda_agent.ledger import ACCEPTED, Ledger
from cuda_agent.worktree import Frontier

ALLOWED = ("kernels/",)
DENIED = ("bench/scripts/",)

SRC = """\
#define BLOCK_SIZE 256
static constexpr int TILE_N = 64;
const int NWARPS = 8;
constexpr int MAGIC_OFFSET = 7;

__global__ void __launch_bounds__(256, 4) hot(const float* a, float* b, int n) {
#pragma unroll 4
    for (int i = 0; i < n; ++i) b[i] = a[i];
}
"""


def test_discovers_each_knob_kind(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.cu").write_text(SRC)
    knobs = discover_knobs(tmp_path, "kernels/k.cu")
    by_name = {k.name or k.kind: k for k in knobs}

    assert by_name["TILE_N"].value == 64 and by_name["TILE_N"].kind == "tile"
    assert by_name["NWARPS"].kind == "tile"          # geometry by name
    assert by_name["BLOCK_SIZE"].kind == "define"
    assert by_name["unroll"].value == 4
    assert by_name["launch_bounds"].value == 256
    # An arbitrary constant is kept but ranked last.
    assert by_name["MAGIC_OFFSET"].priority > by_name["TILE_N"].priority


def test_geometry_knobs_are_swept_before_magic_numbers(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.cu").write_text(SRC)
    order = [k.name or k.kind for k in discover_knobs(tmp_path, "kernels/k.cu")]
    assert order.index("TILE_N") < order.index("MAGIC_OFFSET")


def test_ambiguous_anchors_are_skipped(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.cu").write_text("#pragma unroll 4\n#pragma unroll 4\n")
    assert discover_knobs(tmp_path, "kernels/k.cu") == []


def test_candidate_values_walk_outward_from_a_power_of_two():
    knob = Knob("k.cu", "tile", "TILE_N", 64, "x", "", "")
    assert candidate_values(knob) == [32, 128, 16, 256]


def test_candidate_values_for_unroll_and_launch_bounds():
    unroll = Knob("k.cu", "unroll", "unroll", 4, "x", "", "")
    assert 4 not in candidate_values(unroll)
    lb = Knob("k.cu", "launch_bounds", "launch_bounds", 256, "x", "", "")
    assert candidate_values(lb)[:2] == [128, 512]


def test_edit_round_trips_through_the_anchor(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.cu").write_text(SRC)
    knob = next(k for k in discover_knobs(tmp_path, "kernels/k.cu") if k.name == "TILE_N")
    edit = knob.edit(128)
    assert edit.search in SRC
    assert edit.replace == "static constexpr int TILE_N = 128;"


# --- end-to-end sweep ------------------------------------------------------

TUNABLE = "constexpr int TILE_N = 64;\n"
# tile -> (builds, correct, tok/s). 128 is the win; 256 will not compile.
BEHAVIOUR = {16: 80.0, 32: 90.0, 64: 100.0, 128: 130.0, 256: None}


class SweepHarness:
    def __init__(self) -> None:
        self.evals = 0

    @staticmethod
    def _tile(path: Path) -> int:
        import re
        return int(re.search(r"TILE_N = (\d+)", (path / "kernels" / "k.cu").read_text()).group(1))

    def build(self, path: Path) -> BuildResult:
        self.evals += 1
        ok = BEHAVIOUR.get(self._tile(path)) is not None
        return BuildResult(ok=ok, log="" if ok else "error: out of shared memory")

    def accuracy(self, path: Path) -> AccuracyResult:
        return AccuracyResult(ok=True, top1=1.0, kl=0.0)

    def bench(self, path: Path, *, quick: bool = False) -> BenchResult:
        return BenchResult(ok=True, tps={"4k": BEHAVIOUR[self._tile(path)]})


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "k.cu").write_text(TUNABLE)
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "base"], check=True)
    return root


def _tuner(repo, tmp_path):
    frontier = Frontier(repo, tmp_path / "wt")
    ledger = Ledger(tmp_path / "out" / "ledger.jsonl")
    return Autotuner(frontier, SweepHarness(), ledger, allowed=ALLOWED, denied=DENIED), frontier, ledger


def test_sweep_finds_the_win_with_no_model_calls(repo, tmp_path):
    tuner, frontier, ledger = _tuner(repo, tmp_path)
    baseline, evals = tuner.run(["kernels/k.cu"], {"4k": 100.0}, max_evals=10)

    assert baseline["4k"] == 130.0
    assert "TILE_N = 128" in frontier.diff()
    assert any(a.verdict == ACCEPTED for a in ledger.attempts)
    assert evals <= 10


def test_sweep_stops_at_the_evaluation_budget(repo, tmp_path):
    tuner, _, ledger = _tuner(repo, tmp_path)
    _, evals = tuner.run(["kernels/k.cu"], {"4k": 100.0}, max_evals=1)
    assert evals == 1
    assert len(ledger.attempts) == 1


def test_sweep_honours_an_external_stop(repo, tmp_path):
    tuner, _, _ = _tuner(repo, tmp_path)
    _, evals = tuner.run(["kernels/k.cu"], {"4k": 100.0}, max_evals=10, should_stop=lambda: True)
    assert evals == 0


def test_sweep_moves_on_after_a_knob_pays(repo, tmp_path):
    """Once 128 wins, the sweep must not keep pushing that same knob."""
    tuner, _, ledger = _tuner(repo, tmp_path)
    tuner.run(["kernels/k.cu"], {"4k": 100.0}, max_evals=10)
    tried = [a.detail for a in ledger.attempts]
    assert sum(1 for d in tried if "-> 128" in d) == 1


def test_capacity_constants_are_not_swept(tmp_path):
    """Shrinking a buffer bound can pass a short check and break at scale."""
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.cu").write_text(
        "constexpr int kMaxDevices = 16;\n"
        "constexpr int MAX_SEQ_LEN = 4096;\n"
        "constexpr int kSlotCount = 32;\n"
        "constexpr int TILE_N = 64;\n"
    )
    names = {k.name for k in discover_knobs(tmp_path, "kernels/k.cu")}
    assert names == {"TILE_N"}


def test_geometry_bounds_survive_the_capacity_filter(tmp_path):
    """ARGMAX_ROWS_MAX is a tile bound, not a buffer limit."""
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.cu").write_text("constexpr int ARGMAX_ROWS_MAX = 16;\n")
    knobs = discover_knobs(tmp_path, "kernels/k.cu")
    assert [k.name for k in knobs] == ["ARGMAX_ROWS_MAX"]
