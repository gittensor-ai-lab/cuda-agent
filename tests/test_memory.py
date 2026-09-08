from __future__ import annotations

from pathlib import Path

from cuda_agent.families import Family, FamilyBandit
from cuda_agent.memory import Memory, edit_key

SRC = "constexpr int TILE_N = 64;\n"


def _repo(tmp_path: Path, body: str = SRC) -> Path:
    (tmp_path / "kernels").mkdir(exist_ok=True)
    (tmp_path / "kernels" / "k.cu").write_text(body)
    return tmp_path


def test_round_trips_through_disk(tmp_path):
    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_family("k.cu::hot", "vector_width", accepted=True, reward=0.8)
    mem.note_swept("k.cu::TILE_N", 128)
    mem.note_win("k.cu", "vector_width", "float4 loads", 0.08)
    mem.save()

    again = Memory.load(tmp_path / "mem.json")
    assert again.round_no == 1
    assert again.families["k.cu::hot"]["vector_width"].reward == 0.8
    assert again.swept("k.cu::TILE_N") == {128}
    assert again.wins[0]["speedup"] == 0.08


def test_corrupt_store_starts_clean_instead_of_failing(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{not json")
    mem = Memory.load(path)
    assert mem.round_no == 0 and mem.families == {}


def test_schema_mismatch_is_discarded(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text('{"schema": 999, "round_no": 42}')
    assert Memory.load(path).round_no == 0


def test_save_is_atomic(tmp_path):
    """A killed agent must not leave a truncated store behind."""
    path = tmp_path / "mem.json"
    mem = Memory.load(path)
    mem.start_round()
    mem.save()
    assert not path.with_suffix(".tmp").exists()
    assert Memory.load(path).round_no == 1


# --- family priors ---------------------------------------------------------

def test_seeds_the_bandit_with_what_paid_before(tmp_path):
    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_family("k.cu::hot", "unroll", accepted=False, reward=0.0)
    mem.note_family("k.cu::hot", "vector_width", accepted=True, reward=0.9)
    mem.note_family("k.cu::hot", "fusion", accepted=True, reward=0.3)

    assert mem.preferred_families("k.cu::hot")[0] == "vector_width"

    bandit = FamilyBandit(families=(
        Family("unroll", ""), Family("fusion", ""), Family("vector_width", ""),
    ))
    assert mem.seed(bandit, ["k.cu::hot"]) == 1
    # Exploration order follows memory, not declaration order.
    assert bandit.select("k.cu::hot").name == "vector_width"


def test_priors_only_reorder_exploration(tmp_path):
    """Once measured, this round's evidence must win over last round's."""
    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_family("k::hot", "vector_width", accepted=True, reward=1.0)

    bandit = FamilyBandit(families=(Family("vector_width", ""), Family("unroll", "")))
    mem.seed(bandit, ["k::hot"])

    # The remembered favourite is tried first, and fails today.
    assert bandit.select("k::hot").name == "vector_width"
    bandit.update("k::hot", "vector_width", accepted=False)
    # The other arm is still unexplored, so it is tried next regardless of memory.
    assert bandit.select("k::hot").name == "unroll"


# --- negative memory -------------------------------------------------------

def test_remembers_a_rejected_edit(tmp_path):
    repo = _repo(tmp_path)
    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_rejected("kernels/k.cu", "TILE_N = 64", "TILE_N = 256",
                      verdict="build-failed", detail="out of shared memory")

    assert mem.known_bad(repo, "kernels/k.cu", "TILE_N = 64", "TILE_N = 256") == "build-failed"
    # A different replacement is a different idea.
    assert mem.known_bad(repo, "kernels/k.cu", "TILE_N = 64", "TILE_N = 128") is None


def test_rejection_expires_when_the_code_moves(tmp_path):
    repo = _repo(tmp_path)
    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_rejected("kernels/k.cu", "TILE_N = 64", "TILE_N = 256", verdict="build-failed")

    # The kernel is rewritten; the old anchor is gone, so the memory no longer applies.
    _repo(tmp_path, "constexpr int TILE_N = 96;\n")
    assert mem.known_bad(repo, "kernels/k.cu", "TILE_N = 64", "TILE_N = 256") is None


def test_rejection_expires_by_age(tmp_path):
    repo = _repo(tmp_path)
    mem = Memory.load(tmp_path / "mem.json", max_age_rounds=2)
    mem.start_round()
    mem.note_rejected("kernels/k.cu", "TILE_N = 64", "TILE_N = 256", verdict="build-failed")

    for _ in range(2):
        mem.start_round()
    assert mem.known_bad(repo, "kernels/k.cu", "TILE_N = 64", "TILE_N = 256") == "build-failed"

    mem.start_round()
    assert mem.known_bad(repo, "kernels/k.cu", "TILE_N = 64", "TILE_N = 256") is None


def test_missing_file_is_not_treated_as_a_known_rejection(tmp_path):
    repo = _repo(tmp_path)
    mem = Memory.load(tmp_path / "mem.json")
    mem.start_round()
    mem.note_rejected("kernels/gone.cu", "x", "y", verdict="build-failed")
    assert mem.known_bad(repo, "kernels/gone.cu", "x", "y") is None


def test_edit_key_is_content_addressed():
    a = edit_key("k.cu", "x", "y")
    assert a == edit_key("k.cu", "x", "y")
    assert a != edit_key("k.cu", "x", "z")
    assert a != edit_key("other.cu", "x", "y")
