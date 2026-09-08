from pathlib import Path

import pytest

from cuda_agent.edits import (
    Edit,
    EditError,
    apply_all,
    apply_edit,
    check_path,
    looks_like_attempted_edit,
    parse_edits,
)

ALLOWED = ("kernels/", "runtime/", "moe/")
DENIED = ("bench/scripts/", "eval/", "runtime/examples/dspark_tau_check.cpp")

BLOCK = """I'll widen the tile.

<edit path="kernels/a.cu">
<<<<<<< SEARCH
constexpr int TILE_N = 64;
=======
constexpr int TILE_N = 128;
>>>>>>> REPLACE
</edit>
"""


def test_parses_a_single_edit():
    edits = parse_edits(BLOCK)
    assert len(edits) == 1
    assert edits[0].path == "kernels/a.cu"
    assert edits[0].search == "constexpr int TILE_N = 64;"
    assert edits[0].replace == "constexpr int TILE_N = 128;"


def test_parses_multiple_edits():
    edits = parse_edits(BLOCK + BLOCK.replace("a.cu", "b.cu"))
    assert [e.path for e in edits] == ["kernels/a.cu", "kernels/b.cu"]


def test_prose_yields_no_edits():
    assert parse_edits("I think you should try a wider tile.") == []
    assert not looks_like_attempted_edit("I think you should try a wider tile.")


def test_truncated_edit_is_detected_as_attempted():
    truncated = '<edit path="kernels/a.cu">\n<<<<<<< SEARCH\nfoo\n'
    assert parse_edits(truncated) == []
    assert looks_like_attempted_edit(truncated)


@pytest.mark.parametrize(
    "path",
    ["bench/scripts/label.py", "eval/pr_eval_bot.py", "runtime/examples/dspark_tau_check.cpp"],
)
def test_maintainer_owned_paths_are_rejected(path):
    with pytest.raises(EditError, match="maintainer-owned"):
        check_path(path, allowed=ALLOWED, denied=DENIED)


def test_paths_outside_the_surface_are_rejected():
    with pytest.raises(EditError, match="outside the editable surface"):
        check_path("README.md", allowed=ALLOWED, denied=DENIED)


def test_traversal_is_rejected():
    with pytest.raises(EditError, match="escapes the repo"):
        check_path("kernels/../../etc/passwd", allowed=ALLOWED, denied=DENIED)
    with pytest.raises(EditError, match="escapes the repo"):
        check_path("/etc/passwd", allowed=ALLOWED, denied=DENIED)


def _repo(tmp_path: Path, body: str) -> Path:
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "a.cu").write_text(body)
    return tmp_path


def test_apply_rewrites_the_file(tmp_path):
    repo = _repo(tmp_path, "constexpr int TILE_N = 64;\n")
    apply_edit(repo, parse_edits(BLOCK)[0], allowed=ALLOWED, denied=DENIED)
    assert (repo / "kernels" / "a.cu").read_text() == "constexpr int TILE_N = 128;\n"


def test_ambiguous_anchor_is_rejected(tmp_path):
    repo = _repo(tmp_path, "constexpr int TILE_N = 64;\nconstexpr int TILE_N = 64;\n")
    with pytest.raises(EditError, match="ambiguous"):
        apply_edit(repo, parse_edits(BLOCK)[0], allowed=ALLOWED, denied=DENIED)


def test_missing_anchor_is_rejected(tmp_path):
    repo = _repo(tmp_path, "constexpr int TILE_M = 64;\n")
    with pytest.raises(EditError, match="not found"):
        apply_edit(repo, parse_edits(BLOCK)[0], allowed=ALLOWED, denied=DENIED)


def test_noop_edit_is_rejected(tmp_path):
    repo = _repo(tmp_path, "x\n")
    with pytest.raises(EditError, match="identical"):
        apply_edit(repo, Edit("kernels/a.cu", "x", "x"), allowed=ALLOWED, denied=DENIED)


def test_batch_is_all_or_nothing(tmp_path):
    """A batch whose second edit is bad must not leave the first one applied."""
    repo = _repo(tmp_path, "constexpr int TILE_N = 64;\n")
    good = Edit("kernels/a.cu", "TILE_N = 64", "TILE_N = 128")
    bad = Edit("kernels/a.cu", "NOT PRESENT", "x")
    with pytest.raises(EditError, match="not found"):
        apply_all(repo, [good, bad], allowed=ALLOWED, denied=DENIED)
    assert (repo / "kernels" / "a.cu").read_text() == "constexpr int TILE_N = 64;\n"


def test_batch_sequences_edits_to_the_same_file(tmp_path):
    repo = _repo(tmp_path, "int a = 1;\nint b = 2;\n")
    apply_all(
        repo,
        [Edit("kernels/a.cu", "int a = 1;", "int a = 10;"),
         Edit("kernels/a.cu", "int b = 2;", "int b = 20;")],
        allowed=ALLOWED,
        denied=DENIED,
    )
    assert (repo / "kernels" / "a.cu").read_text() == "int a = 10;\nint b = 20;\n"
