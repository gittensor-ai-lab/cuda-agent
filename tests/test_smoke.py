"""Smoke is a diagnostic, so its own tests are about *reporting*, not passing.

The properties that matter: a check that fails says what it saw, later checks
skip instead of cascading, and a degraded-but-usable box warns rather than
blocking the round.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cuda_agent import smoke as smoke_mod
from cuda_agent.backend import Completion, Usage
from cuda_agent.config import Settings
from cuda_agent.harness import AccuracyResult, BenchResult, BuildResult
from cuda_agent.smoke import FAIL, PASS, SKIP, WARN

KERNEL = """\
__global__ void hot_kernel(const float* a, float* b, int n) {
    constexpr int TILE_N = 64;
    for (int i = 0; i < n; i += TILE_N) b[i] = a[i];
}
"""

EDIT = (
    '<edit path="kernels/hot.cu">\n<<<<<<< SEARCH\n'
    "    constexpr int TILE_N = 64;\n=======\n"
    "    constexpr int TILE_N = 128;\n>>>>>>> REPLACE\n</edit>\n"
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "kernels").mkdir(parents=True)
    (root / "kernels" / "hot.cu").write_text(KERNEL)
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "base"], check=True)
    return root


@pytest.fixture
def settings(repo, tmp_path):
    return Settings(api_key="k", repo=repo, out_dir=tmp_path / "out",
                    worktree_root=tmp_path / "wt")


class FakeGateway:
    def __init__(self, text=EDIT, *, finish="stop", uid="uid-7", tokens=(120, 40)):
        self.text, self.finish, self.uid, self.tokens = text, finish, uid, tokens
        self.calls = 0

    def complete(self, messages, *, max_tokens=None):
        self.calls += 1
        u = Usage(prompt_tokens=self.tokens[0], completion_tokens=self.tokens[1])
        return Completion(text=self.text, usage=u, served_uid=self.uid, finish_reason=self.finish)

    def close(self): pass


class FakeHarness:
    build_cmd = quick_bench_cmd = accuracy_cmd = "fake"

    def __init__(self, *, build_ok=True, tps=None, acc_ok=True):
        self.build_ok, self.tps, self.acc_ok = build_ok, tps if tps is not None else {"4k": 100.0}, acc_ok

    def build(self, path): return BuildResult(ok=self.build_ok, log="error: expected ';'")
    def accuracy(self, path):
        return AccuracyResult(ok=self.acc_ok, top1=0.96, kl=0.012,
                              detail="" if self.acc_ok else "could not parse accuracy output")
    def bench(self, path, *, quick=False):
        return BenchResult(ok=bool(self.tps), tps=self.tps,
                           detail="" if self.tps else "no tok/s found in output: [SKIP] no GPU")


def by_name(result):
    return {c.name: c for c in result.results}


def test_a_healthy_box_passes_everything(settings):
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway(), profiler=None)
    checks = by_name(result)
    assert not result.failed
    assert checks["targets"].status == PASS
    assert checks["knobs"].status == PASS
    assert checks["bench"].status == PASS
    assert checks["sweep-iteration"].status == PASS


def test_a_bench_format_mismatch_shows_what_it_printed(settings):
    """The exact failure that fixtures could not catch."""
    result = smoke_mod.run(settings, harness=FakeHarness(tps={}), gateway=FakeGateway())
    bench = by_name(result)["bench"]
    assert bench.status == FAIL
    assert "[SKIP] no GPU" in bench.evidence
    assert "[SKIP] no GPU" in smoke_mod.render(result)


def test_later_checks_skip_instead_of_cascading(settings):
    result = smoke_mod.run(settings, harness=FakeHarness(build_ok=False), gateway=FakeGateway())
    checks = by_name(result)
    assert checks["build"].status == FAIL
    assert "expected ';'" in checks["build"].evidence
    for name in ("bench", "accuracy", "sweep-iteration"):
        assert checks[name].status == SKIP
        assert "needs" in checks[name].detail


def test_missing_key_fails_before_anything_is_spent(settings):
    settings.api_key = ""
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway())
    checks = by_name(result)
    assert checks["config"].status == FAIL
    assert checks["gateway-auth"].status == SKIP


def test_a_non_git_repo_is_caught(settings, tmp_path):
    plain = tmp_path / "plain"
    (plain / "kernels").mkdir(parents=True)
    settings.repo = plain
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway())
    assert "not a git checkout" in by_name(result)["config"].detail


# --- gateway shape ---------------------------------------------------------

def test_gateway_reports_the_response_shape(settings):
    gw = FakeGateway()
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=gw)
    auth = by_name(result)["gateway-auth"]
    assert auth.status == PASS
    assert "uid-7" in auth.detail


def test_a_missing_served_uid_warns_but_does_not_block(settings):
    """Shape drift is worth knowing about; it is not a reason to refuse a round."""
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway(uid=""))
    auth = by_name(result)["gateway-auth"]
    assert auth.status == WARN
    assert "no gittensor.served_uid" in auth.detail
    assert not result.failed


def test_context_check_reports_prompt_size_and_parses_an_edit(settings):
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway())
    ctx = by_name(result)["gateway-context"]
    assert ctx.status == PASS
    assert "parsed 1 edit(s)" in ctx.detail
    assert ctx.data["prompt_tokens"] == 120


def test_a_truncated_answer_is_surfaced(settings):
    """The 1024-token cap cutting an edit in half is the thing to find early."""
    gw = FakeGateway(text='<edit path="kernels/hot.cu">\n<<<<<<< SEARCH\nx', finish="length")
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=gw)
    ctx = by_name(result)["gateway-context"]
    assert ctx.status == WARN
    assert "token cap" in ctx.detail


def test_a_prose_answer_is_surfaced_with_the_text(settings):
    gw = FakeGateway(text="You could try widening the tile.")
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=gw)
    ctx = by_name(result)["gateway-context"]
    assert ctx.status == WARN
    assert "prose" in ctx.detail
    assert "widening the tile" in ctx.evidence


def test_no_gpu_mode_still_checks_config_and_gateway(settings):
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway(), skip_gpu=True)
    checks = by_name(result)
    assert checks["gateway-context"].status == PASS
    for name in ("build", "bench", "accuracy", "sweep-iteration"):
        assert checks[name].status == SKIP


def test_a_broken_check_is_a_failed_check_not_a_crash(settings):
    class Exploding(FakeHarness):
        def build(self, path): raise RuntimeError("docker is not running")

    result = smoke_mod.run(settings, harness=Exploding(), gateway=FakeGateway())
    assert "docker is not running" in by_name(result)["build"].detail


def test_a_missing_base_ref_fails_the_round(settings):
    """spark2_5 lives only on feat/spark-x25-4b; a checkout without it would
    silently optimise a tree that does not contain the target."""
    settings.base_ref = "feat/does-not-exist"
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway())
    config = by_name(result)["config"]
    assert config.status == FAIL
    assert "not found" in config.detail


def test_the_resolved_ref_is_reported(settings):
    settings.base_ref = "main"
    result = smoke_mod.run(settings, harness=FakeHarness(), gateway=FakeGateway())
    config = by_name(result)["config"]
    assert config.status == PASS
    assert "ref=main@" in config.detail
