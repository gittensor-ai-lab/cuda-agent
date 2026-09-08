"""Parsing tests written against sparkinfer's *actual* output formats.

Taken from runtime/examples/qwen3_gguf_bench.cpp and
bench/scripts/accuracy_compare.py, not invented -- a parser tested only against
made-up fixtures agrees with the fixtures, not with the program.
"""

from __future__ import annotations

from cuda_agent.runner import parse_tps

# qwen3_gguf_bench.cpp:254
SWEEP = (
    'SWEEP_JSON {"0":{"decode_tps":95.7000,"prefill_pp":6942.0000},'
    '"4096":{"decode_tps":93.6000,"prefill_pp":14364.0000},'
    '"16384":{"decode_tps":90.2000,"prefill_pp":13794.0000}}\n'
)

# qwen3_gguf_bench.cpp:196 / :72
HUMAN = (
    "=== sparkinfer — decode (n=128, ctx=4096, bs=1) ===\n"
    "decode tg    : 93.60 tok/s  (11.4 ms/token, n=128, ctx=4096, bs=1)\n"
    "prefill pp   : 14364.00 tok/s  (ctx=4096, sequential KV fill)\n"
)


def test_prefers_the_structured_sweep_line():
    assert parse_tps(SWEEP) == {"0": 95.7, "4k": 93.6, "16k": 90.2}


def test_parses_the_human_line_with_tps_before_ctx():
    """The throughput comes BEFORE the context on this line, not after."""
    assert parse_tps(HUMAN) == {"4k": 93.6}


def test_does_not_confuse_prefill_for_decode():
    """prefill pp is also 'tok/s' -- only decode tg may be scored."""
    assert 14364.0 not in parse_tps(HUMAN).values()


def test_multi_context_human_output():
    out = "".join(
        f"decode tg    : {t:.2f} tok/s  (n=128, ctx={c}, bs=1)\n"
        for c, t in ((0, 95.7), (4096, 93.6), (32768, 81.3))
    )
    assert parse_tps(out) == {"0": 95.7, "4k": 93.6, "32k": 81.3}


def test_a_line_does_not_pair_with_another_lines_context():
    """Regression: a DOTALL pattern matched one line's tok/s to the next ctx."""
    out = (
        "decode tg    : 95.70 tok/s  (n=128, ctx=0, bs=1)\n"
        "decode tg    : 81.30 tok/s  (n=128, ctx=32768, bs=1)\n"
    )
    assert parse_tps(out) == {"0": 95.7, "32k": 81.3}


def test_the_last_sweep_wins():
    stale = SWEEP.replace("95.7000", "10.0000")
    assert parse_tps(stale + SWEEP)["0"] == 95.7


def test_sweep_beats_the_human_lines_when_both_are_present():
    assert parse_tps(HUMAN + SWEEP) == {"0": 95.7, "4k": 93.6, "16k": 90.2}


def test_malformed_sweep_falls_back_to_the_human_lines():
    assert parse_tps(HUMAN + "SWEEP_JSON {broken\n") == {"4k": 93.6}


def test_zero_and_missing_values_are_dropped():
    out = 'SWEEP_JSON {"0":{"decode_tps":0.0},"4096":{"prefill_pp":1.0},"16384":{"decode_tps":90.2}}\n'
    assert parse_tps(out) == {"16k": 90.2}


def test_no_bench_output_yields_nothing():
    assert parse_tps("") == {}
    assert parse_tps("[SKIP] no GPU\n") == {}


# --- accuracy parsing, against accuracy_compare.py's real output -------------

# accuracy_compare.py:115-120 -- the human lines print BEFORE the machine line,
# so a lazy regex can pick up the wrong number.
ACCURACY = (
    "token-match (top-1)   : 123/128 = 0.961   (bar >= 0.90)\n"
    "mean KL(llama||spark) : 0.0120 nats  (top-k approx)\n"
    "H1 top1=0.960000 kl=0.012000 ppl_spark=8.1234 ppl_llama=8.0011\n"
)

# accuracy_compare.py:111 -- the sentinel emitted when nothing was scored.
NO_POSITIONS = "H1 top1=0 kl=99 ppl_spark=0 ppl_llama=0   (NO SCORED POSITIONS)\n"


def _accuracy(out, tmp_path):
    from cuda_agent.runner import SubprocessHarness
    import subprocess

    harness = SubprocessHarness(repo=tmp_path)
    harness._run = lambda *a, **k: subprocess.CompletedProcess([], 0, out, "")
    return harness.accuracy(tmp_path)


def test_accuracy_reads_the_machine_line_not_the_human_one(tmp_path):
    result = _accuracy(ACCURACY, tmp_path)
    assert result.ok
    assert result.top1 == 0.96, "0.961 is the human line; 0.960000 is the machine line"
    assert result.kl == 0.012


def test_the_no_scored_positions_sentinel_fails_the_gate(tmp_path):
    """top1=0 kl=99 must never read as a pass."""
    assert not _accuracy(NO_POSITIONS, tmp_path).ok


def test_unparseable_accuracy_output_fails_closed(tmp_path):
    """'No numbers found' is how a broken gate silently stops gating."""
    result = _accuracy("everything is fine, honest\n", tmp_path)
    assert not result.ok
    assert "could not parse" in result.detail
