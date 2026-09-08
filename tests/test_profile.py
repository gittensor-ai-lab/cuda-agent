from __future__ import annotations

from pathlib import Path

import pytest

from cuda_agent.profile import (
    COMPUTE_BOUND,
    LATENCY_BOUND,
    MEMORY_BOUND,
    OCCUPANCY_LIMITED,
    KernelProfile,
    Profiler,
    build_symbol_index,
    parse_ncu_csv,
    rank_targets,
)

HEADER = (
    '"ID","Process ID","Kernel Name","Block Size","Grid Size",'
    '"Section Name","Metric Name","Metric Unit","Metric Value"'
)

HOT = "flash_decode_gqa8_kernel<256, 8, 16>(const __nv_bfloat16 *, float *, int)"
COLD = "rmsnorm_kernel(float *, int)"
EXTERNAL = "cutlass::device_kernel<...>(...)"


def row(idx, kernel, metric, value, unit=""):
    return f'"{idx}","1","{kernel}","256","1024","","{metric}","{unit}","{value}"'


def csv_of(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


def test_aggregates_durations_and_averages_rates():
    text = csv_of(
        row(0, HOT, "gpu__time_duration.sum", "1,000"),
        row(1, HOT, "gpu__time_duration.sum", "3,000"),
        row(0, HOT, "sm__throughput.avg.pct_of_peak_sustained_elapsed", "20"),
        row(1, HOT, "sm__throughput.avg.pct_of_peak_sustained_elapsed", "40"),
        row(0, COLD, "gpu__time_duration.sum", "1000"),
    )
    profiles = parse_ncu_csv(text)
    hot = profiles[0]

    assert hot.name == HOT
    assert hot.time_ns == 4000, "durations sum across launches"
    assert hot.compute_pct == 30, "rates average across launches"
    assert hot.launches == 2
    assert hot.time_share == pytest.approx(0.8)
    assert profiles[1].time_share == pytest.approx(0.2)


def test_base_name_strips_templates_and_parameters():
    assert KernelProfile(name=HOT).base_name == "flash_decode_gqa8_kernel"
    assert KernelProfile(name="ns::inner<int, 4>(float*)").base_name == "inner"


def test_unparseable_values_are_skipped_not_zeroed():
    text = csv_of(
        row(0, HOT, "gpu__time_duration.sum", "n/a"),
        row(0, HOT, "sm__throughput.avg.pct_of_peak_sustained_elapsed", "55"),
    )
    hot = parse_ncu_csv(text)[0]
    assert hot.time_ns == 0.0
    assert hot.compute_pct == 55


def test_empty_or_headerless_output_is_survivable():
    assert parse_ncu_csv("") == []
    assert parse_ncu_csv("garbage,not,ncu\n1,2,3\n") == []


# --- verdicts --------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(memory_pct=82, compute_pct=20, occupancy_pct=70), MEMORY_BOUND),
        (dict(memory_pct=15, compute_pct=75, occupancy_pct=70), COMPUTE_BOUND),
        (dict(memory_pct=20, compute_pct=18, occupancy_pct=12), OCCUPANCY_LIMITED),
        (dict(memory_pct=25, compute_pct=22, occupancy_pct=80), LATENCY_BOUND),
    ],
)
def test_verdict_separates_the_four_regimes(kwargs, expected):
    assert KernelProfile(name=HOT, **kwargs).verdict == expected


def test_top_stall_and_family_suggestions():
    prof = KernelProfile(
        name=HOT, memory_pct=85, compute_pct=20, occupancy_pct=60,
        stalls={"long_scoreboard": 4.2, "barrier": 0.8},
    )
    assert prof.top_stall == ("long_scoreboard", 4.2)
    families = prof.suggested_families()
    assert families[0] in ("vector_width", "memory_hint", "smem_layout")
    assert "vector_width" in families


def test_hint_names_the_bottleneck_and_the_stall():
    prof = KernelProfile(
        name=HOT, time_share=0.42, launches=8, memory_pct=85,
        compute_pct=20, occupancy_pct=61, registers=64,
        stalls={"long_scoreboard": 4.2},
    )
    hint = prof.hint()
    assert "flash_decode_gqa8_kernel is memory-bound" in hint
    assert "42% of profiled GPU time" in hint
    assert "long_scoreboard" in hint
    assert "waiting on global/local memory loads" in hint


# --- targeting -------------------------------------------------------------

def test_symbol_index_and_ranking_map_kernels_to_source(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "fd.cu").write_text(
        "template <int A>\n__global__ void flash_decode_gqa8_kernel(float* x) {}\n"
    )
    index = build_symbol_index(tmp_path)
    assert index["flash_decode_gqa8_kernel"] == "kernels/fd.cu"

    profiles = [
        KernelProfile(name=HOT, time_ns=900, time_share=0.9),
        KernelProfile(name=EXTERNAL, time_ns=100, time_share=0.1),
    ]
    ranked = rank_targets(profiles, index)
    assert len(ranked) == 1, "a kernel with no editable source is dropped"
    target, prof = ranked[0]
    assert target.path == "kernels/fd.cu"
    assert target.symbol == "flash_decode_gqa8_kernel"


def test_ranking_ignores_negligible_kernels(tmp_path):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "fd.cu").write_text("__global__ void tiny_kernel(float* x) {}\n")
    index = build_symbol_index(tmp_path)
    profiles = [KernelProfile(name="tiny_kernel(float*)", time_ns=1, time_share=0.001)]
    assert rank_targets(profiles, index) == []


# --- degradation -----------------------------------------------------------

def test_missing_ncu_degrades_instead_of_raising(tmp_path):
    prof = Profiler(tmp_path, "true", ncu_bin="definitely-not-a-real-binary")
    assert prof.available() is False
    assert prof.collect(tmp_path) == []
    assert "not found" in prof.last_error


# --- real failure text from an RTX 5090 in an unprivileged container ---------

NVGPUCTRPERM = (
    "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access "
    "NVIDIA GPU Performance Counters on the target device 0. For instructions on "
    "enabling permissions and to get more information see "
    "https://developer.nvidia.com/ERR_NVGPUCTRPERM\n"
)
# What the profiled program itself printed, on stderr, at the same time.
WORKLOAD_NOISE = ">> using local build\n[gguf] layer 0 loaded\n[runtime] RTX 5090\n"


def _profiler_returning(stdout, stderr, tmp_path):
    import subprocess
    from unittest.mock import patch
    prof = Profiler(tmp_path, "bench", ncu_bin="ncu")
    done = subprocess.CompletedProcess([], 1, stdout, stderr)
    with patch("shutil.which", return_value="/usr/local/cuda/bin/ncu"), \
         patch("subprocess.run", return_value=done):
        return prof, prof.collect(tmp_path)


def test_permission_error_is_recognised_on_stdout(tmp_path):
    """ncu reports its errors on stdout; the workload's chatter is on stderr.

    Scanning only stderr surfaced '>> using local build' as the ncu error --
    a real bug this test pins down.
    """
    prof, profiles = _profiler_returning(NVGPUCTRPERM, WORKLOAD_NOISE, tmp_path)
    assert profiles == []
    assert "ERR_NVGPUCTRPERM" in prof.last_error
    assert "round-robin" in prof.last_error
    assert "using local build" not in prof.last_error


def test_an_unclassified_failure_prefers_ncus_own_error_lines(tmp_path):
    prof, _ = _profiler_returning("==ERROR== Unable to profile 3 kernels\n", WORKLOAD_NOISE, tmp_path)
    assert "Unable to profile 3 kernels" in prof.last_error
    assert "gguf" not in prof.last_error
