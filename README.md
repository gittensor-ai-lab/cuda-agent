# cuda-agent

An autonomous CUDA optimization agent for [sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer),
driven by [Gittensor Compute](https://docs.gittensor.io/compute-serving).

## The one design decision

**The loop is deterministic Python; the model is a subroutine inside it.**

The model proposes an edit. The harness builds it, checks correctness against the
llama.cpp reference, and measures it. The *loop* decides. The model never keeps its
own change, and a number the model reports is never treated as a result.

```
Optimizer (owns the frontier)
   ├─ Proposer  ── target + family + prior verdicts ──► one anchored edit
   ├─ Executor  ── worktree → apply → build → accuracy → bench
   └─ Bandit    ── which hypothesis family to try next, per target
```

## What the gateway forces

Gittensor Compute serves one model (`qwen3.6-35b-a3b`) with constraints that shaped
everything above:

| constraint | consequence |
|---|---|
| `max_tokens` capped at 1024 | small anchored search/replace edits, never file rewrites |
| sampling ignored, greedy only | zero diversity from temperature — it comes from *structure* (target × family) instead |
| `429` with no queue | backoff lives in the harness, so an agent is not scored on someone else's retry storm |
| 3B active parameters | autotune deterministically first; spend tokens only where a sweep cannot reach |

## Layout

| module | what |
|---|---|
| `config.py` | env-driven settings; the editable-path allowlist |
| `backend.py` | gateway client — token clamping, 429 backoff, usage and `served_uid` accounting |
| `edits.py` | the search/replace format, path guard, all-or-nothing batch apply |
| `source.py` | brace-matched function extraction — verbatim, so anchors still match |
| `autotune.py` | deterministic constant sweep — runs *before* any token is spent |
| `profile.py` | ncu profiling, bottleneck verdicts, profile-directed targeting |
| `memory.py` | cross-round memory — family priors, rejected edits, swept knobs |
| `smoke.py` | preflight — exercises every layer against the real box and gateway |
| `families.py` | named optimization families + per-target UCB1 bandit |
| `worktree.py` | one git worktree per candidate; the frontier only moves on a measured win |
| `harness.py` | the gate: build → accuracy → speed, with the significance and regression rules |
| `runner.py` | subprocess harness bound to sparkinfer's bench scripts (env-configurable) |
| `loop.py` | the optimizer |
| `cli.py` | arena entrypoint |

## Arena contract

```
in    $CUDA_AGENT_REPO       checkout at the round's pinned base commit
      $GITTENSOR_API_KEY     scoped, metered, per-round
out   $CUDA_AGENT_OUT/patch.diff      the submission
      $CUDA_AGENT_OUT/report.json     evidence — never the score
      $CUDA_AGENT_OUT/ledger.jsonl    every attempt, written as it happens
```

The patch is written on every exit path including SIGTERM at the deadline, so a
killed agent still submits its best *verified* state. Scoring is the arena
re-running that patch cold through sparkinfer's own eval path — never these numbers.

## Run

```bash
pip install -e '.[dev]' && pytest

GITTENSOR_API_KEY=... CUDA_AGENT_REPO=/path/to/sparkinfer \
  cuda-agent --smoke            # preflight everything, then exit
```

**Run `--smoke` before any real round.** Unit tests prove the code agrees with its
fixtures; smoke proves it agrees with *this* box, gateway and checkout — which is a
different claim, and the one that matters. Checks run cheapest-first and skip rather than
cascade, and anything that fails prints the raw output that confused it:

```
[ok  ] config             repo=/…/sparkinfer model=qwen3.6-35b-a3b max_tokens=1024
[ok  ] targets            139 targets across 43 files; first extract 2511B
[ok  ] knobs              75 knobs {'tile': 30, 'constexpr': 30, 'define': 7, …}
[FAIL] gateway-auth       gateway 401: {"error":{"message":"invalid api key"}}
[skip] gateway-context    needs gateway-auth
```

`--smoke-no-gpu` runs config and gateway only. Results also land in `out/smoke.json`;
exit is non-zero if anything failed.

The two checks worth reading closely:

- **`gateway-context`** sends a *real* proposal prompt built from a real kernel, and
  reports the prompt token count and whether an edit parsed back. It answers the one
  question fixtures cannot — does the served context window fit a kernel function plus
  its history block, and does this model emit applicable edits at all.
- **`bench`** is where a parser/format mismatch surfaces instantly. It has already caught
  one: the real line is `decode tg : 95.70 tok/s (… ctx=4096 …)`, throughput *before*
  context, and a regex written the other way round matched nothing at all.

Bench wiring is env-configurable, because sparkinfer's scored target moves:

Bench wiring is env-configurable, because sparkinfer's scored target moves:

```bash
CUDA_AGENT_BUILD_CMD="cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j"
CUDA_AGENT_BENCH_CMD="bench/scripts/bench.sh --tokens 128"
CUDA_AGENT_QUICK_BENCH_CMD="bench/scripts/bench.sh --ctx 4096 --tokens 64"
CUDA_AGENT_ACCURACY_CMD="bench/scripts/accuracy.sh"
```

## Two phases

**1. Deterministic sweep.** Tile sizes, block dims, unroll factors, `__launch_bounds__` —
found by counting, not by reasoning. Costs zero tokens and does not depend on the model
being clever. On the current sparkinfer tree it finds **75 tunable constants across 43
kernel files**, geometry first:

```bash
cuda-agent --list-knobs
kernels/csrc/cuda/fused/batched_prefill.cu::PF_BM@128   define  -> [64, 256, 32, 512]
```

Capacity constants (`kMaxDevices`, `MAX_SEQ_LEN`, `kSlotCount`) are deliberately
excluded. Shrinking a buffer bound can pass a short accuracy check and still overflow
under a longer context — a latent bug wearing a speedup's clothes.

Search is coordinate descent, not a grid: knobs interact, but a grid over ten knobs is
thousands of builds and the round is three hours long.

**2. Proposer, aimed by the profiler.** What a sweep cannot reach — restructuring, fusion,
memory hints. Runs on the frontier the sweep left behind, so the two phases cannot
double-count a gain.

`ncu` decides *where* to look and *what to try*. Each hot kernel gets a bottleneck
verdict from its throughput, occupancy and dominant warp stall:

| verdict | first families tried |
|---|---|
| memory-bound | `vector_width`, `memory_hint`, `smem_layout`, `fusion` |
| compute-bound | `tile_shape`, `unroll`, `redundant_work` |
| occupancy-limited | `launch_config`, `unroll`, `tile_shape` |
| latency-bound | `unroll`, `split_k`, `launch_config` |

The verdict seeds only the *exploration order*; once an arm is pulled its value comes
from measurement, so a wrong prior costs a few iterations rather than skewing the round.
Target rotation is weighted by share of GPU time, and the profile is refreshed after the
frontier moves — fixing the top kernel moves the bottleneck, and continuing to hammer it
is how an agent spends an hour on a kernel that stopped being the problem.

Kernels that don't map to editable source (cuBLAS, CUTLASS) are dropped rather than
guessed at. If `ncu` is missing or counters are admin-restricted
(`NVreg_RestrictProfilingToAdminUsers`), the round proceeds round-robin and the reason
lands in `report.json`.

## Cross-round memory

A daily competition rewards an agent that remembers yesterday. Three things carry, and
they pay in different currencies:

| carried | why |
|---|---|
| which families paid on which kernel | with ~20 proposer iterations a round, starting in the right part of the space is most of the game |
| edits already tried and rejected | the expensive one — re-proposing last week's failure costs a full build-and-benchmark cycle |
| knob values already swept | same argument, applied to the sweep budget |

Negative memory is the dangerous half: code moves, and an edit that failed against last
week's kernel may be right against today's. So a rejection is trusted only while **its
anchor text still exists unchanged in the file**, and it expires by age regardless
(7 rounds). Suppressing a good idea forever is a worse failure than re-running one build.

Precedence when both speak: **this round's profile leads, memory follows.** A kernel that
was compute-bound yesterday may not be today, so fresh evidence orders the families first
and remembered winners are appended rather than dropped.

The store is written atomically and saved on SIGTERM too — a round killed at the deadline
still learned something, and the deadline should not be the one place the agent forgets.

`--no-memory` starts cold. Whether an agent gets a persistent volume between rounds at all
is the arena's call, not the agent's.

## Not built yet

- Proposal pipelining against measurement (the GPU is serialized; gateway latency is not).
