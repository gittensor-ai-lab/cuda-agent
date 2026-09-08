# cuda-agent

An autonomous CUDA optimization agent for [sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer),
and the reference entry for the **SN74 agent competition**.

Miners submit an agent. Every round, each submitted agent gets the same GPU, the same
budget and the same pinned sparkinfer commit, and searches for a real speedup. The best
verified result wins the round, and the winner opens the pull request that settles it.

## How the competition works

```
1. submit      a miner submits their agent
2. evaluate    every agent runs on identical hardware, same budget, same base commit
3. verify      the leader's patch is re-measured cold on a second box
4. publish     the round report goes public: every entry, its measurement, its label
5. settle      the winner opens a PR with their patch; that PR carries the round's label
```

**The PR is the payout instrument, not the competition.** Nothing about the PR's size or
content earns anything — the number was already established by the arena, on hardware the
maintainer controls, against a reference implementation. The PR is how the result reaches
Gittensor's OSS scoring, which pays for merged pull requests.

That means **an unlabeled merge earns nothing**, by configuration. Only the round winner's
PR carries a scoring label.

## What actually scores

The same thing sparkinfer pays for: a **verified speedup that survives cold
re-verification**. Not a plausible one, not a self-reported one.

Every candidate an agent produces passes the same gate in the same order — build, then
correctness, then speed. Correctness first because a speed win that changes the model's
output is not a win. That ordering is not a formality: in testing, a 3B-active proposer
produced an edit that compiled cleanly, applied cleanly, and looked like a textbook
parallelisation win, while silently breaking the online softmax. Only the correctness gate
caught it.

If no agent clears the 2% significance floor, **the round pays nothing and rolls over.** A
quiet day should not pay out for measurement noise.

## The agent

The central decision: **the loop is deterministic Python and the model is a subroutine
inside it.** The model proposes an edit; the harness builds, checks and measures it; the
loop decides. The model never keeps its own change, and a number it reports is never
treated as a result.

```
Optimizer (owns the frontier)
   ├─ Sweep      deterministic constant search — zero tokens
   ├─ Profiler   ncu verdict per hot kernel: memory / compute / occupancy / latency bound
   ├─ Proposer   one anchored edit per turn, aimed by that verdict
   └─ Gate       build → correctness → speed; anything else is reverted
```

**Two phases, one frontier.** The sweep runs first — tile shapes, block dims, unroll
factors, `__launch_bounds__` — finding 75 tunable constants across 43 kernel files at no
token cost. The proposer then works on the frontier the sweep left behind, so the two
cannot double-count a gain.

Capacity constants (`kMaxDevices`, `MAX_SEQ_LEN`, `kSlotCount`) are deliberately excluded
from the sweep: shrinking a buffer bound can pass a short correctness check and still
overflow under a longer context — a latent bug wearing a speedup's clothes.

## What the gateway forces

The agent runs on [Gittensor Compute](https://docs.gittensor.io/compute-serving), which
serves `qwen3.6-35b-a3b` from sparkinfer on miner-owned RTX 5090s. Its constraints shaped
the design:

| constraint | consequence |
|---|---|
| `max_tokens` capped at 1024 | small anchored search/replace edits, never file rewrites |
| sampling ignored, greedy only | no diversity from temperature — it comes from *structure* (target × technique) |
| `429` with no queue | backoff lives in the harness, so an agent is not scored on someone else's retry storm |
| 3B active parameters | sweep deterministically first; spend tokens only where a sweep cannot reach |

There is a flywheel here worth naming: the agent optimising sparkinfer **runs on
sparkinfer**, served by SN74 compute miners. A faster runtime improves serving economics,
which attracts compute miners, which adds the capacity this competition consumes.

## Measured on an RTX 5090

Against `feat/spark-x25-4b`, the Spark-X2.5-4B target:

| | |
|---|---|
| spark2_5 decode | 335 tok/s, 11.1 GB (Q4_K_M) |
| candidate iteration | **80 s** — 79 s incremental build + 1 s bench |
| correctness gate | 2–3 s differential; 298 s independent reference |
| candidates per 3-hour round | ~130 |
| inference cost per round | ~$0.005 |

## Layout

| module | what |
|---|---|
| `autotune.py` | deterministic constant sweep — runs before any token is spent |
| `profile.py` | ncu profiling, bottleneck verdicts, profile-directed targeting |
| `families.py` | named optimization techniques + per-target UCB1 bandit |
| `memory.py` | cross-round memory — technique priors, rejected edits, swept knobs |
| `worktree.py` | the frontier; only moves on a measured win |
| `harness.py` / `runner.py` | the gate: build → correctness → speed |
| `smoke.py` | preflight against the real box, gateway and checkout |
| `adapters/` | correctness gates per target |

## Run it

```bash
pip install -e '.[dev]' && pytest

GITTENSOR_API_KEY=... CUDA_AGENT_REPO=/path/to/sparkinfer \
  cuda-agent --smoke --base-ref feat/spark-x25-4b
```

**Run `--smoke` before anything else.** Unit tests prove the code agrees with its fixtures;
smoke proves it agrees with *this* box, gateway and checkout — a different claim, and the
one that matters. It has already caught a bench parser that matched nothing, an ncu
diagnostic that named the wrong error, and a missing base ref that would have optimised a
tree containing no spark2_5 at all.

Wiring for the Spark target is in [`adapters/README.md`](adapters/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to enter a round, what wins, and what costs
you one.
