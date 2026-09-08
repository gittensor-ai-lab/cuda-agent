# Correctness-gate adapters

The agent's accuracy gate is a shell command whose output it reads with a regex, so a
target with its own correctness reference needs an adapter here — not a code change.

## Why there are two, for one model

Measured on an RTX 5090 against `feat/spark-x25-4b`:

| gate | cost | where it belongs |
|---|---|---|
| `spark25_gate.sh` — independent NumPy reference | **298 s** | arena scoring, once per submission |
| `spark25_diff_gate.sh` — differential vs baseline | **2–3 s** | agent inner loop, once per candidate |

A candidate iteration is ~80 s (79 s incremental build + 1 s bench). Putting the
independent reference inline makes it ~380 s and cuts a 3-hour round from ~130
candidates to ~28. So the loop uses the differential check — score the same token
stream on this build and on the round's baseline, require the distributions to agree —
and the independent reference runs once, on the submission that is actually being paid.

This mirrors `pr_qwen38_bot.py`, which is differential for the same reason: its
checkpoint has no external implementation that can read the same weights.

**What differential cannot do:** catch a bug already present in the baseline. It catches
newly introduced divergence, which is all an agent can introduce in one round.

## spark2_5 wiring

```bash
export CUDA_AGENT_BASE_REF=feat/spark-x25-4b      # spark2_5 is never merged to main
export SPARK25_GGUF=/models/Spark-X2.5-4B-Q4_K_M.gguf
export CUDA_AGENT_BUILD_CMD="cmake --build build -j32"
export CUDA_AGENT_BENCH_CMD="bash bench/scripts/bench.sh $SPARK25_GGUF --tokens 64"
export CUDA_AGENT_QUICK_BENCH_CMD="bash bench/scripts/bench.sh $SPARK25_GGUF --tokens 32"
export CUDA_AGENT_ACCURACY_CMD="bash adapters/spark25_diff_gate.sh"
```

`spark25_gate.sh` needs the **Q8_0** weights — the reference only implements Q8_0
dequantization, and Q4_K_M fails with `NotImplementedError: ggml type 14`. It also needs
a prompt longer than `SPARK25_PROMPT_LEN`; the script's default token stream is short, so
pass `--tokens` through if you raise it.

## Known rough edge

`spark25_diff_gate.sh` pins its baseline dump at round start. That is stricter than
comparing against the moving frontier, but drift can accumulate across several accepted
changes and eventually block legitimate progress. Re-dump on promotion for long rounds.
