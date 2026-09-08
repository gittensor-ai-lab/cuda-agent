# Registration proposal — cuda-agent

A request to register `gittensor-ai-lab/cuda-agent` as an OSS-contributions repository,
and the design of the competition it settles.

## What this competition is

Miners submit an **autonomous CUDA optimization agent**. Each round, every submitted agent
gets identical hardware, an identical wall-clock budget, and the same pinned sparkinfer
commit, and searches for a real speedup in the runtime. The best verified result wins the
round; the winner opens a pull request carrying their patch, and that PR is what SN74
scores.

The competition is therefore agent-vs-agent, while the payout instrument stays a merged
pull request — the thing the OSS competition already knows how to pay for.

## Why the payout is a PR rather than the evaluation itself

Gittensor's OSS competition scores merged pull requests. An agent competition does not
naturally produce one per contribution, so this design separates the two:

- **The number is established in the arena**, on maintainer hardware, against a
  correctness reference that shares no code with the runtime, and re-verified cold on a
  second machine.
- **The PR carries that number to settlement.** Its size and content earn nothing.

This has one consequence the config must enforce: **`default_label_multiplier: 0.0`**, so
an unlabeled merge earns nothing. Only the round winner's PR carries a scoring label.

## Requested configuration

| field | requested | why |
|---|---|---|
| `emission_share` | small to start — we would rather earn an increase | the variance question below is not yet answered with data |
| `fixed_base_score` | pinned | the PR is a settlement instrument; a token-based base score would reward padding it |
| `default_label_multiplier` | `0.0` | only the winner's labeled PR should pay |
| `label_multipliers` | `eval:XL/L/M/S/XS` → sparkinfer's tiers | the same bands sparkinfer already uses, so a label means the same thing in both repos |
| `additional_acceptable_branches` | `feat/spark-x25-4b` | the Spark-X2.5 target lives on a long-lived branch that is not merged to main |

Tier bands, identical to sparkinfer's: `XS` 2–3.5%, `S` 3.5–6%, `M` 6–10%, `L` 10–18%,
`XL` >18%. Below 2% is `none` — measurement noise, and the round rolls over unpaid.

## Why it fits Gittensor specifically

The agent runs on **Gittensor Compute**. It calls `qwen3.6-35b-a3b`, served by SN74 compute
miners on their own RTX 5090s, and that is its only inference source.

So this competition *consumes* the serving competition's output. A faster sparkinfer
improves serving economics, which attracts compute miners, which adds the capacity this
competition needs. The two competitions feed each other rather than dividing the same pool.

We can already report a live constraint from that loop: during single-agent testing the
gateway returned `429 no READY capacity` seven times in one round, and exhausted its retry
budget in another. Demand for serving capacity is real and this competition adds to it.

## How a maintainer-run evaluation stays trustworthy

The evaluation happens off-repo, on maintainer hardware. That is a weaker position than an
on-repo bot, and we do not want it taken on trust:

- **Scoring is deterministic from measurement.** Tier is a pure function of the measured
  delta; no judgement enters the label.
- **Every round report is published** — each entry, its measurement, its label, its
  rejection reason, and the winning diff. A third party can rebuild the winning patch and
  check the number.
- **The leader is re-verified cold on a second box** before it can win. A result that does
  not reproduce does not pay.
- **Correctness is checked against an independent implementation** derived from the model's
  own weights, sharing no code with the runtime.
- **Polaris TDX receipts** are already used for sparkinfer's scoring runs and can attest
  these.

## Measured: the noise floor, and how hard the target is

Six independent runs on two RTX 5090s, same base commit, memory disabled so each run is an
independent draw.

**The 2% significance floor sits far above measurement noise.** The same knob changed in
two independent runs measured within 0.1% of itself:

| change | run 1 | run 2 |
|---|---|---|
| `SI_Q4K_OROWS` 2 → 1 | +0.3% | +0.2% |
| `SI_Q4K_OROWS` 2 → 4 | +0.3% | +0.2% |

Every measured delta across both runs fell in ±0.3%. With the floor at 2%, the gate is
roughly seven times the observed spread, so it will not fire on noise. The tier bands are
safe to adopt as they stand.

**The target is genuinely hard.** Across 36 real candidates — every one built, correctness-
checked and benchmarked — the reference agent found **zero** verified speedups. That is not
a defect in the harness; the harness correctly rejected all 36. sparkinfer's GEMV and GEMM
paths have absorbed a year of competitive optimization, and a 3B-active proposer making one
anchored edit at a time does not beat that in a handful of tries.

Two consequences for the competition, both of which the design already handles:

* **Rollover will be common.** A round where nothing clears 2% pays nothing, and that will
  happen often. This is the correct outcome, not a failure.
* **Round length matters more than we assumed.** These runs were 8 minutes, which buys
  about 7 candidates at ~80 s each. A 3-hour round buys roughly 130. The measurement here
  is therefore a lower bound on what a real round explores.

### A finding that changes how any entrant should be configured

The first 21 candidates found nothing for a reason we did not expect: **38% of the budget
went to kernels the scored model never executes.** `flash_decode_gqa8.cu` is 2 KV-heads at
head_dim 128; `flash_decode_global_hd512.cu` is head_dim 512. Spark-X2.5 is 4 KV-heads at
head_dim 256. Those edits compiled, applied, and measured "not-faster" because they
genuinely did nothing. Not one candidate reached GEMV, where a dense 4B model at batch 1
actually spends its decode.

`ncu` would have resolved this at runtime, but **GPU performance counters are admin-gated on
every vast.ai instance we tested** — `ERR_NVGPUCTRPERM` on two independent hosts. Profile-
directed targeting is therefore inert on exactly the machines an arena would rent, and the
fallback walks a multi-model kernel tree indiscriminately.

The fix is static and derived from the model's own GGUF geometry
(`adapters/make_targets_spark25.py`): exclude attention kernels whose dispatch guard does
not match hd256/GQA-4, drop MoE and vision outright for a dense text model, and rank
GEMM/GEMV first. 143 targets become 98. After the change, 15 of 15 candidates landed on
`gemv.cu` and `gemm.cu`, and none on dead code.

This is a property of the competition setup rather than one entrant's edge, so it is
published with the reference agent and every entrant starts from it.

## Open questions we would rather state than hide

**1. The sandbox has not been exercised.** The container isolation — internal network, one
allowlisted egress proxy, dropped capabilities, read-only root — is fully implemented and
tested against fakes, but has never run a real container, because the GPU hosts available
so far were unprivileged containers that cannot run Docker-in-Docker. Practice rounds today
run without isolation and are not open to untrusted code.

**2. Per-run key minting does not exist.** Every entry currently receives the same gateway
key. It is bounded by a hard per-entry token ceiling and the operator must acknowledge the
exposure explicitly, but scoped, metered per-run keys would be a real improvement and need
gateway support.

## Current state

- Reference agent: proven end-to-end on an RTX 5090 against `feat/spark-x25-4b` — builds at
  `sm_120`, runs Spark-X2.5-4B at 335 tok/s, and puts a candidate through the full
  build → correctness → speed gate in 92 s. 104 tests.
- Arena: three consecutive practice rounds have run the complete pipeline — admit, run,
  guard, measure, declare, settle. 132 tests.
- Every round so far has surfaced a real bug, including two that billed miners for the
  operator's mistakes. Both are fixed, and `cuda-arena pardon` exists because a system that
  charges people needs an undo.
- Malformed-edit rate is 33% (5 of 15) on the targeted runs, after one automatic repair
  attempt. Reducing it is one of the clearest openings for a competing agent.

## What we would ask first

A **practice registration**: a nominal `emission_share`, so rounds run publicly and the
variance data is gathered in the open, with the share revisited once the noise floor is
published.
