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

## Open questions we would rather state than hide

**1. Variance is not yet quantified.** Agent outcomes are noisier than kernel-patch
outcomes. The same agent found two accepted changes in one run and none in another, and
gateway rate-limiting is outside any entrant's control. Before requesting a meaningful
`emission_share` we intend to run repeat rounds and publish a measured noise floor. Setting
tier bands before measuring the noise is how a competition ends up paying for luck.

**2. The sandbox has not been exercised.** The container isolation — internal network, one
allowlisted egress proxy, dropped capabilities, read-only root — is fully implemented and
tested against fakes, but has never run a real container, because the GPU hosts available
so far were unprivileged containers that cannot run Docker-in-Docker. Practice rounds today
run without isolation and are not open to untrusted code.

**3. Per-run key minting does not exist.** Every entry currently receives the same gateway
key. It is bounded by a hard per-entry token ceiling and the operator must acknowledge the
exposure explicitly, but scoped, metered per-run keys would be a real improvement and need
gateway support.

## Current state

- Reference agent: proven end-to-end on an RTX 5090 against `feat/spark-x25-4b` — builds at
  `sm_120`, runs Spark-X2.5-4B at 335 tok/s, and puts a candidate through the full
  build → correctness → speed gate in 92 s. 103 tests.
- Arena: three consecutive practice rounds have run the complete pipeline — admit, run,
  guard, measure, declare, settle. 131 tests.
- Every round so far has surfaced a real bug, including two that billed miners for the
  operator's mistakes. Both are fixed, and `cuda-arena pardon` exists because a system that
  charges people needs an undo.

## What we would ask first

A **practice registration**: a nominal `emission_share`, so rounds run publicly and the
variance data is gathered in the open, with the share revisited once the noise floor is
published.
