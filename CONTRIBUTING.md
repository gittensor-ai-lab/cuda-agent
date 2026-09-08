# Contributing — the SN74 agent competition

You are not paid for a pull request here. You are paid for **an agent that finds a
verified speedup in sparkinfer**, measured on the maintainer's hardware. The pull request
comes afterwards, and only from the winner.

## The round

1. **Submit.** You enter an agent, not a patch. One entry per identity per round.
2. **Evaluate.** Every entry runs on identical hardware, for the same wall clock, from the
   same pinned sparkinfer commit. Your agent searches for a real optimisation.
3. **Verify.** The leading patch is rebuilt and re-measured **cold, on a different box**.
   A gain that only exists on one machine is not a gain.
4. **Publish.** The round report is public: every entry, its measurement, its label, and
   the winning diff.
5. **Settle.** The winner is asked to open a PR against sparkinfer with their patch. That
   PR carries the round's label, and the label is what pays.

**The PR is a settlement instrument.** Its size, its prose and its commit count earn
nothing — the number was established in step 3. Padding it is pointless; an unlabeled
merge earns nothing at all.

## What wins

The best **verified** speedup that clears the significance floor, using sparkinfer's own
tiers:

| label | gain over the frontier |
|---|---|
| `eval:XL` | ≥ 18% |
| `eval:L` | 10–18% |
| `eval:M` | 6–10% |
| `eval:S` | 3.5–6% |
| `eval:XS` | 2–3.5% |
| `eval:none` | below 2% — within measurement noise |

If nothing clears 2%, **the round pays nothing and rolls over.** Ties break on submission
time, then identity — never at random, because the payout is winner-take-all.

## What does not win

- A speedup the arena cannot reproduce cold on a second box.
- A change that alters the model's output. Correctness is checked before speed, and a
  failing correctness gate rejects a patch regardless of how fast it is.
- A patch touching anything outside `kernels/`, `runtime/` and `moe/`. The evaluation
  harness, the bench scripts and the scoring config are maintainer-owned — a number
  produced by a modified instrument is not a number.
- Self-reported measurements. Your agent's own benchmarks steer its search; they are
  never the score.

## What an entry costs

Nothing, if you compete honestly. The entry cost is denominated in **eligibility**, not
tokens, because that is the only currency the competition actually controls.

| outcome | cost |
|---|---|
| qualified — **won or lost** | none, and it forgives one earlier strike |
| searched and found nothing | none |
| crashed, timed out, or would not build | 1 strike |
| touched a maintainer-owned path, oversized or binary patch | 2 strikes |
| attempted egress, or escaped the sandbox | immediate denylist |

Strikes cost rounds sat out: **1 → 3 → 10 → denylist**.

A miner who enters every day and never wins pays **zero, forever**. That is deliberate,
and it is the opposite of an entry fee, which taxes the honest single entry hardest.

Searching your whole budget and finding nothing is an *honest empty round*, not a failure.
Only an agent that produces no output at all is treated as hung.

## One entry per identity per round

This is the cap on variance-farming — submitting one decent agent ten times to keep the
best draw. Winner-take-all invites exactly that, and no absolute scoring rule prevents it.

Sybil resistance is inherited rather than rebuilt: entry binds to the GitHub identity SN74
already pays, so an additional identity costs whatever registration costs.

## Anti-gaming

- **Copycatting.** Re-submitting another entrant's agent, or an already-merged
  optimisation, is caught by diff-containment fingerprinting. First strike freezes your
  evaluations; a second blocks the account.
- **Sybil / duplicate-account farming** is blocked outright.
- **Tuning to the harness.** Correctness runs against an implementation that shares no
  code with the runtime, and the scored run uses held-out inputs. An agent that behaves
  differently when it detects a benchmark is flagged for review.
- **No override.** There is no way to force a result through the gate, not even for a
  maintainer.

## What runs your agent

Your agent gets an exclusive GPU, a scoped budget, and a Gittensor Compute key — you
supply none of these. It runs sealed: no network egress except the inference gateway, all
capabilities dropped, a read-only root filesystem, and hard memory and PID caps.

You need neither a GPU nor an API key to *enter*. You will want both to *develop*, and
that is the real barrier — a rented 5090 is about $0.50/hour and a full round of inference
costs under a cent.

## Building an agent

Fork this repo. The reference agent is a working entry, not a toy — the sweep alone finds
75 tunable constants without spending a token, and is the bar any submission should clear.

Places worth attacking:

- **Search strategy.** The bandit explores techniques per kernel; with ~20 proposer
  iterations a round, starting in the right part of the space is most of the game.
- **Context construction.** The model sees one kernel function. What else belongs in the
  prompt is an open question.
- **Cross-round memory.** In a daily competition, remembering what failed yesterday is
  compounding advantage. In a real round it already skipped five known-bad edits without
  spending a build.
- **Structural diversity.** The gateway decodes greedily, so identical state gives an
  identical answer. Diversity has to come from what you ask, not from sampling.

Run `cuda-agent --smoke` first. It checks your wiring against the real box, the real
gateway and the real checkout, and reports what it actually saw rather than a bare failure.
