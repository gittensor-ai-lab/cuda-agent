#!/usr/bin/env bash
# Differential correctness gate for spark2_5 -- the inner-loop check.
#
# The independent reference (spark25_ref_check.py) is the right gate for a
# *submission*, but it costs 298 s on an RTX 5090 against an 80 s build, so it
# cannot run per candidate. This is pr_qwen38_bot.py's methodology instead:
# score the SAME token stream on this build and on the round's baseline build,
# and require the two distributions to agree. It catches any change that alters
# the model's numerics -- which is what a per-candidate gate is for -- and
# deliberately cannot catch a bug already present in the baseline.
set -uo pipefail
GGUF="${SPARK25_GGUF:?SPARK25_GGUF is required}"
REF="${SPARK25_REF_DUMP:-/workspace/agent-ref-score.txt}"
TOPK="${SPARK25_TOPK:-64}"
BIN=./build/runtime/qwen3_gguf_score

# A fixed token stream. Any valid ids work for a differential comparison so long
# as both sides see the same ones.
IDS="9707 11 847 829 374 1234 5678 91011 2048 4096 777 8192 31 42 9001 1600
     2401 88 4321 555 12345 6789 101 2020 33333 4444 55 66666 7777 888 99999 1010"

dump() { $BIN "$GGUF" "$TOPK" $IDS 2>/dev/null; }

if [ ! -s "$REF" ]; then
  # First call of the round: this build *is* the baseline. Record it and pass.
  dump > "$REF" || { echo "top1=0 kl=99  (baseline score dump failed)"; exit 1; }
  grep -q '^S ' "$REF" || { echo "top1=0 kl=99  (baseline dump had no rows)"; exit 1; }
  echo "top1=1.000000 kl=0.000000  (baseline recorded: $(grep -c '^S ' "$REF") positions)"
  exit 0
fi

CAND="$(mktemp)"; trap 'rm -f "$CAND"' EXIT
dump > "$CAND" || { echo "top1=0 kl=99  (candidate score dump failed)"; exit 1; }
grep -q '^S ' "$CAND" || { echo "top1=0 kl=99  (candidate dump had no rows)"; exit 1; }

out="$(python3 bench/scripts/accuracy_compare_pair.py "$CAND" "$REF" --metric-label SPARK25 2>&1)"
echo "$out"
grep -qE 'top1=[0-9]' <<<"$out" || { echo "top1=0 kl=99  (comparator produced no verdict)"; exit 1; }
