#!/usr/bin/env bash
# Adapter: express the spark2_5 correctness check in the shape the agent's gate reads.
#
# sparkinfer's spark2_5 reference is spark25_ref_check.py -- an independent NumPy
# forward pass built from the GGUF's own bytes -- and it reports greedy agreement
# as "N/M", not the "top1=/kl=" that accuracy.sh emits for the llama.cpp-backed
# targets. The agent does not need to know that: it reads whatever regex it is
# given, so a target with a different reference just needs an adapter.
set -uo pipefail
GGUF="${SPARK25_GGUF:?SPARK25_GGUF is required}"
PROMPT_LEN="${SPARK25_PROMPT_LEN:-8}"

out="$(python3 bench/scripts/spark25_ref_check.py "$GGUF" --verify-greedy "$PROMPT_LEN" 2>&1)"
rc=$?
echo "$out"
[ $rc -ne 0 ] && { echo "top1=0 kl=99  (reference check failed)"; exit 1; }

# "greedy agreement over the 24 continuation tokens: 23/24"
read -r agree total <<<"$(sed -n 's/.*continuation tokens: \([0-9]*\)\/\([0-9]*\).*/\1 \2/p' <<<"$out" | tail -1)"
if [ -z "${total:-}" ] || [ "${total:-0}" -eq 0 ]; then
  # No numbers parsed is a failure, never a pass -- that is how a gate stops gating.
  echo "top1=0 kl=99  (no agreement line in reference output)"; exit 1
fi
python3 -c "print(f'top1={$agree/$total:.6f} kl=0.000000  ({$agree}/{$total} greedy steps)')"
