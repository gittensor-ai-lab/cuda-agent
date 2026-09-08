"""Proposer prompts.

Written for a 3B-active model with a 1024-token answer budget, which means:
one target, one hypothesis family, one small edit per turn. Asking it to "find
optimisations" produces prose; asking it to apply a named technique to a named
function produces an edit.
"""

from __future__ import annotations

from cuda_agent.families import Family
from cuda_agent.source import Target

SYSTEM = """\
You are a CUDA performance engineer optimising the sparkinfer inference runtime \
for NVIDIA Blackwell (sm_120, RTX 5090).

You make ONE small, surgical change per turn. You do not refactor, rename, or \
tidy. Correctness is gated against a llama.cpp reference: a change that alters \
model output is rejected no matter how fast it is.

Answer with edit blocks and nothing else. No explanation before or after.

Format, exactly:

<edit path="relative/path/from/repo/root.cu">
<<<<<<< SEARCH
the exact existing text, byte for byte, including indentation
=======
the replacement text
>>>>>>> REPLACE
</edit>

Rules:
- The SEARCH text must appear EXACTLY ONCE in the file. Include enough \
surrounding lines to be unambiguous.
- Copy the existing text character for character. Do not reformat it.
- Keep the edit small. You have a hard 1024-token answer limit; a truncated \
edit is discarded.
- Only edit kernels/, runtime/ and moe/.\
"""


def build_proposal(
    *,
    target: Target,
    family: Family,
    source: str,
    history: str,
    baseline: dict[str, float] | None = None,
    profile: str = "",
) -> list[dict[str, str]]:
    """One proposer turn: this function, this technique, what has already failed."""
    parts = [f"File: {target.path}"]
    if target.symbol:
        parts.append(f"Function: {target.symbol}")

    if baseline:
        rows = ", ".join(f"{k}={v:.1f} tok/s" for k, v in sorted(baseline.items()))
        parts.append(f"\nCurrent measured decode: {rows}")

    if profile:
        parts.append(f"\nProfiler:\n{profile}")

    parts.append(f"\nTechnique to apply: {family.name}\n{family.hint}")
    parts.append(f"\n{history}")
    parts.append(f"\nSource:\n\n{source}")
    parts.append(
        "\nApply the technique above to this code. Emit one edit block. "
        "If the technique genuinely does not apply here, emit no edit block."
    )

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_repair(previous: str, error: str) -> list[dict[str, str]]:
    """Second chance after a malformed or unapplicable edit.

    Cheap compared with a build, so worth exactly one retry: the common failures
    (anchor reformatted, anchor ambiguous, answer truncated) are all correctable
    from the error text alone.
    """
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Your previous edit was rejected.\n\nError: {error}\n\n"
                f"Your previous answer was:\n{previous[:2000]}\n\n"
                "Emit a corrected edit block. Copy the SEARCH text exactly from the "
                "source you were shown."
            ),
        },
    ]
