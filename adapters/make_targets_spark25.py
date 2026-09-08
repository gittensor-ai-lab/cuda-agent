#!/usr/bin/env python3
"""Generate the target list for Spark-X2.5-4B (spark2_5).

Without a profiler the agent walks every kernel in the tree round-robin, and
most of them are for other models. Measured consequence: in 19 real candidates
it spent its whole budget editing `flash_decode_gqa8.cu` (16 Q / **2** KV,
head_dim **128**) and `flash_decode_global_hd512.cu` (head_dim **512**) --
neither of which spark2_5 ever dispatches. Every edit compiled, changed nothing,
and measured "not-faster" because it genuinely did nothing.

`ncu` would decide this at runtime, but GPU counters are admin-gated on every
vast.ai instance tested (ERR_NVGPUCTRPERM on two independent hosts), so the
selection has to be made statically.

spark2_5's shape, from its GGUF metadata:
    36 layers, dense (no MoE), hidden 2560, ffn 10240
    16 query heads / 4 KV heads  -> GQA-4
    head_dim 256
    3 sliding-window(512) layers : 1 full-attention layer

So:
  * attention kernels are included only when their guard matches hd256 + GQA-4;
  * MoE and vision kernels are excluded outright -- a dense text model runs
    neither;
  * GEMM/GEMV, quant and norm kernels are included, and at batch 1 a dense model
    spends most of its decode there rather than in attention.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Attention kernels whose dispatch guard does NOT match hd256 + GQA-4.
WRONG_SHAPE = {
    "flash_decode_gqa8.cu",          # 16 Q / 2 KV, head_dim 128
    "flash_decode_global_hd512.cu",  # head_dim 512 (Gemma global layers)
    "flash_decode_local_hd256.cu",   # head_dim 256 but GQA-2, not GQA-4
}

# Whole subsystems a dense text model never enters.
SKIP_DIRS = {"moe", "vision"}

# Ranked by where a dense 4B model at batch 1 actually spends decode time:
# the projections and FFN are GEMV-bound long before attention matters.
PRIORITY = {"gemm": 0, "quant": 1, "attention": 2, "fused": 3}


def symbols(repo: Path, rel: str, limit: int) -> list[str]:
    text = (repo / rel).read_text(encoding="utf-8", errors="surrogateescape")
    names: list[str] = []
    for m in re.finditer(r"__global__[^(){;]*?\b(\w+)\s*\(", text):
        if m.group(1) not in names:
            names.append(m.group(1))
    for m in re.finditer(r"^\s*(?:void|bool)\s+(launch_\w+)\s*\(", text, re.MULTILINE):
        if m.group(1) not in names:
            names.append(m.group(1))
    return names[:limit]


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/sparkinfer")
    per_file = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    targets, skipped = [], []
    for cu in sorted((repo / "kernels").rglob("*.cu")):
        rel = str(cu.relative_to(repo))
        parts = Path(rel).parts
        subsystem = parts[3] if len(parts) > 3 else ""

        if subsystem in SKIP_DIRS:
            skipped.append((rel, f"{subsystem}: not used by a dense text model"))
            continue
        if cu.name in WRONG_SHAPE:
            skipped.append((rel, "attention shape does not match hd256/GQA-4"))
            continue

        for sym in symbols(repo, rel, per_file):
            targets.append({"path": rel, "symbol": sym,
                            "_rank": PRIORITY.get(subsystem, 9)})

    targets.sort(key=lambda t: (t["_rank"], t["path"], t["symbol"]))
    for t in targets:
        t.pop("_rank")

    out = repo.parent / "targets-spark25.json" if len(sys.argv) < 4 else Path(sys.argv[3])
    out.write_text(json.dumps(targets, indent=2), encoding="utf-8")
    print(f"{len(targets)} targets -> {out}")
    print(f"skipped {len(skipped)} files:")
    for rel, why in skipped[:8]:
        print(f"  {rel}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
