"""Runtime configuration, read from the environment.

Deliberately dependency-free: the agent runs inside an arena container with no
egress except the inference gateway, so the fewer moving parts the better.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Gittensor Compute serves one blessed model and caps output at 1024 tokens per
# request; sampling parameters are ignored (every answer is a greedy decode).
# See https://docs.gittensor.io/compute-serving.
GATEWAY_MAX_TOKENS = 1024
GATEWAY_DEFAULT_MODEL = "qwen3.6-35b-a3b"
GATEWAY_DEFAULT_BASE = "https://gt.venturalabs.ai/v1"

# Paths a candidate patch is allowed to touch. Mirrors sparkinfer's contributor
# surface: everything else is maintainer-owned (eval harness, bench scripts,
# scoring config, CI) and a diff touching it is rejected before it is ever built.
DEFAULT_ALLOWED_PATHS = ("kernels/", "runtime/", "moe/")

# Carve-outs *inside* the contributor surface. These are the measuring
# instruments -- a number produced by a modified instrument means nothing.
DEFAULT_DENIED_PATHS = (
    "runtime/examples/dspark_tau_check.cpp",
    "bench/scripts/",
    "eval/",
    ".github/",
    ".gittensor/",
)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    """Everything the agent needs to run one arena round."""

    # --- inference gateway (the agent's brain) ---
    api_base: str = field(default_factory=lambda: _env_str("GITTENSOR_API_BASE", GATEWAY_DEFAULT_BASE))
    api_key: str = field(default_factory=lambda: _env_str("GITTENSOR_API_KEY", ""))
    model: str = field(default_factory=lambda: _env_str("GITTENSOR_MODEL", GATEWAY_DEFAULT_MODEL))

    # The gateway hard-caps this; clamping locally turns a 400 into a warning.
    max_tokens: int = field(default_factory=lambda: min(_env_int("CUDA_AGENT_MAX_TOKENS", 1024), GATEWAY_MAX_TOKENS))
    request_timeout_s: float = field(default_factory=lambda: _env_float("CUDA_AGENT_REQUEST_TIMEOUT", 180.0))

    # The gateway answers 429 rather than queueing when no miner is READY, so
    # backoff lives here in the harness -- an agent must not be penalised for
    # someone else's retry logic.
    max_retries: int = field(default_factory=lambda: _env_int("CUDA_AGENT_MAX_RETRIES", 6))
    backoff_base_s: float = field(default_factory=lambda: _env_float("CUDA_AGENT_BACKOFF_BASE", 2.0))
    backoff_cap_s: float = field(default_factory=lambda: _env_float("CUDA_AGENT_BACKOFF_CAP", 60.0))

    # --- budgets (the arena enforces its own wall clock; these are self-limits) ---
    wall_clock_s: float = field(default_factory=lambda: _env_float("CUDA_AGENT_WALL_CLOCK", 3 * 3600.0))
    max_completion_tokens: int = field(default_factory=lambda: _env_int("CUDA_AGENT_TOKEN_BUDGET", 0))
    pipeline_depth: int = field(default_factory=lambda: _env_int("CUDA_AGENT_PIPELINE_DEPTH", 2))

    # --- target repo ---
    repo: Path = field(default_factory=lambda: Path(_env_str("CUDA_AGENT_REPO", "/work/sparkinfer")))
    # The ref the round competes on. spark2_5 lives on feat/spark-x25-4b and is
    # never merged to main, so defaulting to HEAD would silently optimise a tree
    # that does not contain the target at all.
    base_ref: str = field(default_factory=lambda: _env_str("CUDA_AGENT_BASE_REF", "HEAD"))
    out_dir: Path = field(default_factory=lambda: Path(_env_str("CUDA_AGENT_OUT", "/out")))
    worktree_root: Path = field(default_factory=lambda: Path(_env_str("CUDA_AGENT_WORKTREES", "/tmp/cuda-agent-wt")))
    # Cross-round memory. Deliberately outside out_dir: out_dir is the round's
    # submission, this is state the arena carries between rounds. Whether an
    # agent gets a persistent volume at all is the arena's call.
    memory_path: Path = field(default_factory=lambda: Path(_env_str("CUDA_AGENT_MEMORY", "/state/memory.json")))

    allowed_paths: tuple[str, ...] = DEFAULT_ALLOWED_PATHS
    denied_paths: tuple[str, ...] = DEFAULT_DENIED_PATHS

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("GITTENSOR_API_KEY is required")
        if not self.repo.exists():
            raise ValueError(f"target repo not found: {self.repo}")


def get_settings() -> Settings:
    return Settings()
