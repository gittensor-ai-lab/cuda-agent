"""Client for the Gittensor Compute gateway (OpenAI-compatible).

The gateway has constraints that shape the whole agent, so they are enforced
here rather than trusted to the caller:

  * ``max_tokens`` is capped at 1024 -- the proposer must emit small, surgical
    edits, never whole-file rewrites.
  * Sampling parameters are ignored; every answer is a greedy decode. Identical
    state produces identical output, so diversity has to come from *structure*
    (different targets, different hypothesis families), not temperature.
  * ``n`` must be 1 and message content must be a plain string.
  * There is no queue: when no miner is READY the gateway answers 429. Backoff
    belongs here, so an agent is never scored on someone else's retry storm.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import httpx

from cuda_agent.config import GATEWAY_MAX_TOKENS, Settings

# Serving is priced so a 5090 flat out earns $0.70/card-hour, which the docs put
# at ~$0.69 per million output tokens on the current runtime. Used for the run
# report only -- nothing depends on it being exact.
USD_PER_M_OUTPUT_TOKENS = 0.69


class GatewayError(RuntimeError):
    """Non-retryable gateway failure."""


class GatewayExhausted(RuntimeError):
    """Retries exhausted -- usually sustained 429 (no READY serving capacity)."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    # Which miner UID served each request. Logged so a round can be audited for
    # fairness: an agent that ate 429s did not get the same deal as one that did
    # not, and you can only know that afterwards if you recorded it.
    served_uids: dict[str, int] = field(default_factory=dict)

    @property
    def est_cost_usd(self) -> float:
        return self.completion_tokens / 1_000_000 * USD_PER_M_OUTPUT_TOKENS

    def merge(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.requests += other.requests
        self.retries += other.retries
        self.rate_limited += other.rate_limited
        for uid, n in other.served_uids.items():
            self.served_uids[uid] = self.served_uids.get(uid, 0) + n


@dataclass
class Completion:
    text: str
    usage: Usage
    served_uid: str = ""
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        """True when the 1024-token cap cut the answer off mid-edit."""
        return self.finish_reason == "length"


class Gateway:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.base = settings.api_base.rstrip("/")
        self.usage = Usage()
        self._client = client or httpx.Client(timeout=settings.request_timeout_s)
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Gateway":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> Completion:
        body = self.build_body(messages, max_tokens)
        data = self._post_with_retry(body)
        return self._parse(data)

    # -- request construction -------------------------------------------------

    def build_body(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> dict:
        """Build a request body the gateway will accept.

        Split out from ``complete`` so the constraint handling is unit-testable
        without a live key.
        """
        want = max_tokens or self.settings.max_tokens
        return {
            "model": self.settings.model,
            "messages": [self._plain(m) for m in messages],
            # Clamped, not passed through: over the cap the gateway 400s.
            "max_tokens": max(1, min(want, GATEWAY_MAX_TOKENS)),
            "n": 1,
            "stream": False,
            # Sampling params are deliberately absent -- the gateway ignores them
            # and sending them implies a control the agent does not have.
        }

    @staticmethod
    def _plain(message: dict) -> dict[str, str]:
        """Coerce content to a plain string; the gateway rejects content blocks."""
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return {"role": str(message.get("role", "user")), "content": str(content)}

    # -- transport ------------------------------------------------------------

    def _post_with_retry(self, body: dict) -> dict:
        url = f"{self.base}/chat/completions"
        last: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                res = self._client.post(url, headers=self._headers, json=body)
            except httpx.HTTPError as exc:            # transport-level, worth a retry
                last = exc
            else:
                if res.status_code == 200:
                    self.usage.requests += 1
                    return res.json()
                if res.status_code == 429:
                    self.usage.rate_limited += 1
                    last = GatewayExhausted("429 no READY serving capacity")
                elif res.status_code in (500, 502, 503, 504):
                    last = GatewayError(f"gateway {res.status_code}: {res.text[:200]}")
                else:
                    # 401/400/etc. will not get better by waiting.
                    raise GatewayError(f"gateway {res.status_code}: {res.text[:500]}")

            if attempt < self.settings.max_retries:
                self.usage.retries += 1
                time.sleep(self._backoff(attempt))

        raise GatewayExhausted(f"exhausted {self.settings.max_retries} retries: {last}")

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped."""
        ceiling = min(self.settings.backoff_base_s * (2**attempt), self.settings.backoff_cap_s)
        return random.uniform(0.0, ceiling)

    # -- response parsing -----------------------------------------------------

    def _parse(self, data: dict) -> Completion:
        choices = data.get("choices") or [{}]
        choice = choices[0]
        text = str((choice.get("message") or {}).get("content") or "")
        finish = str(choice.get("finish_reason") or "")

        raw = data.get("usage") or {}
        served_uid = str(data.get("gittensor", {}).get("served_uid", "") or "")

        usage = Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            requests=1,
        )
        if served_uid:
            usage.served_uids[served_uid] = 1
        self.usage.merge(Usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            served_uids=dict(usage.served_uids),
        ))
        return Completion(text=text, usage=usage, served_uid=served_uid, finish_reason=finish)
