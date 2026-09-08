"""Source windowing.

`gemv.cu` is over 4,000 lines; the proposer cannot hold it, and pasting a whole
file would spend the context budget on code the model was never going to touch.
So a target is one function plus a little surrounding air.

The extracted text is returned **verbatim** -- no line numbers, no ellipses, no
reformatting. Edits are anchored by exact string match, so anything decorative
added here would make every anchor the model writes fail to apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Rough cap on how much source to hand the model at once. The gateway's own
# limit is on output, but a bloated prompt still crowds out the history block
# that stops it repeating failed ideas.
DEFAULT_MAX_CHARS = 12_000


@dataclass(frozen=True)
class Target:
    """One optimisation target: a file, optionally narrowed to a symbol."""

    path: str
    symbol: str = ""

    @property
    def key(self) -> str:
        return f"{self.path}::{self.symbol}" if self.symbol else self.path

    def __str__(self) -> str:
        return self.key


def find_symbol_span(text: str, symbol: str) -> tuple[int, int] | None:
    """Character span of a C/C++/CUDA function definition, by brace matching.

    Returns None when the symbol is absent or only ever declared (no body), which
    the caller should treat as "fall back to a plain window".
    """
    for m in re.finditer(rf"\b{re.escape(symbol)}\s*\(", text):
        open_brace = _body_start(text, m.end())
        if open_brace is None:
            continue                       # a declaration or a call site, not a definition
        end = _match_brace(text, open_brace)
        if end is None:
            continue
        return _decl_start(text, m.start()), end + 1
    return None


def _body_start(text: str, after_paren: int) -> int | None:
    """Index of the '{' opening this function's body, if the next token is one."""
    depth = 1
    i = after_paren
    while i < len(text) and depth:          # walk to the matching ')'
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    # Between ')' and '{' only qualifiers/whitespace may appear. A ';' means this
    # was a declaration.
    while i < len(text) and text[i] not in "{;":
        i += 1
    return i if i < len(text) and text[i] == "{" else None


def _match_brace(text: str, open_idx: int) -> int | None:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _decl_start(text: str, name_idx: int) -> int:
    """Back up over the return type, template header and attached comments."""
    start = text.rfind("\n\n", 0, name_idx)
    line_start = text.rfind("\n", 0, name_idx) + 1
    # Keep at most a handful of lines of preamble (template<...>, __global__, the
    # doc comment) -- enough for the model to see the signature in context.
    probe = line_start
    for _ in range(8):
        prev = text.rfind("\n", 0, probe - 1) + 1
        if prev <= 0 or prev <= start:
            break
        line = text[prev:probe].strip()
        if not line or line.endswith(("}", ";")) and not line.startswith(("//", "template", "__")):
            break
        probe = prev
    return max(probe, 0)


def read_target(repo: Path, target: Target, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Verbatim source for a target, narrowed to its symbol where possible."""
    file_path = repo / target.path
    text = file_path.read_text(encoding="utf-8", errors="surrogateescape")

    if target.symbol:
        span = find_symbol_span(text, target.symbol)
        if span:
            start, end = span
            snippet = text[start:end]
            if len(snippet) <= max_chars:
                return snippet
            # A function too big to show whole: give the head, where the tiling
            # constants and launch configuration live.
            return snippet[:max_chars]

    return text[:max_chars]


def list_symbols(repo: Path, path: str, *, limit: int = 40) -> list[str]:
    """CUDA kernel and launcher names in a file, for target discovery."""
    text = (repo / path).read_text(encoding="utf-8", errors="surrogateescape")
    names: list[str] = []
    for m in re.finditer(r"__global__[^(){;]*?\b(\w+)\s*\(", text):
        if m.group(1) not in names:
            names.append(m.group(1))
    for m in re.finditer(r"^\s*(?:void|bool)\s+(launch_\w+)\s*\(", text, re.MULTILINE):
        if m.group(1) not in names:
            names.append(m.group(1))
    return names[:limit]
