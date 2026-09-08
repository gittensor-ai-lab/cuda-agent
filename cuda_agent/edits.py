"""Strict search/replace edit format.

The gateway caps output at 1024 tokens, so the proposer cannot rewrite a file --
it emits small anchored edits instead:

    <edit path="kernels/csrc/cuda/gemm/gemv.cu">
    <<<<<<< SEARCH
    constexpr int TILE_N = 64;
    =======
    constexpr int TILE_N = 128;
    >>>>>>> REPLACE
    </edit>

Every rejection carries a message written *for the model*: a malformed edit that
is bounced with a specific reason costs one cheap round trip, while one that is
applied blindly costs a full build-and-benchmark cycle. Validation is therefore
deliberately strict -- an anchor that matches twice is ambiguous and rejected
rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_EDIT_RE = re.compile(
    r"<edit\s+path=[\"'](?P<path>[^\"']+)[\"']\s*>\s*"
    r"<{7}\s*SEARCH\s*\n(?P<search>.*?)\n?={7}\s*\n(?P<replace>.*?)\n?>{7}\s*REPLACE\s*"
    r"(?:</edit>|\Z)",
    re.DOTALL,
)


class EditError(ValueError):
    """Rejected edit. The message is intended to be shown back to the model."""


@dataclass(frozen=True)
class Edit:
    path: str
    search: str
    replace: str

    def describe(self) -> str:
        return f"{self.path}: {len(self.search)}B -> {len(self.replace)}B"


def parse_edits(text: str) -> list[Edit]:
    """Extract every well-formed edit block from a completion.

    Returns an empty list when the model produced prose instead of edits; the
    caller decides whether that is a retry or a wasted round.
    """
    edits: list[Edit] = []
    for m in _EDIT_RE.finditer(text or ""):
        edits.append(
            Edit(
                path=m.group("path").strip(),
                search=m.group("search"),
                replace=m.group("replace"),
            )
        )
    return edits


def looks_like_attempted_edit(text: str) -> bool:
    """True when the model tried to emit an edit but the block did not parse.

    Distinguishes 'the model answered in prose' from 'the model emitted a
    malformed or truncated edit' -- the second is worth an explicit correction,
    the first is worth re-prompting.
    """
    return "<edit" in (text or "") or "<<<<<<< SEARCH" in (text or "")


def check_path(path: str, *, allowed: tuple[str, ...], denied: tuple[str, ...]) -> None:
    """Reject a path outside the contributor surface, before any file is opened."""
    raw = path.strip()
    if not raw:
        raise EditError("empty path")
    # Check absoluteness BEFORE normalising: stripping "./" would also eat the
    # leading slash and quietly turn /etc/passwd into a relative path.
    if raw.startswith("/") or ".." in Path(raw).parts:
        raise EditError(f"path escapes the repo: {path!r}")
    norm = raw[2:] if raw.startswith("./") else raw
    if not norm:
        raise EditError("empty path")
    for bad in denied:
        if norm == bad.rstrip("/") or norm.startswith(bad):
            raise EditError(
                f"{norm} is maintainer-owned (the measuring instrument). "
                f"Edit the runtime or the kernels instead."
            )
    if not any(norm == ok.rstrip("/") or norm.startswith(ok) for ok in allowed):
        raise EditError(
            f"{norm} is outside the editable surface. Allowed prefixes: {', '.join(allowed)}"
        )


def apply_edit(repo: Path, edit: Edit, *, allowed: tuple[str, ...], denied: tuple[str, ...]) -> None:
    """Apply one edit in place, or raise EditError explaining why it did not."""
    check_path(edit.path, allowed=allowed, denied=denied)

    target = (repo / edit.path).resolve()
    root = repo.resolve()
    if root != target and root not in target.parents:
        raise EditError(f"path escapes the repo: {edit.path!r}")
    if not target.is_file():
        raise EditError(f"no such file: {edit.path}")

    original = target.read_text(encoding="utf-8", errors="surrogateescape")

    if not edit.search:
        raise EditError(f"{edit.path}: empty SEARCH block")
    if edit.search == edit.replace:
        raise EditError(f"{edit.path}: SEARCH and REPLACE are identical -- no change")

    hits = original.count(edit.search)
    if hits == 0:
        raise EditError(
            f"{edit.path}: SEARCH block not found. It must match the file byte for byte, "
            f"including indentation."
        )
    if hits > 1:
        raise EditError(
            f"{edit.path}: SEARCH block matches {hits} places and is ambiguous. "
            f"Include more surrounding context so it matches exactly once."
        )

    target.write_text(original.replace(edit.search, edit.replace, 1), encoding="utf-8", errors="surrogateescape")


def apply_all(repo: Path, edits: list[Edit], *, allowed: tuple[str, ...], denied: tuple[str, ...]) -> list[str]:
    """Apply edits in order. Raises on the first failure, having applied none.

    Two passes: validate everything against a scratch copy of the file contents
    first, so a batch that fails halfway does not leave the worktree in a state
    no one intended.
    """
    if not edits:
        raise EditError("no edits to apply")

    staged: dict[Path, str] = {}
    for edit in edits:
        check_path(edit.path, allowed=allowed, denied=denied)
        target = (repo / edit.path).resolve()
        root = repo.resolve()
        if root != target and root not in target.parents:
            raise EditError(f"path escapes the repo: {edit.path!r}")
        if not target.is_file():
            raise EditError(f"no such file: {edit.path}")

        current = staged.get(target)
        if current is None:
            current = target.read_text(encoding="utf-8", errors="surrogateescape")

        if not edit.search:
            raise EditError(f"{edit.path}: empty SEARCH block")
        if edit.search == edit.replace:
            raise EditError(f"{edit.path}: SEARCH and REPLACE are identical -- no change")
        hits = current.count(edit.search)
        if hits == 0:
            raise EditError(
                f"{edit.path}: SEARCH block not found. It must match the file byte for byte, "
                f"including indentation."
            )
        if hits > 1:
            raise EditError(
                f"{edit.path}: SEARCH block matches {hits} places and is ambiguous. "
                f"Include more surrounding context so it matches exactly once."
            )
        staged[target] = current.replace(edit.search, edit.replace, 1)

    for target, content in staged.items():
        target.write_text(content, encoding="utf-8", errors="surrogateescape")
    return [e.path for e in edits]
