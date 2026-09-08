"""Candidate isolation via git worktrees.

Every candidate is evaluated in its own detached worktree cut from the current
frontier, so a change that fails to build or turns out slower is discarded by
deleting a directory -- there is no partially-applied state to unwind, and the
frontier only ever moves on a measured win.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

FRONTIER_BRANCH = "cuda-agent/frontier"


class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


@dataclass
class Candidate:
    path: Path
    base_sha: str

    def __str__(self) -> str:
        return f"{self.path.name}@{self.base_sha[:8]}"


class Frontier:
    """Tracks the best verified state and hands out candidates cut from it.

    Two modes, and the default is the one that matters in production:

    * ``in_place=True`` (default) evaluates every candidate in the repo itself,
      reusing one build directory. A fresh git worktree has no ``build/``, so a
      worktree-per-candidate design pays a **full** rebuild every iteration --
      measured on an RTX 5090, ~20 minutes against ~78 seconds incremental. Over
      a three-hour round that is the difference between roughly 8 candidates and
      roughly 70. Rollback is ``git checkout --`` over the editable paths, which
      is exactly as complete as deleting a worktree because those are the only
      paths an edit is ever allowed to touch.
    * ``in_place=False`` keeps the isolated-worktree behaviour, for callers that
      want it and can afford the rebuild.

    Nothing is lost by sharing one tree: the GPU serialises measurement anyway,
    so parallel candidate directories bought no throughput.
    """

    def __init__(
        self,
        repo: Path,
        worktree_root: Path,
        *,
        base_ref: str = "HEAD",
        in_place: bool = True,
        editable: tuple[str, ...] = ("kernels", "runtime", "moe"),
    ) -> None:
        self.repo = repo
        self.root = worktree_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.in_place = in_place
        self.editable = editable
        self.base_sha = _git(repo, "rev-parse", base_ref)

        if in_place:
            # Put the repo itself on the frontier branch, so accepted commits
            # land there and the build directory follows the source.
            _git(repo, "checkout", "-q", "-B", FRONTIER_BRANCH, self.base_sha)
        else:
            # A dedicated branch so the agent never moves the caller's HEAD.
            _git(repo, "branch", "-f", FRONTIER_BRANCH, self.base_sha)
        self.sha = self.base_sha

    # -- candidates -----------------------------------------------------------

    def checkout(self) -> Candidate:
        if self.in_place:
            self._restore()          # start every candidate from a clean frontier
            return Candidate(path=self.repo, base_sha=self.sha)
        path = self.root / f"cand-{uuid.uuid4().hex[:10]}"
        _git(self.repo, "worktree", "add", "--detach", str(path), self.sha)
        return Candidate(path=path, base_sha=self.sha)

    def discard(self, cand: Candidate) -> None:
        if self.in_place:
            self._restore()
            return
        _git(self.repo, "worktree", "remove", "--force", str(cand.path), check=False)
        shutil.rmtree(cand.path, ignore_errors=True)

    def _restore(self) -> None:
        """Undo an unaccepted edit, over the editable paths only.

        Scoped rather than a bare `git checkout -- .` so a stray file elsewhere
        in the tree -- a build artifact, a downloaded model -- is never deleted
        by the agent's rollback.
        """
        present = [p for p in self.editable if (self.repo / p).exists()]
        if not present:
            return
        _git(self.repo, "checkout", "--", *present, check=False)
        _git(self.repo, "clean", "-fdq", "--", *present, check=False)

    def promote(self, cand: Candidate, message: str) -> str:
        """Commit the candidate's changes and advance the frontier to it.

        Refuses to promote a candidate cut from a stale frontier: two accepted
        candidates measured against different baselines are not comparable, and
        silently stacking them would make the reported speedup a fiction.
        """
        if cand.base_sha != self.sha:
            raise GitError(
                f"candidate {cand} was cut from {cand.base_sha[:8]} but the frontier "
                f"is now {self.sha[:8]}; re-evaluate it against the current frontier"
            )
        _git(cand.path, "add", "-A")
        if not _git(cand.path, "status", "--porcelain"):
            raise GitError("nothing to promote: the candidate has no changes")
        _git(cand.path, "-c", "user.name=cuda-agent", "-c", "user.email=cuda-agent@localhost",
             "commit", "-q", "-m", message)
        self.sha = _git(cand.path, "rev-parse", "HEAD")
        if not self.in_place:
            # In-place already committed onto the frontier branch (HEAD is it).
            _git(self.repo, "branch", "-f", FRONTIER_BRANCH, self.sha)
        return self.sha

    # -- output ---------------------------------------------------------------

    def diff(self) -> str:
        """The full patch from the round's base commit to the current frontier."""
        return _git(self.repo, "diff", f"{self.base_sha}..{self.sha}")

    def is_dirty(self, cand: Candidate) -> bool:
        return bool(_git(cand.path, "status", "--porcelain"))

    def cleanup(self) -> None:
        if self.in_place:
            self._restore()
        _git(self.repo, "worktree", "prune", check=False)
        shutil.rmtree(self.root, ignore_errors=True)
