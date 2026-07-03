"""Path confinement helper so dev tools can't be steered outside the workspace.

A model choosing tool arguments is untrusted input in the same sense a web
request is — `../../etc/passwd` or an absolute path must not escape the
sandboxed root, so every dev tool routes its `path` argument through here.
"""

from __future__ import annotations

from pathlib import Path


def resolve_safe_path(root: str, relative_path: str) -> Path:
    """Join relative_path under root and resolve it; raise ValueError if the
    result escapes root (blocks ../.. traversal and absolute-path overrides)."""
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    candidate = (root_path / relative_path).resolve()
    if not candidate.is_relative_to(root_path):
        raise ValueError(f"path {relative_path!r} escapes workspace root {root_path}")
    return candidate
