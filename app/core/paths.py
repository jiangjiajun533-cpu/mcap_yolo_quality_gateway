"""
Resolve MCAP / output paths for API and dashboard (cwd-independent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def clean_path(raw: str) -> str:
    """Strip quotes/whitespace; normalize slashes (keep Windows drive letters)."""
    s = raw.strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        return s
    # Preserve POSIX absolute paths (/workspace/...); lstrip("./") would eat the leading slash.
    if s.startswith("/"):
        return s
    return s.lstrip("./")


def _is_windows_absolute(s: str) -> bool:
    return len(s) > 2 and s[1] == ":" and s[0].isalpha()


def _strip_windows_path_for_container(raw: str) -> str:
    """
    On Linux/Docker, ``C:/Users/.../outputs/foo`` is not a valid host path.
    Map to project-relative path (e.g. ``outputs/foo``) when possible.
    """
    if not _is_windows_absolute(raw):
        return raw
    norm = raw.replace("\\", "/")
    low = norm.lower()
    marker = "mcap_yolo_quality_gateway/"
    idx = low.find(marker)
    if idx >= 0:
        return norm[idx + len(marker) :].lstrip("/")
    for needle in ("/outputs/", "/test_data/"):
        pos = low.find(needle)
        if pos >= 0:
            return norm[pos + 1 :]
    return raw


def resolve_mcap_path(mcap_path: str) -> Path:
    raw = clean_path(mcap_path)
    if _is_windows_absolute(raw):
        raw = _strip_windows_path_for_container(raw)
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    candidates = [
        Path.cwd() / raw,
        PROJECT_ROOT / raw,
        PROJECT_ROOT / "test_data" / p.name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return (PROJECT_ROOT / raw).resolve()


def resolve_output_dir(output_dir: str) -> Path:
    raw = clean_path(output_dir)
    if _is_windows_absolute(raw):
        raw = _strip_windows_path_for_container(raw)
    p = Path(raw)
    if p.is_absolute() and p.is_dir():
        return p.resolve()
    candidates = [
        Path.cwd() / raw,
        PROJECT_ROOT / raw,
    ]
    # Bare run name → outputs/<name>
    if "/" not in raw and "\\" not in raw:
        candidates.insert(0, OUTPUTS_ROOT / raw)
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return (PROJECT_ROOT / raw).resolve()


def path_hints() -> dict:
    """Return actual runtime paths — no templates, no hardcoding."""
    in_docker = Path("/.dockerenv").exists()
    root = PROJECT_ROOT.resolve()
    out = OUTPUTS_ROOT.resolve()
    root_s = str(root).replace("\\", "/")
    out_s = str(out).replace("\\", "/")

    # List actual output run directories
    output_runs: list[str] = []
    if out.is_dir():
        output_runs = sorted(
            d.name for d in out.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    # List actual MCAP files in test_data/
    test_data = root / "test_data"
    mcap_files: list[str] = []
    if test_data.is_dir():
        mcap_files = sorted(f.name for f in test_data.iterdir() if f.suffix == ".mcap")

    return {
        "in_docker": in_docker,
        "project_root": root_s,
        "outputs_root": out_s,
        "output_runs": output_runs,
        "mcap_files": mcap_files,
    }


def output_dir_static_rel(root: Path) -> Optional[str]:
    """Relative path under outputs/ for StaticFiles mount, or None."""
    root = root.resolve()
    try:
        return str(root.relative_to(OUTPUTS_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return None


def output_dir_display(root: Path) -> str:
    """Prefer project-relative path for UI (e.g. outputs/sample_run)."""
    root = root.resolve()
    try:
        return str(root.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(root)
