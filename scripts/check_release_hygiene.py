#!/usr/bin/env python3
"""Reject private paths, large tracked artifacts, and removed experiment names."""

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH_MARKERS = (
    b"/vepfs" + b"-",
    b"/mnt/parallel" + b"_ssd/",
    b"/home/" + b"zdhs",
)
REMOVED_NAME_MARKERS = ("w4d_model", "law_model", "rft_model", "transfuser_old")
MAX_TRACKED_BYTES = 10 * 1024 * 1024


def main() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
    ).split(b"\0")
    failures = []
    for raw_path in filter(None, tracked):
        relative = raw_path.decode()
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"tracked file exceeds 10 MiB: {relative}")
        if any(marker in relative.lower() for marker in REMOVED_NAME_MARKERS):
            failures.append(f"removed experiment remains tracked: {relative}")
        if path.suffix.lower() in {".py", ".sh", ".yaml", ".yml", ".md"}:
            data = path.read_bytes()
            if any(marker in data for marker in PRIVATE_PATH_MARKERS):
                failures.append(f"private absolute path remains: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"release hygiene passed for {len(tracked) - 1} tracked paths")


if __name__ == "__main__":
    main()
