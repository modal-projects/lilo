"""Resolve a moving Miles ref before constructing the cached Modal image."""

import os
import re
import subprocess
from functools import lru_cache

from lilo.backends.miles_config import MILES_REF

MILES_REPOSITORY = "https://github.com/radixark/miles.git"


def validate_commit(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("LILO_MILES_COMMIT must be a full lowercase Git commit SHA")
    return value


@lru_cache(maxsize=1)
def resolve_miles_commit() -> str:
    """Resolve once per deployment process; allow an exact reproducibility override."""
    override = os.environ.get("LILO_MILES_COMMIT")
    if override is not None:
        return validate_commit(override)
    ref = f"refs/heads/{MILES_REF}"
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", MILES_REPOSITORY, ref],
        check=True, capture_output=True, text=True, timeout=30,
    )
    entries = [line.split() for line in result.stdout.splitlines()]
    commits = [sha for sha, name in entries if name == ref]
    if len(commits) != 1:
        raise RuntimeError(f"Expected exactly one Miles {ref} commit")
    return validate_commit(commits[0])
