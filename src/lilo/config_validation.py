"""Checks for backend options that would override Lilo-managed settings."""


def reject_managed_options(options, managed):
    if not isinstance(options, dict):
        raise ValueError("backend configuration must be a mapping")
    conflicts = options.keys() & managed
    if conflicts:
        raise ValueError(f"options managed by Lilo: {sorted(conflicts)}")
