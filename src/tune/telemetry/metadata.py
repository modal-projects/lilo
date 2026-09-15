"""Explicit experiment labels; arbitrary user metadata is never exported."""

from collections.abc import Mapping

METADATA_KEYS = {"run_id": "tune.run_id", "attempt_id": "tune.run_attempt_id"}


def experiment_tags(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        return {}
    return {
        tag: value
        for key, tag in METADATA_KEYS.items()
        if isinstance(value := metadata.get(key), str) and 0 < len(value) <= 256
    }


def common_tags(items: list[Mapping]) -> dict[str, str]:
    """A shared batch belongs to an experiment only when all commands agree."""
    if not items:
        return {}
    return {
        tag: items[0][tag]
        for tag in METADATA_KEYS.values()
        if tag in items[0] and all(item.get(tag) == items[0][tag] for item in items)
    }
