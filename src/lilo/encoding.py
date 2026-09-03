import hashlib
import json


def canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def fingerprint(kind: str, payload: object) -> str:
    return hashlib.sha256(f"{kind}\0{canonical_json(payload)}".encode()).hexdigest()
