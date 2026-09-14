"""Pinned CodeContests data, Codeforces only; retain public, private and generated tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = "deepmind/code_contests"
REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"


def prepare(destination: Path, count: int = 160):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
        if f.startswith("data/train-") and f.endswith(".parquet")
    )
    problems = []
    for filename in files:
        path = hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION)
        for row in pq.read_table(path).to_pylist():
            if row["source"] != 2 or not 800 <= row.get("cf_rating", 0) <= 1400:
                continue
            tests = []
            for key in ("public_tests", "private_tests", "generated_tests"):
                tests += [
                    {"input": i, "output": o, "kind": key}
                    for i, o in zip(row[key]["input"], row[key]["output"], strict=True)
                ]
            # Avoid custom-output problems; reference validation adds a runtime gate.
            if (
                not row["private_tests"]["input"]
                or not tests
                or len(tests) > 100
                or sum(len(t["input"]) + len(t["output"]) for t in tests) > 100000
            ):
                continue
            refs = [
                s
                for lang, s in zip(
                    row["solutions"]["language"],
                    row["solutions"]["solution"],
                    strict=True,
                )
                if lang == 3
            ]
            if not refs:
                continue
            problems.append(
                {
                    "id": row["name"],
                    "statement": row["description"],
                    "rating": row["cf_rating"],
                    "tests": tests,
                    "reference": min(refs, key=len),
                }
            )
            if len(problems) >= count:
                break
        if len(problems) >= count:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"repo": REPO, "revision": REVISION, "problems": problems}
    destination.write_text(json.dumps(payload))
    print(
        json.dumps(
            {
                "problems": len(problems),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/problems.json"))
    p.add_argument("--count", type=int, default=160)
    a = p.parse_args()
    prepare(a.output, a.count)
