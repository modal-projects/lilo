"""Run isolated client counts sequentially; refuse to proceed after failed cleanup."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument(
        "--prefix",
        default="dapo-sweep-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
    )
    a = p.parse_args()
    assert set(a.counts) <= {1, 2, 4, 8, 16, 32}
    assert re.fullmatch("[a-zA-Z0-9-]+", a.prefix)
    out = ROOT / "scripts/results/dapo-client-sweep"
    out.mkdir(parents=True, exist_ok=True)
    definition = ROOT / "src/lilo/providers/modal/definitions/qwen3_5_9b_dapo_sweep.py"
    template = definition.read_text()
    ledger = dict(prefix=a.prefix, counts=a.counts, runs=[], started_at=time.time())
    try:
        for count in a.counts:
            run = f"{a.prefix}-c{count:02d}"
            run_dir = out / run
            run_dir.mkdir(exist_ok=False)
            definition.write_text(
                re.sub(
                    r'SWEEP_ISOLATION_ID = "[^"]*"',
                    f'SWEEP_ISOLATION_ID = "{run}"',
                    template,
                )
            )
            sources = {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for tree in ["src/lilo", "scripts"]
                for p in (ROOT / tree).rglob("*.py")
                if "results" not in p.parts
            }
            (run_dir / "source-sha256.json").write_text(json.dumps(sources, indent=2))
            print("START_POINT", run, flush=True)
            with (run_dir / "launch.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/dapo_client_sweep_point.py",
                        "--clients",
                        str(count),
                        "--run",
                        run,
                    ],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": "src:scripts"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            state_path = run_dir / "supervisor.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            ledger["runs"].append(
                dict(
                    run=run,
                    clients=count,
                    returncode=result.returncode,
                    status=state.get("status"),
                    drained=state.get("drained"),
                )
            )
            (out / (a.prefix + ".json")).write_text(json.dumps(ledger, indent=2))
            if (
                result.returncode
                or state.get("status") != "completed"
                or not state.get("drained")
            ):
                raise RuntimeError(
                    f"{run} did not complete and drain; inspect {run_dir}/launch.log"
                )
            print("FINISHED_POINT", run, flush=True)
        ledger["status"] = "completed"
    finally:
        definition.write_text(template)
        ledger["finished_at"] = time.time()
        (out / (a.prefix + ".json")).write_text(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
