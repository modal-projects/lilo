"""Read-only inventory of the experiment's rollout replicas."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROBE = """import collections,json,subprocess,urllib.request
models=json.load(urllib.request.urlopen("http://127.0.0.1:8001/v1/models",timeout=8))["data"][1:]
print(json.dumps({"adapter_count":len(models),"first_adapters":[r["id"] for r in models[:3]],"by_model":dict(collections.Counter(r["id"].split("/")[0] for r in models)),"gpu":subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],text=True).strip()}))"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="tailrl-hero-v1")
    args = p.parse_args()
    root = Path(__file__).resolve().parent / "results/multilora-codegolf" / args.run
    state = json.loads((root / "supervisor.json").read_text())
    spec = state["pool"]
    digest = hashlib.sha256(
        (spec["definition_id"] + "\0" + spec["revision"]).encode()
    ).hexdigest()[:16]
    containers = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "modal", "container", "list", "--json"], text=True
        )
    )
    containers = [c for c in containers if c["app_name"] == "lilo-lora-" + digest]

    def inspect(c):
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "container",
                    "exec",
                    c["container_id"],
                    "--",
                    "python",
                    "-c",
                    PROBE,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (
                {**c, **json.loads(r.stdout)}
                if r.returncode == 0
                else {**c, "error": r.stderr[-1000:]}
            )
        except Exception as e:
            return {**c, "error": repr(e)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, containers))
    (root / "replica-inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(
            r["container_id"],
            "adapters",
            r.get("adapter_count"),
            "GPU",
            r.get("gpu"),
            "error",
            r.get("error"),
        )


if __name__ == "__main__":
    main()
