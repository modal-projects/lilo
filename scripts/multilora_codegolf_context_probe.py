"""Exercise near-64K inference on adapters from a running capacity test."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time

import modal
import tinker
from tinker import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent / "results/multilora-codegolf" / args.run
    state = json.loads((root / "supervisor.json").read_text())
    artifacts = modal.Dict.from_name(state["app_id"] + "-artifacts")
    service = tinker.ServiceClient(
        base_url=state["base_url"], api_key=os.environ["TINKER_API_KEY"]
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    chunk = tok.encode(
        "def solve():\n    x = int(input())\n    print(x + 1)\n",
        add_special_tokens=False,
    )
    prompt = (chunk * (65519 // len(chunk) + 1))[:65519]
    deadline = time.monotonic() + 900
    while True:
        selected = {}
        for key, value in artifacts.items():
            if key.startswith("sampler_artifact:"):
                if (
                    value["model_id"] not in selected
                    or value["publish_version"]
                    > selected[value["model_id"]]["publish_version"]
                ):
                    selected[value["model_id"]] = value
        if len(selected) == 4:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Four published adapters not available")
        time.sleep(3)

    def sample(record):
        start = time.monotonic()
        sampler = service.create_sampling_client(model_path=record["model_path"])
        result = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=16, temperature=0, stop=[]),
        ).result(timeout=300)
        seq = result.sequences[0]
        assert len(seq.tokens) == 16 and len(seq.logprobs) == 16, (
            len(seq.tokens),
            len(seq.logprobs),
            seq.stop_reason,
        )
        assert all(math.isfinite(x) for x in seq.logprobs)
        return {
            "model_id": record["model_id"],
            "model_path": record["model_path"],
            "prompt_tokens": len(prompt),
            "output_tokens": len(seq.tokens),
            "seconds": time.monotonic() - start,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(sample, selected.values()))
    report = {"status": "passed", "clients": results}
    (root / "inference-context.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
