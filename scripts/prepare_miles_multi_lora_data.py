"""Download small numeric-answer GSM8K and DAPO subsets for the Tinker E2E test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2e_miles_multi_lora import numeric_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/data/multi_lora"))
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.rows < 1 or not 0 < args.max_new_tokens < args.context_length:
        parser.error("positive row count and generation budget below context length required")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sources = [
        ("gsm8k", "openai/gsm8k", "main"),
        ("dapo", "open-r1/DAPO-Math-17k-Processed", None),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, repo, config in sources:
        dataset = load_dataset(repo, config, split="train", streaming=True)
        rows = []
        for index, row in enumerate(dataset):
            if name == "gsm8k":
                prompt, answer = row["question"], row["answer"]
            else:
                source_prompt = row.get("source_prompt")
                prompt = (
                    source_prompt[0].get("content", "")
                    if isinstance(source_prompt, list) and source_prompt
                    else row.get("prompt", "")
                )
                answer = row.get("solution") or row.get("reward_model", {}).get("ground_truth", "")
            answer = numeric_answer(str(answer))
            if not isinstance(prompt, str) or not prompt or answer is None:
                continue
            ids = tokenizer.encode(
                prompt + "\nGive the final answer in \\boxed{}.\nAnswer:",
                add_special_tokens=True,
            )
            if len(ids) + args.max_new_tokens >= args.context_length:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "answer": str(answer),
                    "source": repo,
                    "source_row": index,
                }
            )
            if len(rows) == args.rows:
                break
        if len(rows) < args.rows:
            raise RuntimeError(f"only {len(rows)} eligible rows in {repo}")
        path = args.output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
