# DAPO sweep inputs

`dapo.jsonl` is the same 320-row numeric-answer input file used by the earlier 12-client DAPO cost experiment. It was prepared from the `all` / `train` split of [open-r1/DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed), retaining the source-row index in each object. The sweep records the file's SHA-256 in every run report.

The runner adds `Give the final answer in \boxed{}.` and `Answer:` to each stored prompt, then tokenizes it as plain text for Qwen3.5-9B-Base. Client `i`, update `s` (one-based), group `g` uses row `(i*97 + (s-1)*8 + g) % 320`. The same client index uses the same prompt sequence in every sweep point.
