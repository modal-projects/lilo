"""Run a bounded selection of upstream correctness and 1000-repeat checks."""

import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import traceback
import torch

sys.path.append("/root/flash-attention/hopper")
import test_flash_attn_bwd_determinism as suite


def run(output):
    rows = []
    # Each selected upstream test checks reference accuracy, then repeats backward 1000 times.
    for dtype, mha_type, mode, layout in itertools.product(
        [torch.bfloat16, torch.float16],
        ["mha", "gqa", "mqa"],
        ["causal", "full", "local", "softcap"],
        ["dense", "varlen"],
    ):
        row = dict(dtype=str(dtype), mha_type=mha_type, mode=mode, layout=layout)
        kwargs = dict(
            seqlen_q=113,
            seqlen_k=203,
            d=256,
            causal=mode == "causal",
            local=mode == "local",
            softcap=15.0 if mode == "softcap" else 0.0,
            deterministic=True,
            has_qv=False,
            mha_type=mha_type,
            dtype=dtype,
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if layout == "dense":
                    suite.test_flash_attn_output(V_colmajor=False, **kwargs)
                else:
                    suite.test_flash_attn_varlen_output(add_unused_qkv=True, **kwargs)
            row["passed"] = True
        except Exception:
            row["passed"] = False
            row["error"] = traceback.format_exc()
        rows.append(row)
        output.write_text(
            json.dumps(
                {"checks": rows, "passed": all(r["passed"] for r in rows)}, indent=2
            )
            + "\n"
        )
        print("UPSTREAM " + json.dumps(row), flush=True)
    return all(r["passed"] for r in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
