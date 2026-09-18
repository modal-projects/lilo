"""Fresh FA3 forward + autograd backward: repeatability and batch invariance."""

import argparse
import itertools
import json
from pathlib import Path
import traceback

import torch
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func


def compare(a, b):
    equal = a == b
    delta = torch.where(equal, 0, (a.float() - b.float()).abs())
    return {
        "exact": torch.equal(a, b),
        "max_abs": delta.max().item(),
        "different_elements": int((~equal).sum()),
        "has_nan": bool(torch.isnan(a).any() or torch.isnan(b).any()),
    }


def run(output):
    torch.manual_seed(20260914)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "method": "Fresh public FA3 forward and torch.autograd.grad on every invocation; no injected forward intermediates; identical dO per sequence.",
        "settings": {
            "head_dim": 256,
            "q_heads": 8,
            "deterministic": True,
            "pack_gqa": False,
            "num_splits": 1,
        },
        "checks": [],
    }

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def check(kind, info, result, expected, repeats=1):
        checks = {
            name: compare(a, b)
            for name, a, b in zip(("out", "lse", "dq", "dk", "dv"), result, expected)
        }
        row = dict(
            kind=kind,
            **info,
            repeats=repeats,
            comparisons=checks,
            passed=all(x["exact"] and not x["has_nan"] for x in checks.values()),
        )
        report["checks"].append(row)
        if not row["passed"]:
            print("MISMATCH " + json.dumps(row), flush=True)
        return row["passed"]

    def data(sq, sk, kv, dtype):
        q = torch.randn(sq, 8, 256, device="cuda", dtype=dtype)
        k, v = [torch.randn(sk, kv, 256, device="cuda", dtype=dtype) for _ in range(2)]
        return q, k, v, torch.randn_like(q)

    def evaluate(samples, order, mode, layout):
        values = [samples[i] for i in order]
        options = dict(
            deterministic=True,
            num_splits=1,
            pack_gqa=False,
            causal=mode == "causal",
            window_size=(37, 11) if mode == "local" else (-1, -1),
            softcap=15.0 if mode == "softcap" else 0.0,
            return_attn_probs=True,
        )
        combine = torch.stack if layout == "dense" else torch.cat
        q, k, v = [
            combine([s[j] for s in values]).detach().requires_grad_() for j in range(3)
        ]
        do = combine([s[3] for s in values])
        if layout == "dense":
            out, lse = flash_attn_func(q, k, v, **options)
        else:
            lengths_q = [s[0].shape[0] for s in values]
            lengths_k = [s[1].shape[0] for s in values]
            cuq = torch.tensor(
                [0] + list(itertools.accumulate(lengths_q)),
                device="cuda",
                dtype=torch.int32,
            )
            cuk = torch.tensor(
                [0] + list(itertools.accumulate(lengths_k)),
                device="cuda",
                dtype=torch.int32,
            )
            out, lse = flash_attn_varlen_func(
                q, k, v, cuq, cuk, max(lengths_q), max(lengths_k), **options
            )
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
        # Return by original sequence identity; neither forward nor backward sees these reference outputs.
        if layout == "dense":
            parts = list(
                zip(
                    out.detach().unbind(),
                    lse.detach().unbind(),
                    dq.unbind(),
                    dk.unbind(),
                    dv.unbind(),
                )
            )
        else:
            parts = list(
                zip(
                    out.detach().split(lengths_q),
                    lse.detach().split(lengths_q, dim=-1),
                    dq.split(lengths_q),
                    dk.split(lengths_k),
                    dv.split(lengths_k),
                )
            )
        return dict(zip(order, parts))

    try:
        for dtype, kvheads, mode, layout in itertools.product(
            [torch.bfloat16, torch.float16],
            [8, 2, 1],
            ["causal", "full", "local", "softcap"],
            ["dense", "varlen"],
        ):
            shapes = (
                [(257, 257)] * 6
                if layout == "dense"
                else [
                    (113, 203),
                    (257, 129),
                    (64, 64),
                    (509, 383),
                    (203, 317),
                    (129, 257),
                ]
            )
            samples = [data(sq, sk, kvheads, dtype) for sq, sk in shapes]
            info = dict(dtype=str(dtype), kvheads=kvheads, mode=mode, layout=layout)
            refs = {i: evaluate(samples, [i], mode, layout)[i] for i in range(6)}
            for repeat in range(3):
                for i in range(6):
                    check(
                        "single_repeat",
                        dict(info, sequence=i, repeat=repeat),
                        evaluate(samples, [i], mode, layout)[i],
                        refs[i],
                    )
            for order in [list(range(6)), list(reversed(range(6))), [2, 0, 5, 1, 4, 3]]:
                first = None
                for repeat in range(10):
                    results = evaluate(samples, order, mode, layout)
                    if first is None:
                        first = results
                    for i in order:
                        check(
                            "single_vs_batch",
                            dict(info, sequence=i, order=order, repeat=repeat),
                            results[i],
                            refs[i],
                        )
                        check(
                            "batch_repeat",
                            dict(info, sequence=i, order=order, repeat=repeat),
                            results[i],
                            first[i],
                        )
            save()
            print("COMPLETE " + json.dumps(info), flush=True)
        # Realistic sequence lengths with six concurrent sequences, including fresh packed forward.
        for length in (4096, 8192):
            shapes = [
                (length, length),
                (length - 17, length - 17),
                (113, 113),
                (257, 257),
                (203, 203),
                (509, 509),
            ]
            samples = [data(sq, sk, 2, torch.bfloat16) for sq, sk in shapes]
            refs = {i: evaluate(samples, [i], "causal", "varlen")[i] for i in range(6)}
            for order in [list(range(6)), list(reversed(range(6)))]:
                for repeat in range(20):
                    results = evaluate(samples, order, "causal", "varlen")
                    for i in order:
                        check(
                            "long_single_vs_batch",
                            dict(length=length, sequence=i, order=order, repeat=repeat),
                            results[i],
                            refs[i],
                        )
            save()
            print("LONG_COMPLETE " + str(length), flush=True)
        report["passed"] = all(c["passed"] for c in report["checks"])
    except Exception:
        report["error"] = traceback.format_exc()
        report["passed"] = False
        print(report["error"], flush=True)
    report["total_checks"] = len(report["checks"])
    report["failed_checks"] = sum(not c["passed"] for c in report["checks"])
    save()
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
