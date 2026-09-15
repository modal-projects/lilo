"""Attention-only checks: reference accuracy, repeatability, packing, and timing."""

from pathlib import Path
import argparse
import itertools
import json
import traceback
import torch
from flash_attn_interface import flash_attn_func, _flash_attn_backward


def diff(a, b):
    return {
        "exact": torch.equal(a, b),
        "max_abs": (a.float() - b.float()).abs().max().item(),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def run(output):
    torch.manual_seed(123)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checks": [],
        "timing": [],
    }

    def record(case, **values):
        row = {"case": case, **values}
        report["checks"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    def forward(q, k, v, causal, window, softcap):
        return flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            window_size=window,
            softcap=softcap,
            pack_gqa=False,
            deterministic=True,
            return_attn_probs=True,
        )

    def backward(
        q,
        k,
        v,
        o,
        lse,
        do,
        causal,
        window,
        softcap,
        deterministic=True,
        cu=None,
        maxlen=None,
    ):
        grads = [torch.empty_like(x) for x in (q, k, v)]
        _flash_attn_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=maxlen,
            max_seqlen_k=maxlen,
            dq=grads[0],
            dk=grads[1],
            dv=grads[2],
            softmax_scale=q.shape[-1] ** -0.5,
            is_causal=causal,
            window_size_left=window[0],
            window_size_right=window[1],
            softcap=softcap,
            deterministic=deterministic,
        )
        return grads

    try:
        # Use an explicit FP32 reference and bottom-right mask, including fully masked rows.
        for dtype, kvheads, (sq, sk), (causal, window), softcap in itertools.product(
            [torch.bfloat16, torch.float16],
            [8, 2, 1],
            [(113, 203), (203, 113)],
            [(False, (-1, -1)), (True, (-1, -1)), (False, (37, 11))],
            [0.0, 15.0],
        ):
            q = torch.randn(2, sq, 8, 256, device="cuda", dtype=dtype)
            k, v = [
                torch.randn(2, sk, kvheads, 256, device="cuda", dtype=dtype)
                for _ in range(2)
            ]
            do = torch.randn_like(q)
            o, lse = forward(q, k, v, causal, window, softcap)
            grads = backward(q, k, v, o, lse, do, causal, window, softcap)
            repeats = [
                backward(q, k, v, o, lse, do, causal, window, softcap) for _ in range(5)
            ]
            qr, kr, vr = [x.float().detach().requires_grad_() for x in (q, k, v)]
            scores = (
                torch.einsum(
                    "bqhd,bkhd->bhqk", qr, kr.repeat_interleave(8 // kvheads, dim=2)
                )
                / 16
            )
            if softcap:
                scores = torch.tanh(scores / softcap) * softcap
            qi = torch.arange(sq, device="cuda")[:, None] + sk - sq
            ki = torch.arange(sk, device="cuda")[None, :]
            allowed = torch.ones((sq, sk), dtype=torch.bool, device="cuda")
            if causal:
                allowed &= ki <= qi
            if window[0] >= 0:
                allowed &= ki >= qi - window[0]
            if window[1] >= 0:
                allowed &= ki <= qi + window[1]
            valid = allowed.any(-1)
            scores = scores.masked_fill(~allowed, -torch.inf)
            scores = torch.where(
                valid[None, None, :, None], scores, torch.zeros_like(scores)
            )
            p = scores.softmax(-1).masked_fill(~allowed, 0)
            ref = torch.einsum(
                "bhqk,bkhd->bqhd", p, vr.repeat_interleave(8 // kvheads, dim=2)
            )
            refgrads = torch.autograd.grad(ref, (qr, kr, vr), do.float())
            error = [diff(a, b) for a, b in zip(grads, refgrads)]
            repeat = [
                all(torch.equal(g[i], grads[i]) for g in repeats) for i in range(3)
            ]
            # Fused attention rounds P/dS to input dtype; compare against FP32 with dtype-aware tolerance.
            tol = 0.035 if dtype == torch.bfloat16 else 0.006
            accurate = all(
                torch.allclose(a.float(), b, atol=tol, rtol=tol)
                for a, b in zip(grads, refgrads)
            )
            record(
                "reference_repeat",
                dtype=str(dtype),
                kvheads=kvheads,
                sq=sq,
                sk=sk,
                causal=causal,
                window=window,
                softcap=softcap,
                repeat_exact=repeat,
                reference=error,
                pass_check=accurate and all(repeat),
            )
        # Packing checks use identical saved forward intermediates to isolate backward.
        for dtype, kvheads, causal in itertools.product(
            [torch.bfloat16, torch.float16], [8, 2, 1], [False, True]
        ):
            lengths = [113, 257, 64, 509, 203, 129]
            saved = []
            for length in lengths:
                q = torch.randn(1, length, 8, 256, device="cuda", dtype=dtype)
                k, v = [
                    torch.randn(1, length, kvheads, 256, device="cuda", dtype=dtype)
                    for _ in range(2)
                ]
                do = torch.randn_like(q)
                o, lse = forward(q, k, v, causal, (-1, -1), 0)
                grads = backward(q, k, v, o, lse, do, causal, (-1, -1), 0)
                saved.append((q, k, v, o, lse, do, grads))
            for order in [list(range(6)), list(reversed(range(6))), [2, 0, 5, 1, 4, 3]]:
                packed = [
                    torch.cat([saved[i][j][0] for i in order], dim=0)
                    for j in (0, 1, 2, 3, 5)
                ]
                q, k, v, o, do = packed
                lse = torch.cat([saved[i][4][0] for i in order], dim=-1).contiguous()
                cu = torch.tensor(
                    [0] + [sum(lengths[i] for i in order[:n]) for n in range(1, 7)],
                    device="cuda",
                    dtype=torch.int32,
                )
                grads = backward(
                    q, k, v, o, lse, do, causal, (-1, -1), 0, cu=cu, maxlen=max(lengths)
                )
                expected = [
                    torch.cat([saved[i][6][j][0] for i in order]) for j in range(3)
                ]
                checks = [diff(a, b) for a, b in zip(grads, expected)]
                record(
                    "packing",
                    dtype=str(dtype),
                    kvheads=kvheads,
                    causal=causal,
                    order=order,
                    gradients=checks,
                    pass_check=all(c["exact"] and c["finite"] for c in checks),
                )
        # Larger batches stress semaphore scheduling beyond the number of resident CTAs.
        for length in (4096, 8192):
            q = torch.randn(2, length, 8, 256, device="cuda", dtype=torch.bfloat16)
            k, v = [
                torch.randn(2, length, 2, 256, device="cuda", dtype=q.dtype)
                for _ in range(2)
            ]
            o, lse = forward(q, k, v, True, (-1, -1), 0)
            do = torch.randn_like(q)
            reference = backward(q, k, v, o, lse, do, True, (-1, -1), 0)
            exact = all(
                all(
                    torch.equal(a, b)
                    for a, b in zip(
                        reference, backward(q, k, v, o, lse, do, True, (-1, -1), 0)
                    )
                )
                for _ in range(10)
            )
            record("long_repeat", length=length, pass_check=exact)
            for deterministic in (False, True):
                for _ in range(5):
                    backward(q, k, v, o, lse, do, True, (-1, -1), 0, deterministic)
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(20):
                    backward(q, k, v, o, lse, do, True, (-1, -1), 0, deterministic)
                end.record()
                end.synchronize()
                report["timing"].append(
                    {
                        "length": length,
                        "deterministic": deterministic,
                        "backward_ms": start.elapsed_time(end) / 20,
                    }
                )
        report["passed"] = all(c["pass_check"] for c in report["checks"])
    except Exception:
        report["error"] = traceback.format_exc()
        report["passed"] = False
        print(report["error"], flush=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
