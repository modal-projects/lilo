# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Additivity check: does Lilo's full-batch B-gradient equal the sum of its per-group gradients?

All calls share one LoRA (same A, B=0 kept via lr=0 optimizer steps), so
    grad(full 128 datums)  ==  sum_k grad(group k)
must hold up to bf16 noise if micro-batch accumulation and DP partitioning are lossless.
Only ``grads_local`` files are needed (the DP mean is reconstructed).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from compare import _LAYER, DP, TP, cosine, flat, lilo_name_pattern


def load_grads(dump_dir: Path, slot: int) -> dict[str, list[torch.Tensor]]:
    """{canon_name: [DP-mean grad per tp rank]} for the slot's language-model B (linear_out) params."""
    pat = lilo_name_pattern(slot)
    out: dict[str, list[torch.Tensor]] = {}
    for tp in range(TP):
        locals_ = []
        for dp in range(DP):
            files = glob.glob(str(dump_dir / f"rank*_tp{tp}_dp{dp}_grads_local.pt"))
            assert len(files) == 1, (dump_dir, tp, dp, files)
            locals_.append(torch.load(files[0], weights_only=False))
        for raw in locals_[0]:
            m = pat.match(raw)
            if (
                not m
                or not m.group(1).startswith("language_model.")
                or m.group(2) != "linear_out"
            ):
                continue
            name = f"{m.group(1)}.{m.group(2)}"
            out.setdefault(name, []).append(
                torch.stack([loc[raw].double() for loc in locals_]).mean(0)
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=Path("~/work/graddiff/lilo_pergroup").expanduser()
    )
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--groups", type=int, default=16)
    args = ap.parse_args()
    full = load_grads(args.root / "full", args.slot)
    groups = [load_grads(args.root / f"g{k}", args.slot) for k in range(args.groups)]
    names = sorted(full, key=lambda n: (int(_LAYER.match(n).group(1)), n))
    assert len(names) == 80, len(names)
    F = torch.cat([flat(full[n]) for n in names])
    S = torch.cat([sum(flat(g[n]) for g in groups) for n in names])
    print(
        f"total: |full|={F.norm():.6g} |sum_groups|={S.norm():.6g} ratio sum/full={S.norm() / F.norm():.4f} cos={cosine(F, S):.4f}"
    )
    per_group = [torch.cat([flat(g[n]) for n in names]) for g in groups]
    print(
        "sqrt(sum |g_k|^2) =", f"{sum(float(G.dot(G)) for G in per_group) ** 0.5:.6g}"
    )
    for k, G in enumerate(per_group):
        print(
            f"g{k}: |g|={G.norm():.4g}  <full,g>/|g|^2={float(F.dot(G) / G.dot(G)):.3f}  <sum,g>/|g|^2={float(S.dot(G) / G.dot(G)):.3f}"
        )
    print("\nper-module cos(full, sum) / ratio sum/full:")
    for n in names:
        f, s = flat(full[n]), sum(flat(g[n]) for g in groups)
        print(
            f"  {n.replace('language_model.decoder.', '').replace('.linear_out', '')}: cos={cosine(f, s):.4f} ratio={s.norm() / f.norm():.3f}"
        )


if __name__ == "__main__":
    main()
