# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
"""Compare the step-0 Lilo and raw-Miles dumps produced by lilo_arm.py / miles_arm.py.

Usage:
    uv run scripts/graddiff/compare.py --work ~/work/graddiff --out docs/graddiff_step0_results.md

Reads:
    {work}/batch.json                       fixed batch (make_batch.py)
    {work}/lilo_arm_outputs.json            Lilo client outputs (per-datum trainer logprobs, loss)
    {work}/lilo_dump/lilo/*.pt              Lilo trainer dumps (grads/params/delta per rank)
    {work}/miles_dump/miles_dump/miles/*.pt Miles trainer dumps
    {work}/miles_dump/miles_dump/details/policy_loss_debug/*.pt  Miles per-sample train logprobs
Writes a markdown results file and a JSON with the per-module table.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import torch

LR = 1e-4
TP = 4
DP = 2


def lilo_name_pattern(slot: int) -> re.Pattern[str]:
    return re.compile(
        rf"^slot{slot}\.module\.module\.(.*)\.adapters\.{slot}\.(linear_in|linear_out)\.weight$"
    )


_LILO_NAME = lilo_name_pattern(0)
_MILES_NAME = re.compile(
    r"^module\.module\.(.*)\.adapter\.(linear_in|linear_out)\.weight$"
)
_LAYER = re.compile(r"language_model\.decoder\.layers\.(\d+)\.(.*)$")


def canon(name: str, pattern: re.Pattern[str]) -> str | None:
    m = pattern.match(name)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}"


def load_rank(dump_dir: Path, tp: int, dp: int, kind: str) -> dict[str, torch.Tensor]:
    files = glob.glob(str(dump_dir / f"rank*_tp{tp}_dp{dp}_{kind}.pt"))
    assert len(files) == 1, (dump_dir, tp, dp, kind, files)
    return torch.load(files[0], weights_only=False)


def load_arm(
    dump_dir: Path, pattern: re.Pattern[str]
) -> dict[str, dict[str, list[torch.Tensor]]]:
    """Return {kind: {canon_name: [tensor per tp rank]}} for language-model params.

    grads_reduced is reconstructed as the DP mean of grads_local (verified against the
    dumped reduced tensor on the owning rank; Miles' LayerWise optimizer leaves a stale
    partial value on the non-owner rank, so the dumped value is not usable directly).
    """
    out: dict[str, dict[str, list[torch.Tensor]]] = {
        "grad": {},
        "delta": {},
        "before": {},
        "after": {},
        "reduce_check": {},
    }
    for tp in range(TP):
        local = [load_rank(dump_dir, tp, dp, "grads_local") for dp in range(DP)]
        reduced = [load_rank(dump_dir, tp, dp, "grads_reduced") for dp in range(DP)]
        before = load_rank(dump_dir, tp, 0, "params_before")
        after = load_rank(dump_dir, tp, 0, "params_after")
        delta = load_rank(dump_dir, tp, 0, "delta")
        for raw in before:
            name = canon(raw, pattern)
            if name is None or not name.startswith("language_model."):
                continue
            g_mean = torch.stack([loc[raw].double() for loc in local]).mean(0)
            # the reduced tensor must equal the DP mean on at least one DP rank
            errs = [
                float((r[raw].double() - g_mean).norm() / (g_mean.norm() + 1e-30))
                for r in reduced
            ]
            out["reduce_check"].setdefault(name, []).append(min(errs))
            out["grad"].setdefault(name, []).append(g_mean.float())
            out["delta"].setdefault(name, []).append(delta[raw].float())
            out["before"].setdefault(name, []).append(before[raw].float())
            out["after"].setdefault(name, []).append(after[raw].float())
    return out


def flat(ts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1).double() for t in ts])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.dot(b) / (a.norm() * b.norm() + 1e-30))


def module_type(name: str) -> str:
    m = _LAYER.match(name)
    assert m, name
    rest = m.group(2)
    return rest.replace(".linear_in", "").replace(".linear_out", "").split(".")[-1]


def _fingerprint(logprobs: torch.Tensor) -> tuple:
    return (logprobs.numel(), *[round(float(x), 4) for x in logprobs[:16]])


def load_miles_samples(work: Path, datums: list[dict]) -> dict[int, dict]:
    """Map datum index -> Miles policy_loss_debug sample.

    Miles' `index` is the position inside the micro-batch, so samples are matched to
    datums by their (identical) rollout logprobs instead.
    """
    lookup = {}
    for i, datum in enumerate(datums):
        mask = torch.tensor(datum["mask"], dtype=torch.bool)
        key = _fingerprint(
            torch.tensor(datum["sampled_logprobs"], dtype=torch.float32)[mask]
        )
        assert key not in lookup, i
        lookup[key] = i
    out: dict[int, dict] = {}
    for f in glob.glob(
        str(work / "miles_dump/miles_dump/details/policy_loss_debug/*.pt")
    ):
        d = torch.load(f, weights_only=False)
        for s in d["samples"]:
            mmask = s["local_loss_mask"].bool()
            key = _fingerprint(s["rollout_log_probs"].float()[mmask])
            i = lookup[key]
            assert i not in out, i
            out[i] = s
    assert len(out) == len(datums), len(out)
    return out


def forward_parity(work: Path) -> dict:
    batch = json.loads((work / "batch.json").read_text())
    lilo = json.loads((work / "lilo_arm_outputs.json").read_text())
    datums = batch["datums"]
    assert len(datums) == len(lilo["loss_fn_outputs"]) == 128
    miles_samples = load_miles_samples(work, datums)

    abs_lm, k2_lm, abs_ls, abs_ms, k2_ls, k2_ms, n_tok = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for i, (datum, out) in enumerate(zip(datums, lilo["loss_fn_outputs"])):
        mask = torch.tensor(datum["mask"], dtype=torch.bool)
        sampled = torch.tensor(datum["sampled_logprobs"], dtype=torch.float64)[mask]
        lp_lilo = torch.tensor(out["logprobs"], dtype=torch.float64)[mask]
        s = miles_samples[i]
        mmask = s["local_loss_mask"].bool()
        lp_miles = s["train_log_probs"].double()[mmask]
        roll_miles = s["rollout_log_probs"].double()[mmask]
        assert lp_miles.numel() == lp_lilo.numel() == datum["response_len"], (
            i,
            lp_miles.numel(),
            lp_lilo.numel(),
            datum["response_len"],
        )
        assert torch.allclose(roll_miles, sampled, atol=1e-5), i
        d = lp_lilo - lp_miles
        abs_lm += d.abs().sum().item()
        k2_lm += 0.5 * (d**2).sum().item()
        abs_ls += (lp_lilo - sampled).abs().sum().item()
        abs_ms += (lp_miles - sampled).abs().sum().item()
        k2_ls += 0.5 * ((lp_lilo - sampled) ** 2).sum().item()
        k2_ms += 0.5 * ((lp_miles - sampled) ** 2).sum().item()
        n_tok += d.numel()
    assert n_tok == batch["meta"]["total_tokens"], (
        n_tok,
        batch["meta"]["total_tokens"],
    )
    return {
        "action_tokens": n_tok,
        "lilo_vs_miles_mean_abs": abs_lm / n_tok,
        "lilo_vs_miles_k2": k2_lm / n_tok,
        "lilo_vs_sampler_mean_abs": abs_ls / n_tok,
        "lilo_vs_sampler_k2": k2_ls / n_tok,
        "miles_vs_sampler_mean_abs": abs_ms / n_tok,
        "miles_vs_sampler_k2": k2_ms / n_tok,
    }


def advantage_check(work: Path) -> dict:
    """Miles per-token advantage vs Lilo's (r-mean)/(std+1e-6)/total_tokens."""
    batch = json.loads((work / "batch.json").read_text())
    datums = batch["datums"]
    total_tokens = batch["meta"]["total_tokens"]
    miles_adv: dict[int, float] = {}
    for i, s in load_miles_samples(work, datums).items():
        adv = s["advantages"].double()[s["local_loss_mask"].bool()]
        assert adv.numel() == 0 or torch.allclose(adv, adv[0].expand_as(adv), atol=1e-7)
        miles_adv[i] = float(adv[0]) if adv.numel() else 0.0
    ratios = []
    for i, datum in enumerate(datums):
        lilo_adv = [a for a, m in zip(datum["advantages"], datum["mask"]) if m]
        assert max(lilo_adv) - min(lilo_adv) < 1e-12
        if abs(lilo_adv[0]) > 1e-12:
            ratios.append(miles_adv[i] / lilo_adv[0])
    ratios_t = torch.tensor(ratios)
    return {
        "miles_over_lilo_advantage_ratio_mean": float(ratios_t.mean()),
        "miles_over_lilo_advantage_ratio_std": float(ratios_t.std()),
        "lilo_total_tokens": total_tokens,
        "n_nonzero_adv": len(ratios),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("~/work/graddiff").expanduser())
    ap.add_argument("--out", type=Path, default=Path("docs/graddiff_step0_results.md"))
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--miles-rerun",
        type=Path,
        default=Path("miles_dump2/miles"),
        help="relative to --work; second bit-identical Miles run (nondeterminism baseline)",
    )
    ap.add_argument("--lilo-dump", type=Path, default=Path("lilo_dump/lilo"))
    ap.add_argument(
        "--miles-dump", type=Path, default=Path("miles_dump/miles_dump/miles")
    )
    ap.add_argument(
        "--lilo-slot",
        type=int,
        default=0,
        help="adapter slot the Lilo client registered",
    )
    ap.add_argument(
        "--skip-forward",
        action="store_true",
        help="skip per-token logprob/advantage checks",
    )
    args = ap.parse_args()
    work = args.work

    lilo = load_arm(work / args.lilo_dump, lilo_name_pattern(args.lilo_slot))
    miles = load_arm(work / args.miles_dump, _MILES_NAME)
    miles2 = (
        load_arm(work / args.miles_rerun, _MILES_NAME)
        if (work / args.miles_rerun).exists()
        else None
    )
    assert set(lilo["grad"]) == set(miles["grad"]), set(lilo["grad"]) ^ set(
        miles["grad"]
    )
    names = sorted(lilo["grad"], key=lambda n: (int(_LAYER.match(n).group(1)), n))

    rows = []
    agg: dict[str, dict[str, list[float]]] = {}
    max_reduce_err = 0.0
    a_init_err = 0.0
    all_g_l, all_g_m, all_d_l, all_d_m, all_g_m2 = [], [], [], [], []
    for name in names:
        for arm in (lilo, miles):
            max_reduce_err = max(max_reduce_err, max(arm["reduce_check"][name]))
        assert all(
            a.shape == b.shape for a, b in zip(lilo["grad"][name], miles["grad"][name])
        ), name
        b_l, b_m = flat(lilo["before"][name]), flat(miles["before"][name])
        if name.endswith("linear_in"):
            a_init_err = max(a_init_err, float((b_l - b_m).abs().max()))
        else:
            assert b_l.abs().max() == 0 and b_m.abs().max() == 0, name
        g_l, g_m = flat(lilo["grad"][name]), flat(miles["grad"][name])
        d_l, d_m = flat(lilo["delta"][name]), flat(miles["delta"][name])
        g_m2 = flat(miles2["grad"][name]) if miles2 is not None else None
        if name.endswith("linear_in"):
            assert (
                g_l.abs().max() == 0
                and g_m.abs().max() == 0
                and d_l.abs().max() == 0
                and d_m.abs().max() == 0
            ), name
            continue
        all_g_l.append(g_l)
        all_g_m.append(g_m)
        all_d_l.append(d_l)
        all_d_m.append(d_m)
        n = g_l.numel()
        row = {
            "module": name.replace("language_model.decoder.", "").replace(
                ".linear_out", ""
            ),
            "type": module_type(name),
            "numel": n,
            "grad_norm_lilo": float(g_l.norm()),
            "grad_norm_miles": float(g_m.norm()),
            "grad_ratio_miles_over_lilo": float(g_m.norm() / g_l.norm()),
            "grad_cosine": cosine(g_l, g_m),
            "delta_norm_lilo": float(d_l.norm()),
            "delta_norm_miles": float(d_m.norm()),
            "delta_ratio_miles_over_lilo": float(d_m.norm() / d_l.norm()),
            "delta_cosine": cosine(d_l, d_m),
            "delta_over_lr_sqrtN_lilo": float(d_l.norm() / (LR * math.sqrt(n))),
            "delta_over_lr_sqrtN_miles": float(d_m.norm() / (LR * math.sqrt(n))),
            "sign_agreement": float(
                (torch.sign(d_l) == torch.sign(d_m)).double().mean()
            ),
            "layer": int(_LAYER.match(name).group(1)),
        }
        if g_m2 is not None:
            row["grad_cosine_miles_vs_miles_rerun"] = cosine(g_m, g_m2)
            row["grad_ratio_miles_rerun_over_miles"] = float(g_m2.norm() / g_m.norm())
            row["grad_cosine_lilo_vs_miles_rerun"] = cosine(g_l, g_m2)
            all_g_m2.append(g_m2)
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                agg.setdefault(row["type"], {}).setdefault(k, []).append(v)

    G_l, G_m, D_l, D_m = map(torch.cat, (all_g_l, all_g_m, all_d_l, all_d_m))
    summary = {
        "n_B_modules": len(rows),
        "a_init_max_abs_diff": a_init_err,
        "max_dp_reduce_vs_mean_rel_err": max_reduce_err,
        "total_grad_norm_lilo": float(G_l.norm()),
        "total_grad_norm_miles": float(G_m.norm()),
        "total_grad_ratio": float(G_m.norm() / G_l.norm()),
        "total_grad_cosine": cosine(G_l, G_m),
        "grad_ratio_min": min(r["grad_ratio_miles_over_lilo"] for r in rows),
        "grad_ratio_max": max(r["grad_ratio_miles_over_lilo"] for r in rows),
        "grad_cosine_min": min(r["grad_cosine"] for r in rows),
        "total_grad_cosine_miles_vs_miles_rerun": cosine(G_m, torch.cat(all_g_m2))
        if all_g_m2
        else None,
        "total_grad_ratio_miles_rerun_over_miles": float(
            torch.cat(all_g_m2).norm() / G_m.norm()
        )
        if all_g_m2
        else None,
        "total_delta_norm_lilo": float(D_l.norm()),
        "total_delta_norm_miles": float(D_m.norm()),
        "total_delta_cosine": cosine(D_l, D_m),
        "delta_sign_agreement": float(
            (torch.sign(D_l) == torch.sign(D_m)).double().mean()
        ),
        "forward": None if args.skip_forward else forward_parity(work),
        "advantages": None if args.skip_forward else advantage_check(work),
    }
    lilo_out = json.loads((work / "lilo_arm_outputs.json").read_text())
    summary["lilo_loss_sum"] = lilo_out["metrics"]["loss:sum"]
    summary["lilo_grad_norm_reported"] = lilo_out["optim_step"]["grad_norm:mean"]
    meta_files = glob.glob(
        str(work / "miles_dump/miles_dump/miles/rank0_*optim_step_meta.json")
    )
    if meta_files:
        with open(meta_files[0]) as f:
            summary["miles_optim_meta"] = json.load(f)

    def fmt(x: float) -> str:
        return f"{x:.4g}"

    lines = ["# Step-0 LoRA gradient / update parity: Lilo vs raw Miles", ""]
    lines.append("## Summary")
    for k, v in summary.items():
        if isinstance(v, dict):
            lines.append(f"- **{k}**:")
            for kk, vv in v.items():
                lines.append(f"  - {kk}: `{fmt(vv) if isinstance(vv, float) else vv}`")
        else:
            lines.append(f"- {k}: `{fmt(v) if isinstance(v, float) else v}`")
    lines += [
        "",
        "## Per module type (mean over modules)",
        "",
        "| type | n | grad ratio M/L | grad cos | Δ ratio M/L | Δ cos | Δ/(lr√N) L | Δ/(lr√N) M | sign agree |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, cols in agg.items():
        m = {k: sum(v) / len(v) for k, v in cols.items()}
        lines.append(
            f"| {t} | {len(cols['grad_cosine'])} | {fmt(m['grad_ratio_miles_over_lilo'])} | {fmt(m['grad_cosine'])} | "
            f"{fmt(m['delta_ratio_miles_over_lilo'])} | {fmt(m['delta_cosine'])} | {fmt(m['delta_over_lr_sqrtN_lilo'])} | "
            f"{fmt(m['delta_over_lr_sqrtN_miles'])} | {fmt(m['sign_agreement'])} |"
        )
    if all_g_m2:
        lines += [
            "",
            "## Per layer: B-grad cosine, Lilo-vs-Miles against the Miles-vs-Miles(rerun) noise floor",
            "",
            "| layer | modules | cos Lilo/Miles | cos Miles/Miles' | cos Lilo/Miles' | ratio M/L | ratio M'/M |",
            "|---|---|---|---|---|---|---|",
        ]
        by_layer: dict[int, list[dict]] = {}
        for r in rows:
            by_layer.setdefault(r["layer"], []).append(r)
        for layer, rs in sorted(by_layer.items()):
            mean = lambda k, rs=rs: sum(r[k] for r in rs) / len(rs)
            lines.append(
                f"| {layer} | {','.join(r['type'].replace('linear_', '') for r in rs)} | {fmt(mean('grad_cosine'))} | "
                f"{fmt(mean('grad_cosine_miles_vs_miles_rerun'))} | {fmt(mean('grad_cosine_lilo_vs_miles_rerun'))} | "
                f"{fmt(mean('grad_ratio_miles_over_lilo'))} | {fmt(mean('grad_ratio_miles_rerun_over_miles'))} |"
            )
    lines += [
        "",
        "## Per module (B = linear_out)",
        "",
        "| module | ‖g‖ Lilo | ‖g‖ Miles | ratio M/L | grad cos | ‖Δ‖ Lilo | ‖Δ‖ Miles | Δ cos | sign agree |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['module']} | {fmt(r['grad_norm_lilo'])} | {fmt(r['grad_norm_miles'])} | {fmt(r['grad_ratio_miles_over_lilo'])} | "
            f"{fmt(r['grad_cosine'])} | {fmt(r['delta_norm_lilo'])} | {fmt(r['delta_norm_miles'])} | {fmt(r['delta_cosine'])} | {fmt(r['sign_agreement'])} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    json_out = args.json_out or args.out.with_suffix(".json")
    json_out.write_text(
        json.dumps({"summary": summary, "modules": rows}, indent=1, default=str)
    )
    print("\n".join(lines[: lines.index("## Per module (B = linear_out)")]))


if __name__ == "__main__":
    main()
