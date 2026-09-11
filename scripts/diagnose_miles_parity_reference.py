"""Compare retained Miles PEFT snapshots using explicit LoRA updates on a Hugging Face model.

Run with modal run; --dtype-name float32 disables TF32 for a higher-precision reference.
This reads snapshots and performs forwards only; it does not update weights.
"""

import json
from pathlib import Path

from lilo.providers.modal.miles_image import image

import modal

app = modal.App("lilo-hf-parity-reference")
assets = modal.Volume.from_name("lilo-model-assets")
bulletin = modal.Volume.from_name("lilo-snapshot-bulletin", version=2)


@app.function(
    image=image,
    gpu="H200",
    volumes={"/assets": assets, "/bulletin": bulletin},
    timeout=1200,
)
def reference(tokens, refs, dtype_name, checkpoint):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=dtype, device_map="cuda", attn_implementation="eager"
    ).eval()
    ids = torch.tensor([tokens], device="cuda")

    def forward():
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        return F.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None]).squeeze(1).cpu().tolist()

    results = [{"ref": "base", "logprobs": forward()}]
    modules = dict(model.named_modules())
    for ref in refs:
        path = Path("/bulletin") / ref
        cfg = json.loads((path / "adapter_config.json").read_text())
        weights = {k: v.to(dtype) for k, v in load_file(str(path / "adapter_model.safetensors"), device="cuda").items()}
        hooks = []
        missing = []
        matched = []
        for key, a in weights.items():
            if not key.endswith(".lora_A.weight"):
                continue
            name = key.removesuffix(".lora_A.weight")
            b = weights[name + ".lora_B.weight"]
            if name not in modules:
                missing.append(name)
                continue
            scale = cfg["lora_alpha"] / cfg["r"]

            def hook(module, inputs, output, a=a, b=b, scale=scale):
                return output + F.linear(F.linear(inputs[0].to(a.dtype), a), b) * scale

            hooks.append(modules[name].register_forward_hook(hook))
            matched.append(name)
        print("Reference", ref, "matched", len(matched), "missing", missing, flush=True)
        if any("language_model" in name or name == "lm_head" for name in missing):
            raise RuntimeError("Missing text adapter modules")
        try:
            results.append(
                {
                    "ref": ref,
                    "logprobs": forward(),
                    "matched": matched,
                    "missing": missing,
                }
            )
        finally:
            for hook in hooks:
                hook.remove()
        del weights
    return results


@app.local_entrypoint()
def main(
    report: str,
    dtype_name: str = "bfloat16",
    additional_report: str = "",
    checkpoint: str = "/assets/Qwen3.5-9B-Base",
):
    if dtype_name not in {"bfloat16", "float32"}:
        raise ValueError("dtype-name must be bfloat16 or float32")
    path = Path(report)
    record = json.loads(path.read_text())
    if "probe_tokens" not in record:
        raise ValueError("Report must contain saved probe tokens")
    diagnostic = record.get("diagnostic_parity", [])
    artifacts = diagnostic[-1]["artifacts"] if diagnostic else record["artifacts"][-1]
    if additional_report:
        extra = json.loads(Path(additional_report).read_text())
        if extra["datasets"] != record["datasets"]:
            raise ValueError("Additional report must use the same probe dataset")
        artifacts = artifacts + extra["artifacts"][-1]
    refs = [f"{a['model_id']}/weight_v{a['publish_version']:06d}" for a in artifacts]
    result = reference.remote(record["probe_tokens"], refs, dtype_name, checkpoint)
    out = path.with_suffix(f".hf-{dtype_name}-reference.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
