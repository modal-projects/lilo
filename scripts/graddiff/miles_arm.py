"""Miles arm of the step-0 grad diff: one raw-Miles training step on the batch
regenerated in ``batch.json`` — no SGLang (``--load-debug-rollout-data`` forces
``debug_train_only``).

Run:
    MODAL_SERVER_URL=https://api.modal.com MODAL_PROFILE= MODAL_ENVIRONMENT=micah-dev \
        uv run modal run scripts/graddiff/miles_arm.py
"""

import json
import os
import shlex
import subprocess
import sys
import time

import modal

app = modal.App("lilo-graddiff-miles")

image = modal.Image.from_registry("radixark/miles:dev-202608120325").add_local_file(
    __import__("pathlib").Path(__file__).parent / "miles_hook.py",
    "/root/graddiff/miles_hook.py",
)

GRADDIFF_VOL = modal.Volume.from_name("lilo-graddiff", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Path baked into slim-hill's config: Modal mounts the volume by object id here.
HF_SNAPSHOT = "/__modal/volumes/vo-jeCR35EcAFzh7lWyHqWh06/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HF_MOUNT = "/__modal/volumes/vo-jeCR35EcAFzh7lWyHqWh06"

BATCH_JSON = "/graddiff/batch/batch.json"
ROLLOUT_PT = "/graddiff/miles_batch/rollout_0.pt"
PROMPT_PARQUET = "/graddiff/miles_batch/prompts.parquet"

# Optional sub-batch run: DATUM_RANGE="a:b" slices batch.json datums;
# MILES_DUMP_DIR/WANDB_NAME/GBS/RBS override the dump dir, run name, and
# batch sizes for sub-batch comparisons.
DATUM_RANGE = os.environ.get("DATUM_RANGE")
MILES_DUMP_DIR = os.environ.get("MILES_DUMP_DIR", "/graddiff/miles_dump2")
WANDB_NAME = os.environ.get("WANDB_NAME", "graddiff-step0-miles-rerun")
GBS = int(os.environ.get("GBS", "128"))
RBS = int(os.environ.get("RBS", "16"))


def convert_batch(datum_range: str | None = None) -> None:
    """batch.json datums -> Miles debug rollout .pt (rollout_id 0)."""
    sys.path.insert(0, "/root/miles")
    import torch
    from miles.utils.types import Sample

    with open(BATCH_JSON) as f:
        batch = json.load(f)

    entries = batch["datums"]
    if datum_range:
        lo, hi = (int(x) for x in datum_range.split(":"))
        entries = entries[lo:hi]
    samples = []
    for index, dm in enumerate(entries):
        mask = dm["mask"]
        first = next(i for i, x in enumerate(mask) if x)
        assert all(x == 1 for x in mask[first:]), "mask is not a contiguous tail"
        response_length = len(mask) - first
        tokens = dm["input_tokens"] + [dm["target_tokens"][-1]]
        assert len(tokens) == len(dm["input_tokens"]) + 1
        rollout_log_probs = dm["sampled_logprobs"][-response_length:]
        sample = Sample(
            group_index=dm["group_idx"],
            index=index,
            rollout_id=dm["traj_idx"],
            prompt="",
            tokens=tokens,
            response="",
            response_length=response_length,
            reward=dm["reward"],
            loss_mask=[1] * response_length,
            rollout_log_probs=rollout_log_probs,
            status=Sample.Status.COMPLETED,
        )
        samples.append(sample)

    assert len(samples) == len(entries), len(samples)
    import os

    os.makedirs(os.path.dirname(ROLLOUT_PT), exist_ok=True)

    # RolloutManager still instantiates a prompt dataset in debug-train-only;
    # give it a real file (contents are never sampled).
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_SNAPSHOT, trust_remote_code=True)
    seen = set()
    prompts = []
    for dm in batch["datums"]:
        if dm["group_idx"] in seen:
            continue
        seen.add(dm["group_idx"])
        prompt_len = len(dm["mask"]) - int(sum(dm["mask"]))
        prompts.append(tok.decode(dm["input_tokens"][:prompt_len]))
    convs = [[{"role": "user", "content": p}] for p in prompts]
    pd.DataFrame({"prompt": convs, "label": ["0"] * len(convs)}).to_parquet(
        PROMPT_PARQUET
    )
    print(f"wrote {PROMPT_PARQUET}: {len(prompts)} prompts", flush=True)

    torch.save(
        {"rollout_id": 0, "metadata": {}, "samples": [s.to_dict() for s in samples]},
        ROLLOUT_PT,
    )
    print(f"wrote {ROLLOUT_PT}: {len(samples)} samples", flush=True)


# slim-hill-371ae1cfcb8e argv with graddiff overrides applied.
def train_cmd(miles_dump_dir: str, wandb_name: str, gbs: int, rbs: int) -> str:
    args = [
        "--spec",
        "miles_plugins.models.qwen3_5",
        "get_qwen3_5_spec",
        "--disable-bias-linear",
        "--qk-layernorm",
        "--group-query-attention",
        "--num-attention-heads",
        "16",
        "--num-query-groups",
        "4",
        "--kv-channels",
        "256",
        "--num-layers",
        "32",
        "--hidden-size",
        "4096",
        "--ffn-hidden-size",
        "12288",
        "--normalization",
        "RMSNorm",
        "--apply-layernorm-1p",
        "--position-embedding-type",
        "rope",
        "--norm-epsilon",
        "1e-6",
        "--rotary-percent",
        "0.25",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--vocab-size",
        "248320",
        "--rotary-base",
        "10000000",
        "--attention-output-gate",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "8",
        "--rollout-num-gpus",
        "8",
        "--rollout-num-gpus-per-engine",
        "1",
        "--train-backend",
        "megatron",
        "--tensor-model-parallel-size",
        "4",
        "--sequence-parallel",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        # overrides: one step, debug data in, no sglang, no save
        "--num-rollout",
        "1",
        "--load-debug-rollout-data",
        "/graddiff/miles_batch/rollout_{rollout_id}.pt",
        "--dump-details",
        miles_dump_dir + "/details",
        "--rollout-batch-size",
        str(rbs),
        "--n-samples-per-prompt",
        "8",
        "--rollout-max-response-len",
        "4096",
        "--rollout-temperature",
        "1.0",
        "--rollout-shuffle",
        "--rollout-top-p",
        "1.0",
        "--use-fault-tolerance",
        "--prompt-data",
        PROMPT_PARQUET,
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--apply-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--rollout-health-check-interval",
        "30",
        "--rollout-health-check-timeout",
        "30",
        "--rollout-health-check-first-wait",
        "1800",
        "--sglang-mem-fraction-static",
        "0.8",
        "--sglang-cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
        "16",
        "24",
        "32",
        "--sglang-max-running-requests",
        "32",
        "--sglang-reasoning-parser",
        "qwen3",
        "--advantage-estimator",
        "grpo",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--kl-loss-type",
        "low_var_kl",
        "--kl-loss-coef",
        "0.0",
        "--kl-coef",
        "0.0",
        "--entropy-coef",
        "0.0",
        "--calculate-per-token-loss",
        # dropped: --use-tis, --tis-clip; added:
        "--use-rollout-logprobs",
        "--over-sampling-batch-size",
        "24",
        "--dynamic-sampling-filter-path",
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        "--balance-data",
        "--global-batch-size",
        str(gbs),
        "--lr",
        "0.0001",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--optimizer",
        "adam",
        "--lora-rank",
        "32",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.0",
        "--target-modules",
        (
            "language_model.decoder.layers.*.self_attention.linear_qkv,"
            "language_model.decoder.layers.*.self_attention.linear_proj,"
            "language_model.decoder.layers.*.mlp.linear_fc1,"
            "language_model.decoder.layers.*.mlp.linear_fc2"
        ),
        "--no-gradient-accumulation-fusion",
        "--sglang-lora-backend",
        "triton",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--attention-softmax-in-fp32",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-backend",
        "flash",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--qkv-format",
        "thd",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        "16384",
        "--update-weight-buffer-size",
        "2147483648",
        "--hf-checkpoint",
        HF_SNAPSHOT,
        "--megatron-to-hf-mode",
        "bridge",
        "--n-samples-per-eval-prompt",
        "2",
        "--eval-max-response-len",
        "4096",
        "--eval-top-p",
        "1.0",
        "--use-wandb",
        "--wandb-project",
        "miles-lora-longcontext",
        "--wandb-group",
        "graddiff-step0",
        "--wandb-exp-name",
        wandb_name,
        "--disable-wandb-random-suffix",
        "--custom-megatron-before-train-step-hook-path",
        "miles_hook.before_train_step",
    ]
    return f"python3 /root/miles/train.py {shlex.join(args)}"


@app.function(
    image=image,
    gpu="H200:8",
    timeout=6 * 3600,
    volumes={HF_MOUNT: HF_CACHE, "/graddiff": GRADDIFF_VOL},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run(
    datum_range: str | None = None,
    miles_dump_dir: str = MILES_DUMP_DIR,
    wandb_name: str = WANDB_NAME,
    gbs: int = GBS,
    rbs: int = RBS,
    lilo_dump_dir: str = "/graddiff/lilo_dump/lilo",
) -> str:
    import os

    def _sh(cmd: str) -> str:
        r = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, check=False
        )
        return (r.stdout + r.stderr).strip()

    print(
        "image versions:",
        _sh(
            "git -C /root/miles rev-parse HEAD; "
            "git -C /root/Megatron-LM rev-parse HEAD; "
            "pip list 2>/dev/null | grep -iE 'megatron|transformer.engine|miles'"
        ),
        flush=True,
    )
    # Match slim-hill's env: miles ~ef3807c (main 2026-09-16),
    # megatron-core==0.19.0+73b54618f (packed-GDN), megatron-bridge==0.5.0+40b93089.
    print(
        "miles sync:",
        _sh(
            "git -C /root/miles fetch --depth 1 origin ef3807c0ef659d7c6d8494c4933bd7ee0332700f && "
            "git -C /root/miles checkout -f FETCH_HEAD && "
            "git -C /root/miles rev-parse HEAD"
        ),
        flush=True,
    )
    print(
        "mglm sync:",
        _sh(
            "git -C /root/Megatron-LM fetch --depth 200 origin miles-main && "
            "git -C /root/Megatron-LM checkout -f 73b54618f && "
            "git -C /root/Megatron-LM rev-parse HEAD"
        ),
        flush=True,
    )
    print(
        "bridge sync:",
        _sh(
            "pip install --no-deps "
            "'git+https://github.com/radixark/Megatron-Bridge.git@40b930897717941cfe2bd9806f417e28bf1bfa65' && "
            "pip list 2>/dev/null | grep megatron-bridge"
        ),
        flush=True,
    )
    commit = _sh("git -C /root/miles rev-parse HEAD")

    convert_batch(datum_range)

    env = {**os.environ, "PYTHONPATH": "/root/graddiff"}
    subprocess.Popen(["ray", "start", "--head", "--dashboard-host=0.0.0.0"], env=env)
    # wait for head
    import ray

    for _ in range(60):
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            break
        except ConnectionError:
            time.sleep(2)
    else:
        raise RuntimeError("ray head never came up")
    print("ray head up; nodes:", len(ray.nodes()), flush=True)

    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient("http://127.0.0.1:8265")
    # Mirror miles' own ray-job runtime env (external_utils/command_utils.py):
    # PYTHONUNBUFFERED, CUDA_DEVICE_MAX_CONNECTIONS, NCCL_NVLS_ENABLE, MASTER_ADDR.
    runtime_env = {
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "1"),
            "no_proxy": "127.0.0.1",
            "MASTER_ADDR": "127.0.0.1",
            "MILES_GRADDIFF_DUMP_DIR": miles_dump_dir,
            "LILO_DUMP_DIR": lilo_dump_dir,
            "PYTHONPATH": "/root/graddiff"
            + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HF_HOME": HF_MOUNT,
        }
    }
    job_id = client.submit_job(
        entrypoint=train_cmd(miles_dump_dir, wandb_name, gbs, rbs),
        runtime_env=runtime_env,
    )
    print("submitted ray job", job_id, flush=True)

    status = "PENDING"
    last_log = ""
    while status not in ("SUCCEEDED", "FAILED", "STOPPED"):
        try:
            status = client.get_job_status(job_id).value
            last_log = client.get_job_logs(job_id)
        except Exception as e:  # noqa: BLE001
            print("poll error:", e, flush=True)
        time.sleep(15)
    print("final status:", status, flush=True)
    print(last_log[-30000:], flush=True)

    out = {"status": status, "miles_commit": commit, "job_id": job_id}
    with open("/graddiff/miles_arm_run.json", "w") as f:
        json.dump(out, f, indent=1)
    return json.dumps({"status": status, "miles_commit": commit})


@app.local_entrypoint()
def main():
    print(
        run.remote(
            os.environ.get("DATUM_RANGE"),
            os.environ.get("MILES_DUMP_DIR", "/graddiff/miles_dump2"),
            os.environ.get("WANDB_NAME", "graddiff-step0-miles-rerun"),
            int(os.environ.get("GBS", "128")),
            int(os.environ.get("RBS", "16")),
            os.environ.get("LILO_DUMP_DIR", "/graddiff/lilo_dump/lilo"),
        )
    )
