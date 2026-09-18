"""Miles arm of the step-0 grad diff: one raw-Miles training step on the batch
regenerated in ``batch.json`` — no SGLang (``--load-debug-rollout-data`` forces
``debug_train_only``).

Run:
    MODAL_SERVER_URL=https://api.modal.com MODAL_PROFILE= MODAL_ENVIRONMENT=micah-dev \
        uv run modal run scripts/graddiff/miles_arm.py
"""

import json
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


def convert_batch() -> None:
    """batch.json datums -> Miles debug rollout .pt (rollout_id 0)."""
    sys.path.insert(0, "/root/miles")
    import torch
    from miles.utils.types import Sample

    with open(BATCH_JSON) as f:
        batch = json.load(f)

    samples = []
    for index, dm in enumerate(batch["datums"]):
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
            rollout_id=0,
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

    assert len(samples) == 128, len(samples)
    torch.save(
        {"rollout_id": 0, "metadata": {}, "samples": [s.to_dict() for s in samples]},
        ROLLOUT_PT,
    )
    print(f"wrote {ROLLOUT_PT}: {len(samples)} samples", flush=True)


# slim-hill-371ae1cfcb8e argv with graddiff overrides applied.
def train_cmd() -> str:
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
        "/graddiff/miles_dump/details",
        "--rollout-batch-size",
        "16",
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
        "128",
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
        "graddiff-step0-miles",
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
def run() -> str:
    import os

    commit = subprocess.run(
        ["git", "-C", "/root/miles", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"miles commit in image: {commit}", flush=True)

    convert_batch()

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
    runtime_env = {
        "env_vars": {
            "MILES_GRADDIFF_DUMP_DIR": "/graddiff/miles_dump",
            "LILO_DUMP_DIR": "/graddiff/lilo_dump/lilo",
            "PYTHONPATH": "/root/graddiff",
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HF_HOME": HF_MOUNT,
        }
    }
    job_id = client.submit_job(entrypoint=train_cmd(), runtime_env=runtime_env)
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
    print(run.remote())
