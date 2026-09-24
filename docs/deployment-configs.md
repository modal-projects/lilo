# Python deployment configs

A recipe subclasses `BaseConfig` and exports `config = Config()`. Settings are flat, untyped Python attributes. Backend options are ordinary dictionaries.

```python
from lilo.configuration import BaseConfig


class Config(BaseConfig):
    name = "my-9b"
    model = "Qwen/Qwen3.5-9B-Base"
    max_context_length = 16384
    backend = "miles"
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 16
    trainer_memory_mib = 65536
    trainer_max_clients_per_instance = 6
    inference_gpu = "H200"
    inference_max_replicas = 8
    miles_cfg = {
        "model_type": "qwen3.5-9B",
        "tensor_model_parallel_size": 4,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
    }
    sglang_cfg = {"max_running_requests": 32}


config = Config()
```

See the [9B LoRA recipe](../src/lilo/configs/qwen35_9b_lora_16k.py) and [4B FFT recipe](../src/lilo/configs/qwen35_4b_fft_64k.py) for complete examples. Deployment resolves the model's `main` revision to an exact commit; set `revision` only when you want a different revision.

## Variants

Override ordinary attributes directly. Use dotted `overrides` to change individual backend options:

```python
from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "my-9b-more-memory"
    trainer_memory_mib = 98304
    overrides = {"miles_cfg.max_tokens_per_gpu": 8192}


config = Config()
```

Each parent's settings and overrides apply before its child's. Constructor fields and overrides apply last: `Config(trainer_gpu="H200", overrides={"sglang_cfg.max_running_requests": 16})`. Assigning a dictionary or list replaces that value; `trainer_env = {}` clears inherited environment settings. Instances own independent copies of mutable values and can also be edited directly.

## Backend options

| Setting | Consumed by |
| --- | --- |
| `trainer_*`, `inference_*` | Modal GPU/CPU/memory allocation, scaling, timeouts and Lilo admission limits |
| `megatron_cfg` | Existing Megatron `EngineModelConfig`; provider, optimizer and distributed options use its native dictionaries |
| `miles_cfg` | Existing `MilesBackendConfig`; `cli_options` supplies additional Miles arguments |
| `sglang_cfg` | SGLang `ServerArgs` |

Modal and backend libraries validate their own options. Lilo checks integration requirements such as trainer slot capacity, supported training modes, and parallelism agreeing with allocated GPUs. It supplies managed model paths, context length and adapter settings; conflicting backend overrides are rejected.

`BaseConfig` does not enforce field types or reject arbitrary attributes. Extra backend options belong in the corresponding backend dictionary. A misspelled top-level attribute is ordinary Python data and may be unused.

Miles argument conversion lives in [miles_arguments.py](../src/lilo/backends/miles_arguments.py). SGLang receives `ServerArgs(**settings)` in its [worker entrypoint](../src/lilo/inference/sglang.py). Backend libraries validate native options when workers start; frontend config imports remain CPU-only.

## Resolve and launch

~~~text
load(config.py) → config: BaseConfig
  → resolve model commit
  → resolve_backend_settings(config, asset_path)
  → save DeploymentRecord with trainer_settings and inference_settings
  → deploy independent worker apps
  → update frontend references
~~~

The launcher consumes the saved settings. It does not reparse backend configuration or add another set of backend defaults. JSON decoding in a worker reconstructs the saved record, without importing the author's config file.

| File | Responsibility |
| --- | --- |
| [configuration.py](../src/lilo/configuration.py) | BaseConfig defaults, inheritance and overrides |
| [deployments.py](../src/lilo/deployments.py) | Python object loader, resolved records and config hashes |
| [backends/deployment.py](../src/lilo/backends/deployment.py) | Resolve backend settings before launch |
| [megatron_runtime/common/settings.py](../src/lilo/backends/megatron_runtime/common/settings.py) | Shared Megatron ownership rules and constructor dictionaries |
| [deployment_cli.py](../src/lilo/deployment_cli.py) | Model lookup, worker release selection and deploy ordering |
| [deployment_apps.py](../src/lilo/providers/modal/deployment_apps.py) | Resource declarations and launch using resolved settings |
| [deployment_records.py](../src/lilo/providers/modal/deployment_records.py) | Read saved records and route pool provisioning |
| [deployment_worker_app.py](../src/lilo/providers/modal/deployment_worker_app.py) | Deploy one trainer or inference provisioner |

## Multi-node Miles

[qwen38_27b_lora_256k.py](../src/lilo/configs/qwen38_27b_lora_256k.py) configures two nodes with eight H200s per node, TP2 and CP8. The trainer app uses Modal's clustered launcher and RDMA. Every rank mounts the same volumes; rank 0 starts the engine, and the other ranks join Ray using the launcher merged in #39. The driver receives the Ray address. GPU counts cannot be independently overridden through Miles options.

This path has CPU construction/topology tests. This refactor has not been redeployed or tested on multiple GPU nodes.

## Deploy and update

~~~bash
lilo config init --preset qwen35-9b-lora-16k > my_model.py
lilo config validate my_model.py
lilo deploy my_model.py
~~~

Validation resolves backend settings and checks Lilo integration constraints without loading GPU libraries or provisioning compute. Backend option support and GPU memory capacity still require worker startup.

The checked-in [deploy_models.sh](../scripts/deploy_models.sh) lists the complete active config set. Add a config path there, then run it. The deployment command owns frontend selection and worker-code updates:

~~~bash
./scripts/deploy_models.sh --app my-lilo --env dev --region us-west
./scripts/deploy_models.sh --refresh-trainer qwen35-9b-lora-16k
./scripts/deploy_models.sh --refresh-inference qwen35-9b-lora-16k
~~~

These settings are not model-config fields. Secret/volume names come from provider defaults. Credentials remain in Modal secrets.

One frontend serves all models through Tinker’s `base_model`. When recipes share a model, the first matching recipe in the `lilo deploy` argument list is used. Training also matches the requested LoRA/FFT mode; base sampling uses the first recipe regardless of training mode. To select a specific recipe, pass its definition ID from `/api/v1/lilo/deployments` as `base_model`. All supplied definitions are listed, including retained generations. Current recipes precede retained ones; existing jobs keep their saved definition IDs.

## Hashes and update isolation

Hashes identify settings, not compatibility with a source checkout:

- generation identifies the saved configuration, resolved backend settings, platform settings and code releases.
- trainer_hash includes trainer/model/platform settings, resolved trainer settings, Miles commit and its recorded code release.
- inference_hash includes inference/model/platform settings, resolved inference settings and its recorded code release.

SHA-256 hashes sorted JSON. Trainer and inference app names use the first 24 hex characters; definition IDs use the first 16 characters of generation. There is no whole-source fingerprint.

An inference-only change reuses the trainer app. A trainer batch-setting change reuses the inference provisioner. Changes to adapter rank/targets update both. Code-only upgrades use the refresh flags; later ordinary deploys retain those releases.

The frontend carries its resolved configurations and exposes them through the Modal `deployment_manifest` function. The CLI reads that function to retain existing definitions and worker releases, and checks worker existence directly with Modal. There is no separate deployment Dict, pending journal, or deployment lock. Run deploy commands sequentially.

Old jobs keep their recorded worker apps. Deployment retries discover and reuse workers already created successfully. Trainer limits are enforced per saved definition; old and new definitions may use their configured capacity simultaneously while old jobs finish. Pools remain associated with full job configurations, so new jobs may get separate pools even if they share an inference provisioner.

Old worker app definitions are retained; automatic cleanup is not implemented. Changes to the frontend/worker protocol still need deliberate compatibility handling. Earlier draft config formats require migration or a fresh frontend. See [validation history](deployment-validation.md) for the distinction between current CPU checks and historical GPU runs.
