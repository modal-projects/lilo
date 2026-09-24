# Python deployment configs

A recipe subclasses `BaseConfig` and exports `config = Config()`. Model settings are ordinary class attributes; trainer and inference settings are dictionaries. No `Compute`, `Model`, or `Routing` constructors are needed.

```python
from lilo.configuration import BaseConfig


class Config(BaseConfig):
    name = "my-9b"
    model = "Qwen/Qwen3.5-9B-Base"
    max_context_length = 16384

    trainer = {
        "gpu": "H100",
        "gpus_per_node": 4,
        "cpu": 16,
        "memory_mib": 65536,
        "max_clients_per_instance": 6,
        "config": {
            "model_type": "qwen3.5-9B",
            "tensor_model_parallel_size": 4,
            "max_lora_slots": 6,
            "max_lora_rank": 32,
        },
    }
    inference = {"gpu": "H200", "max_replicas": 8}


config = Config()
```

See the [9B LoRA recipe](../src/lilo/configs/qwen35_9b_lora_16k.py) and [4B FFT recipe](../src/lilo/configs/qwen35_4b_fft_64k.py) for complete examples. Deployment resolves the model's `main` revision to an exact commit; set `revision` only when you want a different revision.

## Variants

Use Python inheritance to change a recipe. Extend a section explicitly when you want to keep its other settings:

```python
from lilo.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "my-9b-more-memory"
    trainer = {**Parent.trainer, "memory_mib": 98304}


config = Config()
```

Assigning a new dictionary replaces that section. There is no implicit deep merge or dotted override language. Extend backend options with `{**Parent.trainer["config"], "max_tokens_per_gpu": 8192}`. Constructor arguments can also override fields: `Config(name="another-run")`.

Construction copies the recipe's dictionaries and validates Lilo-owned fields. Unknown fields, invalid types, negative capacities, and inconsistent scaling limits fail before deployment. The validated instance has attribute access (`config.trainer.gpu`); backend options stay dictionaries. Instances do not share mutable options with each other or with their recipe class.

## Ownership and validation

| Setting | Owner and behavior |
| --- | --- |
| trainer / inference resources | GPU type, GPUs per node, CPU and memory directly in each section; `nodes` is trainer-only. Unknown keys such as `memroy_mib` are rejected. |
| trainer | Maximum instances/clients, publication concurrency and function timeout. Trainers start on demand; there is no min_instances field. |
| inference | Replica scaling and startup_timeout_s, passed to the Modal server and startup health checks. There is no unused timeout_s. Each replica uses one node. |
| trainer.config | Existing MilesBackendConfig or EngineModelConfig fields, plus their explicit extra-option dictionaries. |
| inference.config | SGLang ServerArgs fields. Lilo reserves paths, context, topology and adapter settings that must agree with its own configuration. |

Compute topology is configured directly in `trainer`; Miles receives actor_num_nodes and actor_num_gpus_per_node from it. Setting those again in backend options is rejected.

Megatron's provider_overrides, optimizer_overrides and distributed_overrides may add backend fields, but may not replace Lilo-owned fields. For example, put the learning rate in optimizer={"lr": ...}; optimizer_overrides={"lr": ...} is rejected. The same settings builders are used during validation and worker construction. The provider is constructed with dataclasses.replace, without an override-by-setattr pass.

Miles has a cli_options dictionary for additional Miles arguments. Its argument conversion is isolated in [miles_arguments.py](../src/lilo/backends/miles_arguments.py), because Miles exposes an argparse interface. SGLang uses ServerArgs(**settings) directly in the worker-only [sglang.py](../src/lilo/inference/sglang.py) entrypoint.

Backend libraries validate their own extra options when workers start. Lilo does not maintain another schema for every upstream tuning option. Frontend config imports remain CPU-only.

## Resolve and launch

~~~text
load(config.py) → config: BaseConfig
  → validate typed compute/scaling/model settings
  → resolve model commit
  → resolve_backend_settings(config, asset_path)
  → save DeploymentRecord with trainer_settings and inference_settings
  → deploy independent worker apps
  → update frontend references
~~~

The launcher consumes the saved settings. It does not reparse backend configuration or add another set of backend defaults. JSON decoding in a worker reconstructs the saved record, without importing the author's config file.

| File | Responsibility |
| --- | --- |
| [configuration.py](../src/lilo/configuration.py) | BaseConfig and validation of model, trainer, and inference settings |
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

Validation checks orchestration and integration constraints without loading GPU libraries or provisioning compute. Backend option support and GPU memory capacity still require worker startup.

The checked-in [deploy_models.sh](../scripts/deploy_models.sh) lists the complete active config set. Add a config path there, then run it. The deployment command owns frontend selection and worker-code updates:

~~~bash
./scripts/deploy_models.sh --app my-lilo --env dev --region us-west
./scripts/deploy_models.sh --refresh-trainer qwen35-9b-lora-16k
./scripts/deploy_models.sh --refresh-inference qwen35-9b-lora-16k
~~~

These settings are not model-config fields. Secret/volume names come from provider defaults. Credentials remain in Modal secrets.

One frontend serves all models through Tinker's base_model. `default = True` selects among multiple training configurations for one model; `sampling_default = True` selects a sampling configuration when LoRA/FFT configurations coexist.

## Hashes and update isolation

Hashes identify settings, not compatibility with a source checkout:

- generation identifies the saved configuration, resolved backend settings, platform settings and code releases. Routing defaults are excluded.
- trainer_hash includes trainer/model/platform settings, resolved trainer settings, Miles commit and its recorded code release.
- inference_hash includes inference/model/platform settings, resolved inference settings and its recorded code release.

SHA-256 hashes sorted JSON. Trainer and inference app names use the first 24 hex characters; definition IDs use the first 16 characters of generation. There is no whole-source fingerprint.

An inference-only change reuses the trainer app. A trainer batch-setting change reuses the inference provisioner. Changes to adapter rank/targets update both. Code-only upgrades use the refresh flags; later ordinary deploys retain those releases.

Old jobs keep their recorded worker apps. Deployment retries reuse workers already created successfully. Trainer limits are enforced per saved definition; old and new definitions may use their configured capacity simultaneously while old jobs finish. Pools remain associated with full job configurations, so new jobs may get separate pools even if they share an inference provisioner.

Old worker app definitions are retained; automatic cleanup is not implemented. Changes to the frontend/worker protocol still need deliberate compatibility handling. Earlier draft manifests need migration or a fresh frontend/registry. See [validation history](deployment-validation.md) for the distinction between current CPU checks and historical GPU runs.
