# Python deployment configs

Each deployment is a Python file exporting a `Config` dataclass that inherits from `BaseConfig`. The class contains the model, trainer, inference and Modal settings. There is no YAML loader or model catalog.

The layout follows the Python recipe approach used by [training-gym](https://github.com/modal-labs/training-gym) and the [multinode training guide](https://github.com/modal-labs/multinode-training-guide/blob/main/nemo-rl/configs/llama3_1_8b_math_2node.py). Lilo's config classes use standard-library dataclasses; importing either project is not required.

## Define a deployment

Start with an example:

```bash
lilo config init --preset qwen35-9b-lora-16k > src/lilo/configs/my_model.py
```

The generated file imports a packaged config and subclasses it. Customize it using ordinary Python:

```python
from dataclasses import dataclass
from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


@dataclass(kw_only=True)
class Config(ParentConfig):
    name: str = "my-9b-64k"

    def __post_init__(self):
        self.model.max_context_length = 65536
        self.trainer.resources.gpu = "H200:8"
        self.trainer.config["options"]["tensor_model_parallel_size"] = 8
        self.trainer.config["options"]["max_tokens_per_gpu"] = 65536
        self.inference.scaling.max_replicas = 6
```

When a parent defines `__post_init__`, call `super().__post_init__()` before your changes. Mutable defaults use `field(default_factory=...)`, so modifying one instance does not change another config. New deployments can also inherit directly from `BaseConfig` and supply `Model`, `Trainer` and `Inference` fields; the packaged [16K example](../src/lilo/configs/qwen35_9b_lora_16k.py) shows the complete structure.

Python imports provide reuse. There is no `extends` key or implicit dictionary merge. Backend dictionaries support normal Python operations such as `update()` or `|`. Config files execute as Python when loaded; keep provisioning and training calls outside them. Sibling imports are available while loading a file.

All 14 previous presets are available under [`src/lilo/configs/`](../src/lilo/configs), with the same model, resource and backend settings. The [64K config](../src/lilo/configs/qwen35_9b_lora_64k.py) inherits from the 16K config. The 9B LoRA and 4B FFT configs directly pin the model revisions used in the earlier GPU checks; derived configs inherit them. There is no separate `deployments/` wrapper directory.

`model.revision` is the Hugging Face commit or branch containing the base weights and tokenizer. You can omit it to use `"main"`; the CLI resolves that branch to an exact commit before deployment. A fixed commit makes repeated deployments use the same files even if the repository's main branch changes. This is separate from training steps and published adapter versions.

## Deploy the complete active set

Use Python 3.12, matching the serialized trainer and inference images:

```bash
lilo config validate src/lilo/configs/my_model.py
lilo config resolve src/lilo/configs/my_model.py --output /tmp/deployment.json
lilo deploy src/lilo/configs/model_a.py src/lilo/configs/model_b.py
```

`validate` loads the Python classes and checks shared frontend settings. It does not start backend libraries or prove that the model fits in GPU memory. `resolve` additionally pins model revisions and emits the deployment records as JSON; it does not provision compute.

For a checked-in list, edit [`scripts/deploy_models.sh`](../scripts/deploy_models.sh). Add a Python config file and its path to the `deployment_files` array, then run:

```bash
./scripts/deploy_models.sh
```

Supply every configuration that should remain available to new clients. Omitted configurations are retained for existing jobs but removed from new-client selection. All files in the list must agree on shared frontend, region, secrets, storage and lifecycle settings. Pin `LILO_MILES_COMMIT` for repeatable deployments. Credentials remain in Modal secrets; the config contains only secret names.

## Code path

```text
lilo deploy config.py
  deployment_cli.main()
    compile_configs()
      deployments.load() → execute config.py → Config()
      resolve model tag to a Hugging Face commit
      DeploymentRecord.create() → copy settings and compute hash
    deploy()
      retain existing job configurations
      save/pass manifest JSON
      modal deploy -m lilo.providers.modal.app
```

[`load()`](../src/lilo/deployments.py) executes the file and instantiates its exported `Config` subclass. The returned dataclass goes directly to the orchestration code. There is no dictionary-to-config conversion on this path.

`DeploymentRecord` adds the code identity, pinned Miles revision, configuration hash and active status. JSON is used only to store records and pass them to other processes. Pydantic reconstructs the standard dataclasses when reading those records; it does not import or run the user's config file in a GPU worker. Records preserve the full computed settings rather than a reference to the original Python file.

| File | Responsibility |
| --- | --- |
| [`deployments.py`](../src/lilo/deployments.py) | Dataclasses, Python file loader, saved record and shared-frontend checks |
| [`deployment_cli.py`](../src/lilo/deployment_cli.py) | Revision lookup, manifest updates and Modal deployment |
| [`deployment_apps.py`](../src/lilo/providers/modal/deployment_apps.py) | Trainer functions, inference server classes and process startup |
| [`app.py`](../src/lilo/providers/modal/app.py) | Shared frontend and `app.include()` for generated trainers |
| [`deployment_pool_app.py`](../src/lilo/providers/modal/deployment_pool_app.py) | Construct an inference pool from its saved record |
| [`backends/deployment.py`](../src/lilo/backends/deployment.py) | Select the backend configuration reader |

`build_trainer_app(record)` configures resources, secrets, storage and limits. The shared app includes its generated trainer function. When that function starts, `run_trainer()` obtains backend settings and passes them as `LILO_BACKEND_CONFIG` to the existing Miles or Megatron executor.

Inference pools are separate apps created on demand. `build_rollout_app()` constructs a server from the saved configuration. Startup launches SGLang with native options, waits for its health endpoint, then starts the LoRA or FFT sidecar.

## Backend options

`trainer.config` and `inference.config` remain open dictionaries. Adding an upstream option does not require adding a deployment dataclass field. Backend readers check values used by Lilo's integration, while installed backend libraries check native options at worker startup.

For Miles, `trainer.config` contains:

```python
{
    "model_args": "qwen3.5-9B",
    "options": {
        "tensor_model_parallel_size": 4,
        "multi_lora_n_adapters": 6,
        "recompute_granularity": "full",
    },
}
```

`model_args` selects a Miles architecture preset; it can be omitted when explicit architecture options are provided. Miles's actual argument parser validates native options. Lilo also reads parallelism, adapter slots/rank and targets for admission and adapter export.

For Megatron, the [FFT example](../src/lilo/configs/qwen35_4b_fft_64k.py) separates:

- `runtime`: Lilo's training loop, packing and parallelism settings.
- `provider`: attributes assigned to the actual Megatron Bridge model provider.
- `optimizer`: optimizer settings, including additional native Megatron constructor options.
- `distributed`: additional `DistributedDataParallelConfig` constructor options.

These section names are owned by Lilo. Native provider/optimizer/distributed fields do not need a Lilo allowlist. Shared precision and parallelism controls remain protected so Lilo's packing and collectives agree with Megatron. Tinker optimizer steps use Adam parameters supplied by the client. Native optimizer/distributed settings are recorded and checked for exact FFT checkpoint resume.

`inference.config` contains SGLang options directly, such as `tp_size`, `mem_fraction_static` and `max_running_requests`. The real SGLang parser checks them during startup. Miles and SGLang support ordinary scalar, boolean and list arguments; unsupported custom/repeated argparse actions fail explicitly.

Backend readers still check managed paths, GPU topology, adapter capacity and communication settings. Python configs do not bypass backend compatibility, adapter export/load requirements or available GPU memory.

## Routing and updates

The frontend URL is shared across models. Clients select the model through `base_model`:

```python
service = tinker.ServiceClient(base_url=lilo_url, api_key=api_key)
training = service.create_lora_training_client(
    base_model="Qwen/Qwen3.5-9B-Base", rank=32,
)
```

If multiple configurations serve the same model and training mode, select one with `routing.default=True`. Otherwise client creation reports the ambiguity. `routing.sampling_default` resolves sampling-only selection across LoRA and FFT configurations. Training-derived sampling remains attached to the client's saved configuration.

Every client stores its selected definition ID. Changing routing defaults affects new clients. Changing compute or backend settings creates a new configuration ID, and old configurations remain registered for existing jobs and checkpoints. The CLI serializes applies and retains interrupted deployments for recovery.

The shared app is deployed on each apply. Existing trainer functions use stable captured configuration JSON, and unchanged inference pools retain their apps. This is the shared-app architecture, not independent trainer-app deployment. Source/runtime upgrades still require a separate frontend in this draft. The `src/lilo/configs/` directory is excluded from runtime fingerprints and image source mounts. Its computed values are stored and hashed in each deployment record, so editing a config does not count as a runtime-code upgrade. Backend code changes still do.

Existing resource names (`lilo-yaml`, `*-yaml-deployments`, and the `yaml_` definition prefix) are retained so this authoring change does not rename saved resources. They no longer indicate a YAML ingestion path. PyYAML is not a direct Lilo dependency; other installed libraries may depend on it.

See [validation results](deployment-validation.md) for CPU coverage and the earlier GPU smoke tests. The Python-config migration has not been redeployed.
