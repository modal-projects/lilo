# Python deployment configs

Each deployment is a Python file exporting a `Config` class that inherits from `BaseConfig`. The class contains model, trainer, inference, routing and lifecycle settings. App names, environments and worker-code releases are managed by the deployment command. There is no YAML loader or model catalog.

The layout follows the Python recipe approach used by [training-gym](https://github.com/modal-labs/training-gym) and the [multinode training guide](https://github.com/modal-labs/multinode-training-guide/blob/main/nemo-rl/configs/llama3_1_8b_math_2node.py). Only the outer BaseConfig is a dataclass; its sections are ordinary dictionaries. Config files need no decorators or default factories. Importing either project is not required.

## Define a deployment

Start with an example:

```bash
lilo config init --preset qwen35-9b-lora-16k > src/lilo/configs/my_model.py
```

The generated file imports a packaged config and subclasses it. Customize it using ordinary Python:

```python
from lilo.configs.qwen35_9b_lora_16k import Config as ParentConfig


class Config(ParentConfig):
    name = "my-9b-64k"
    overrides = {
        "model.max_context_length": 65536,
        "trainer.resources.gpu": "H200:8",
        "trainer.config.options.tensor_model_parallel_size": 8,
        "trainer.config.options.max_tokens_per_gpu": 65536,
        "inference.scaling.max_replicas": 6,
    }
```

New deployments can inherit directly from `BaseConfig` and declare ordinary class defaults such as `model = {"id": "...", "max_context_length": 16384}` and `trainer = {"resources": {"gpu": "H100:4"}, "config": {...}}`. The [16K example](../src/lilo/configs/qwen35_9b_lora_16k.py) shows the complete structure. No `@dataclass`, `field(default_factory=...)`, or `__post_init__` is needed in config files.

`BaseConfig` copies defaults for each instance, then applies each parent's overrides before its child's. Dotted paths select a section and traverse its dictionary keys. A value replaces the selected field/key, including whole lists and dictionaries; other keys remain unchanged. Native backend dictionaries can receive new option names. Unknown top-level sections and missing intermediate paths raise an error naming the override. Section keys are open; their consumers read the options they need. Constructor keywords, when supplied, replace top-level fields last.

Omitted orchestration settings come from one defaults dictionary in `deployments.py`. Backend options have no schema there. Replacing an entire section fills its omitted orchestration defaults; use dotted overrides to retain the parent’s other settings.

Copying happens inside the base class, so edits to a config's nested lists or dictionaries do not change its parent, another instance, or the class's override dictionary. Only the resulting settings enter the saved deployment record; workers do not apply inheritance again.

Python imports provide reuse. There is no YAML loader or `extends` key. Config files execute as Python when loaded; keep provisioning and training calls outside them. Sibling imports are available while loading a file.

All 14 previous presets are available under [`src/lilo/configs/`](../src/lilo/configs), with the same model, resource and backend settings. The [64K config](../src/lilo/configs/qwen35_9b_lora_64k.py) inherits from the 16K config. There is no separate `deployments/` wrapper directory.

No revision is required in a config. The CLI resolves the model's Hugging Face `main` branch automatically and saves the exact commit in the deployment record. An explicit `model.revision` remains optional for users who need particular weights; none of the built-in examples specify one. The example defaults therefore no longer pin the weights used in the historical GPU checks.

## Deploy the complete active set

Use Python 3.12, matching the serialized trainer and inference images:

```bash
lilo config validate src/lilo/configs/my_model.py
lilo config resolve src/lilo/configs/my_model.py --output /tmp/deployment.json
lilo deploy src/lilo/configs/model_a.py src/lilo/configs/model_b.py
```

`validate` loads the Python classes and checks routing and shared lifecycle settings. It does not start backend libraries or prove that the model fits in GPU memory. `resolve` additionally pins model revisions and emits the deployment records as JSON; it does not provision compute.

For a checked-in list, edit [`scripts/deploy_models.sh`](../scripts/deploy_models.sh). Add a Python config file and its path to the `deployment_files` array, then run:

```bash
./scripts/deploy_models.sh
```

Supply every configuration that should remain available to new clients. Omitted configurations are retained for existing jobs but removed from new-client selection. All files in the list must agree on shared lifecycle settings. Frontend deployment settings are not part of `BaseConfig`. The command selects the app, environment and region:

```bash
./scripts/deploy_models.sh --app my-lilo --env dev --region us-west
```

These flags are optional; existing provider defaults apply when omitted. Secret and volume names come from provider defaults and are saved as platform metadata in deployment records. Credentials remain in Modal secrets. Pin `LILO_MILES_COMMIT` when a specific Miles build is needed.

## Code path

```text
lilo deploy config.py
  deployment_cli.main()
    compile_configs()
      deployments.load() → execute config.py → Config()
      resolve model tag to a Hugging Face commit
      DeploymentRecord.create() → copy settings and compute config hash
    deploy()
      retain existing job configurations
      deploy new trainer/inference apps; skip existing apps
      save/pass manifest JSON
      modal deploy -m lilo.providers.modal.app
```

[`load()`](../src/lilo/deployments.py) executes the file and instantiates its exported `Config` subclass. The returned dataclass goes directly to the orchestration code. There is no dictionary-to-config conversion on this path.

`DeploymentRecord` adds the pinned Miles revision, configuration hash and active status. JSON is used only to store records and pass them to other processes. Pydantic reconstructs BaseConfig with its dictionary sections when reading those records; it does not import or run the user's config file in a GPU worker. Records preserve the full computed settings rather than a reference to the original Python file.

| File | Responsibility |
| --- | --- |
| [`deployments.py`](../src/lilo/deployments.py) | BaseConfig defaults and overrides, Python file loader, saved record and shared-frontend checks |
| [`deployment_cli.py`](../src/lilo/deployment_cli.py) | Revision lookup, manifest updates and Modal deployment |
| [`deployment_apps.py`](../src/lilo/providers/modal/deployment_apps.py) | Independent trainer apps, inference provisioners, server classes and process startup |
| [`app.py`](../src/lilo/providers/modal/app.py) | Shared frontend referencing deployed trainer functions |
| [`deployment_worker_app.py`](../src/lilo/providers/modal/deployment_worker_app.py) | Entrypoint for deploying one trainer or inference provisioner |
| [`deployment_pool_app.py`](../src/lilo/providers/modal/deployment_pool_app.py) | Construct an inference pool from its saved record |
| [`backends/deployment.py`](../src/lilo/backends/deployment.py) | Select the backend configuration reader |

`build_trainer_app(record)` configures resources, secrets, storage and limits. The CLI deploys it as its own app. The frontend looks up its `trainer` function by app name. When that function starts, `run_trainer()` obtains backend settings and passes them as `LILO_BACKEND_CONFIG` to the existing Miles or Megatron executor.

Inference pools are separate apps created on demand by a deployed `provision` function. That function keeps the source used when its inference app was deployed, including when an idle pool needs to be recreated after a frontend upgrade. `build_rollout_app()` constructs a server from the saved configuration. Startup launches SGLang with native options, waits for its health endpoint, then starts the LoRA or FFT sidecar.

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

## Hashes and independent updates

There is no source fingerprint and no check that a config matches the current Lilo checkout. Hashes identify settings; they do not certify model support or successful training.

| Identifier | Inputs | Used for |
| --- | --- | --- |
| `generation` | Computed config except routing defaults, platform settings, saved worker releases and Miles commit | Saved job/checkpoint configuration and routing |
| `trainer_hash` | Deployment name, model, trainer section, platform settings, trainer release and Miles commit | Independently deployed trainer app name |
| `inference_hash` | Deployment name, model, inference section, platform settings, inference release, adapter rank and target modules | Independently deployed inference provisioner app name |
| Asset hash | Model repository and resolved model commit | Download directory |

The hashes use SHA-256 over sorted JSON. Trainer/inference app names use the first 24 hex characters. Definition IDs use the first 16 characters of `generation`. Code is retained by the deployed Modal apps, rather than reconstructed from a source fingerprint.

Worker-code versions are not config fields. The CLI keeps the previous release IDs in deployment records. To deploy changed code for selected workers:

```bash
./scripts/deploy_models.sh --refresh-trainer qwen35-9b-lora-16k
./scripts/deploy_models.sh --refresh-inference qwen35-9b-lora-16k
```

The CLI generates a release ID for each requested update; users do not specify or maintain it. Subsequent ordinary deploys retain that release. Refresh flags can be repeated for multiple config names. Editing source alone does not update existing workers. The frontend itself is redeployed on every apply.

Examples:

- Change inference concurrency: deploy a new inference provisioner; keep the existing trainer app.
- Change trainer batch settings or request a trainer code refresh: deploy a new trainer app; keep the inference provisioner.
- Change adapter rank or target modules: update both because inference must load the changed adapters.
- Change routing defaults: keep both worker apps.
- Change one Miles configuration: other Miles configurations and Megatron apps remain deployed as they were.

The CLI records each successfully deployed worker app before updating the frontend. Retries skip completed apps and recover a worker deployment that succeeded before its registry write. Retained configurations reference old apps; the CLI does not rebuild those apps with new source. Old apps are retained, with GPU trainers scaling to zero when idle. Automatic deletion of unused worker app definitions is not implemented.

Trainer capacity is enforced per saved definition by control-plane admission and reconciliation. The reusable Modal trainer function has no additional global container cap, so retained jobs do not prevent new definitions from starting. Old and new definitions can consume their configured capacity simultaneously during an update.

Pools remain scoped to the full deployment definition (and FFT job/version). A trainer update can therefore cause a new job to obtain a separate pool even when it uses the same inference provisioner. Existing pools are not redeployed.

Backend startup checks and optional smoke tests remain separate from these identifiers. Worker/frontend protocol changes still require compatible APIs or an explicit migration; removing the source fingerprint does not guarantee arbitrary old and new versions interoperate.

This changes the draft's deployment-record format and replaces its earlier shared-app trainers. Existing deployments from that draft need a fresh frontend/registry or an explicit migration; this change does not silently convert running shared-app trainers.

Existing resource names (`lilo-yaml`, `*-yaml-deployments`, and the `yaml_` definition prefix) are retained for naming continuity. They no longer indicate a YAML ingestion path. PyYAML is not a direct Lilo dependency; other installed libraries may depend on it.

See [validation results](deployment-validation.md) for CPU coverage and the earlier GPU smoke tests. The Python-config migration has not been redeployed.
