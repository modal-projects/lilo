# Proposed YAML deployments for Lilo

Status: design only. None of the YAML fields, CLI commands, URL layouts, or new Python APIs below are implemented by this document. It does not change existing deployments. Inspected Lilo main at `67f21ee` and upstream Miles at `12754e9507e64d5e537288da17793246e913c525` on September 21, 2026.

## Decision

Use one declarative specification per deployment. It identifies a real base model, training mode, context limit, trainer resources and backend options, and inference resources and backend options. The same specification builds shared and scoped deployments. Lilo ships editable presets for tested model/context combinations. A user-provided specification is sufficient to add a model; no model-specific Lilo Python module, central import, or catalog edit is required.

A running deployment necessarily has a concrete configuration for a concrete model. It does not follow that Lilo must ship a configuration for every possible model. Templates reduce duplication, and the user selects which deployments to run.

The initial version provisions only explicitly applied specifications. A plain Tinker create request does not choose hardware, build an image, or provision an arbitrary unconfigured model. Automatic provisioning from a default hardware profile can be added later as a separate policy.

## What Miles Tinker does

Upstream `serve_tinker.py` parses startup arguments, checks that trainer and inference use the same frozen HF base, initializes the inference controller and trainer, and builds a gateway whose public model name is `args.tinker_base_model or args.hf_checkpoint`. Its service checks incoming model names against that single configured value. Its capabilities endpoint advertises that one name. Each training client gets an adapter on those base weights; it does not select a different architecture.

Sources at the inspected commit:

- [Startup and gateway model name](https://github.com/radixark/miles/blob/12754e9507e64d5e537288da17793246e913c525/serve_tinker.py#L59)
- [Training model-name check](https://github.com/radixark/miles/blob/12754e9507e64d5e537288da17793246e913c525/miles/tinker/core/service.py#L133)
- [Capabilities](https://github.com/radixark/miles/blob/12754e9507e64d5e537288da17793246e913c525/miles/tinker/server/app.py#L92)

Lilo currently invokes Miles directly as a backend, rather than forwarding requests to Miles' standalone Tinker server. Keep this arrangement: Lilo owns sessions, futures, checkpoints and provisioning; Miles owns trainer construction and operations. These upstream observations do not imply that Lilo's pinned runtime implements every feature on Miles main.

## One complete deployment file

Illustrative configuration based on the existing Qwen3.5-9B-Base 16K deployment. Native option spelling below uses argparse destination names, generally underscores. Exact accepted options are validated against the selected runtime image.

```yaml
api_version: lilo/v1
name: qwen35-9b-lora-16k

model:
  id: Qwen/Qwen3.5-9B-Base
  revision: main                     # Resolved to a commit before application.
  parameterization: lora
  max_context_length: 16384

deployment:
  mode: shared                       # Or scoped, tied to lilo.run lifetime.
  modal:
    environment: dev
    region: us-west
  secrets:
    api: lilo-api
    sampler_proxy: lilo-proxy
    huggingface: huggingface          # Optional; secret reference, never a token.
  storage:
    assets: lilo-model-assets
    checkpoints: lilo-checkpoints
    bulletin: lilo-snapshot-bulletin

trainer:
  backend: miles
  image:
    preset: miles                    # Resolved to a concrete build/runtime revision.
  resources:
    gpu: H100:4
    cpu: 16
    memory_mib: 65536
    timeout_s: 86400
  scaling:
    min_instances: 0
    max_instances: 1
  engine:
    max_clients_per_instance: 6
    sampler_persistence_concurrency: 8
  miles:
    model_args: qwen3.5-9B            # Backend architecture preset; not a Lilo model entry.
    options:
      tensor_model_parallel_size: 4
      context_parallel_size: 1
      expert_model_parallel_size: 1
      expert_tensor_parallel_size: 1
      multi_lora_n_adapters: 6
      lora_rank: 32                   # Allocated maximum; clients may request supported lower ranks.
      lora_alpha: 32
      lora_dropout: 0.0
      target_modules:
        - linear_qkv
        - linear_proj
        - linear_fc1
        - linear_fc2
        - output_layer
      max_tokens_per_gpu: 16384
      recompute_granularity: full
      recompute_method: uniform
      recompute_num_layers: 1
  env:
    PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True

inference:
  backend: sglang
  image:
    preset: sglang
  resources:
    gpu: H200:1
    cpu: 8
    memory_mib: 32768
  scaling:
    min_replicas: 0
    max_replicas: 8
    target_concurrency: 16
    scaledown_window_s: 300
  sglang:
    options:
      tp_size: 1
      mem_fraction_static: 0.8
      max_running_requests: 32
      max_queued_requests: 8
      max_loaded_loras: 64
      max_loras_per_batch: 8
      schedule_policy: lpm

lifecycle:
  session_idle_timeout_s: 300
  pool_idle_timeout_s: 300
  sweep_interval_s: 300
```

The numbers describe a deployment choice, not a guarantee of optimal performance. In particular, zero inference minimum is an intentional departure from presets that keep workers warm. `lora_rank`, trainer slots, simultaneous adapters in an inference batch, retained adapter versions, and HTTP concurrency are different capacities.

The deployment spec owns alpha and allocated rank. A Tinker client supplies its supported rank, seed and trainable-module selection. The resolved module selection must be compatible with the allocation and the exported adapter schema. Optimizer parameters and training batches remain per-client Tinker operations, rather than deployment-wide training-loop configuration.

## Presets, overrides and resolution

Proposed commands:

```bash
lilo config init --preset qwen35-9b-lora-16k > deployment.yaml
lilo config validate deployment.yaml
lilo config resolve deployment.yaml --output deployment.resolved.yaml
lilo deploy deployment.yaml
lilo deployment check qwen35-9b-lora-16k --gpu
```

`config init` writes the full editable YAML. This is the simplest supported workflow. As a convenience, allow a single `extends: builtin:qwen35-9b-lora-16k` or `extends: ./base.yaml`. Local relative references resolve against the containing file. Reject cycles and duplicate YAML keys. The resolver records hashes of referenced content so later preset edits cannot silently alter an existing deployment.

Resolution order: schema defaults, parent preset, current YAML, explicitly provided CLI overrides. Maps merge recursively; lists replace; null clears only optional fields and is otherwise rejected. Changing a field to false must actually disable it, including when enabled by a preset. Show the final resolved values and their origins; never execute shell interpolation in YAML.

Changing `model.id` does not imply that a copied architecture preset remains valid. Require successful backend validation; incompatible explicit architecture dimensions must be rejected. Initial architecture sources are a Miles model-argument preset or explicitly supplied architecture options. Automatic HF-to-backend inference is used only where the selected backend provides a reliable implementation; do not build a second Lilo model-name table that guesses upstream recipe names.

A normalized `ResolvedDeploymentSpec` contains the pinned model revision, rendered backend arguments, concrete image/runtime revisions, resources, storage references and module mapping. Persist it before provisioning. Redact secrets from all rendered output; specifications contain secret names only.

## Routing from Tinker's base_model

The primary workflow uses the URL returned for that deployment:

```python
service = tinker.ServiceClient(base_url=deployment_url, api_key=api_key)
training = service.create_lora_training_client(
    base_model="Qwen/Qwen3.5-9B-Base", rank=32,
)
```

The URL selects `qwen35-9b-lora-16k`; `base_model` confirms the real model. Another URL can expose the same model with 64K context. This needs no SDK change and no synthetic Hugging Face model IDs. These URLs can be paths on one gateway, for example `/deployments/qwen35-9b-lora-16k`, mounted so SDK `/api/v1/...` requests work below that prefix; they need not each be a separate CPU service. Routing remains authenticated.

For an optional shared root URL, register deployments automatically on apply. Generate the index `(canonical_model_id, parameterization) -> deployed configurations`. One candidate routes directly. Multiple candidates require an explicit default for that model/mode; applying a second default is an error. Without a default, fail with the available deployment URLs rather than choosing the shortest context, newest deployment, or lowest GPU price.

A shared gateway can configure defaults using deployment names. This is the only manual routing choice required for ambiguous variants. It is not a model allowlist and never requires a Python edit. The canonical model ID is read from each deployment spec, not repeated in that defaults file.

Initially, prefer deployment URLs over names such as `Qwen/model@64k` in `base_model`. Tinker and surrounding libraries can use model names for tokenizers and metadata. If aliases are later added, resolve them at the gateway and consistently return the canonical model in model/tokenizer metadata.

At a deployment URL, capabilities reports its model and context length. The shared root advertises only unambiguous/default routes; a separate Lilo deployment-list endpoint reports every variant, its status and URL. Sampling-only requests follow the same selection rules. Training-derived sampling clients and checkpoint restores inherit the selected deployment generation; they must not be rerouted through the current default mid-run.

## Backend option passthrough

Use a strict schema for Lilo-owned settings. Backend-native options are open maps, validated by their selected backend version, rather than a hand-maintained exhaustive list in Lilo.

| Source | Destination / rule |
| --- | --- |
| `model.id`, resolved revision | Prepare one exact HF snapshot; trainer and sampler get its path. |
| `model.max_context_length` | Advertised context and supported trainer/sampler sequence limits. Packing/token budgets remain separate. |
| `trainer.resources` | Modal trainer function resources; derive actual world size from provisioned GPUs. |
| `trainer.engine` | Lilo admission, scheduling and persistence settings. |
| `trainer.miles.model_args` | Load the selected backend architecture preset in its runtime image. |
| `trainer.miles.options` | Native Miles/Megatron options, after architecture defaults. |
| `inference.resources/scaling` | Modal serving resources and autoscaling. |
| `inference.sglang.options` | Native SGLang server options. |
| Backend-exported adapter schema | Validate/derive serving target names and maximum rank. |
| `env` | Role-local environment; managed `LILO_*` variables cannot be overridden. |

The adapter first obtains the actual parser/schema in the selected image, normalizes argument aliases, merges architecture defaults with user options, and validates/serializes the final configuration. Handle booleans, negative flags, multi-value/repeated options and comma-separated values according to that parser. Do not implement a naive `--key str(value)` loop. Unsupported encoding or unknown native options fail with the YAML path and backend error. Build argv lists; do not evaluate shell strings. Backends whose schema requires a GPU receive structural checks in preflight and full checks at initialization.

Some settings are owned by Lilo because they determine integration behavior: model/checkpoint paths, launch world size, communication addresses, managed storage paths, trainer-only mode, and dispatch/publication hooks. Reject attempts to redefine these through passthrough, even under a CLI alias. For example, reject a Miles rollout allocation because Lilo provisions the serving pool itself. Show generated managed arguments in resolved output so this is inspectable.

Context and rank limits are configured once. Reject conflicting native values rather than silently overriding them. Preserve hard integration constraints independently of upstream argument acceptance: an option accepted by Miles is not evidence that Lilo implements its scheduling, export, or distributed layout. Enabling PP or changing transfer mode must not bypass these checks.

The current `_PEFT_TARGETS` mapping is not a universal model compatibility solution. Prefer backend-resolved export names and validate SGLang support. Allow explicit mapping/provider configuration when necessary, with a startup check. A custom provider must already be installed in the selected image; accepting YAML does not install arbitrary new runtime dependencies.

## Provisioning and validation

`lilo deploy` compiles the YAML into generic trainer and sampler definitions, deploys/registers them, and returns the URL and generation ID. Model-specific Python source is not generated or imported. Modal resource declarations are built during application construction, when GPU and image choices are known. Passing a new `gpu` value to an already-deployed function invocation cannot change its allocation.

Use dedicated generated function/app identities per deployment generation. The current single-use trainer container behavior stays intact. A serving container is permanently associated with one resolved base/model configuration; a warm container cannot pick up another model because a registry pointer changed. A finite set of generic resource pools could be an optimization later, but is unnecessary for this design.

With minimum capacity zero, applying a specification registers resources without warming model GPUs. The first client starts capacity; an explicit warm/check command performs initialization earlier. Nonzero configured minima intentionally reserve capacity.

Validation has three stages:

1. Local schema checks: types, required values, context/rank relationships, supported Lilo integration features, duplicate names/defaults, resources and topology.
2. Preflight in selected runtime images: resolve model config/revision and tokenizer assets, parse native options, resolve architecture/provider and adapter export mapping. No successful preflight should claim to prove GPU memory fit.
3. First startup (or explicit GPU check): load trainer and sampler, validate an adapter through export/load/generation, and test a small forward/backward operation on a disposable slot where appropriate. Discard probe state so no user optimizer or checkpoint is changed. Reuse the initialized resources for real work. Validate the requested allocation, but do not imply that a tiny probe proves every advertised maximum batch fits; boundary-capacity checks are a separate explicit test.

A lightweight base-generation probe alone is insufficient: verify that the exported adapter contains expected tensors and the serving request actually selects that adapter. Cache successful verification against exact model/runtime/configuration revisions, not a moving model name. Every new container still performs normal load/readiness checks.

Creation remains asynchronous through Tinker's existing future. Internally expose `resolving`, `provisioning`, `initializing`, `ready`, and `failed`, with timestamps and a structured failure cause. At model creation, require readiness of the trainer and the serving compatibility check for the training+sampling deployment. A valid checkpoint or active trainer must not later be destroyed solely because an individual sampling request fails.

Startup errors must reach the control plane even if the backend dies before `accept_model` becomes available. Persist operation/attempt IDs and progress before launch, watch deployment/function completion, and complete the creation future with the original diagnostic. Retry capacity/network failures with bounds; do not repeatedly provision a permanently invalid configuration. Deduplicate simultaneous creates for the same generation. Use recoverable provisioning leases with ownership checks, not permanent claim markers.

On failure/cancellation, release resources created solely by the failed request when no other client needs them. Preserve shared resources and existing client jobs. Cleanup is idempotent and reconciles after process death. It must include pending provisioning, not only already placed models.

## Identities, upgrades and storage

Keep four distinct identifiers:

- Deployment name: user-facing, stable (`qwen35-9b-lora-16k`).
- Generation ID: hash of the normalized model/runtime/execution configuration; used for containers and compatibility verification.
- Model ID: individual Tinker client's adapter/training state.
- Publication version: a particular set of that client's adapter weights.

Compute compatibility fingerprints from exact base revision, tokenizer/architecture settings, parameterization, export schema, parallelism and runtime revisions. Store a separate deployment-policy revision for scaling limits and timeouts so changing replica count does not invalidate model weights. Image/environment settings affecting numerical behavior belong to the execution fingerprint, not only the scaling policy.

Assets are keyed by full repository ID and revision, not only basename. Checkpoints record canonical model, exact revision, adapter schema and topology alongside the existing metadata. Reopening a checkpoint uses recorded information and explicit compatibility checks, not the currently selected model default. Changing replica count should not make a checkpoint unloadable. Older metadata remains readable through existing compatibility rules; unknown old revisions are not silently treated as a new revision.

Applying changed execution settings creates a new generation. New sessions use it only after deployment/preflight succeeds; old sessions remain pinned and drain normally. Startup failure on a lazily warmed generation is reported without rewriting existing sessions. Explicit warm-before-switch can provide stronger rollout guarantees. Retain old configuration records while referenced by clients, pools, futures or checkpoints. Removal disables new admission; stopping active jobs requires an explicit operation.

The current `_lose_undefined_models` behavior must be replaced with checks against durable deployment records. A removed Python module or a restarted control plane must not invalidate a YAML deployment. The same recorded spec drives trainer reconciliation, LoRA pool cleanup, FFT latest/pinned pools and scoped teardown.

## Shared/scoped deployment parity

The shared CLI and a proposed `lilo.run(config="deployment.yaml")` load the same schema, resolver and builders. Lifecycle ownership differs: shared applications outlive the invoking command; scoped applications follow their owner. Preserve existing checkpoint-volume, proxy-auth and pinned-pool behavior in both modes. Secret references and deployment management belong to the operator; ordinary Tinker API keys do not grant configuration-management access.

Multi-node, alternate trainer backends, multimodal training and different weight-transfer mechanisms are extensions behind backend adapters. They are not implicitly enabled because a YAML option exists. V1 preserves the execution layouts Lilo can actually support and rejects unsupported combinations explicitly.

## Implementation plan and acceptance criteria

1. Introduce schema/resolver and a normalized specification. Translate existing definitions into presets with identical values, including warm minima, adapter targets and historical GPU layouts. Snapshot-test resolved settings.
2. Introduce a deployment registry and generic builders. Keep old definition IDs as compatibility aliases while migrating references. Remove runtime Python-module imports and source-file hashing from pool resolution.
3. Add CLI validate/resolve/deploy and deployment-specific URL routing. Implement explicit shared-gateway defaults and test ambiguity. Keep the ordinary Tinker call unchanged.
4. Add native backend parser adapters, managed-option collision checks, image-based preflight, and durable startup failure reporting. Replace the deleted-module cleanup assumption before allowing dynamic entries.
5. Add startup compatibility checks, generation-aware update/drain behavior, and shared/scoped parity. Remove the old hand-maintained imports/catalog after migration coverage passes.

Required tests: preset parity; merge/list/false override semantics; unknown options and protected aliases; backend parser errors; two contexts for one model; base versus instruct model IDs; sampling-only requests; tokenizer metadata; checkpoint restore after default changes; simultaneous cold creates; interrupted provisioning; failed trainer/sampler initialization; old-client draining; idle teardown and scoped owner loss. Run real GPU smoke tests on an existing dense LoRA preset and at least one backend-supported model absent from Lilo's old catalog, plus FFT and a supported MoE configuration before claiming those migrations complete.

Success means a user can copy a YAML, set a new supported model and appropriate backend/hardware options, deploy it, and use an unchanged Tinker script. No edits to Lilo's Python catalog are required. Failures identify the exact unsupported setting or runtime operation instead of reporting merely that the model name is missing.
