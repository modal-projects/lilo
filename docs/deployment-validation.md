# Deployment configuration validation

These checks exercise PR #55. The current implementation deploys trainers and inference provisioners independently; the shared frontend references them by app name. Earlier sections record validation of the previous shared-app implementation.

## Current: typed config composition and one backend resolution

Configs export a `Deployment` built from validated dataclasses. Ordinary `dataclasses.replace` composes variants; custom inheritance, dotted overrides, and recursive defaults have been removed. Lilo-owned fields reject unknown keys. Backend tuning remains in explicit dictionaries, with duplicates of Lilo-managed settings rejected before constructor calls.

Deployment records save complete trainer and inference settings. Workers consume those settings without resolving them again. SGLang receives `ServerArgs(**settings)` directly; the argparse adapter is now Miles-only. Megatron provider, optimizer, and distributed constructors receive one merged settings dictionary, without attribute-patching loops.

Trainer timeout, CPU, memory, inference startup timeout, and replica scaling are wired through. Unsupported trainer minimum instances and generic inference resource timeout fields are rejected. The merged multi-node Miles launcher is integrated with `Compute.nodes`; a two-node Qwen3.8-27B 256K config replaces the legacy catalog definition.

Validation: **651 CPU tests passed, 1 skipped**. Regression coverage includes misspelled orchestration fields, unsupported settings, optimizer/provider/distributed override collisions, all 15 example configs, saved-record round trips, direct SGLang construction, compute-setting propagation, independent worker updates, and multi-node launcher wiring. No apps were redeployed. GPU backend startup and the new multi-node config have not been live-tested in this revision.

The follow-up reduction removes three adapter modules and their dispatch wrappers, the unused pool-environment helper, and duplicated preset definitions. The two composed presets were compared field-for-field with their previous values. The CPU suite remains at **651 passed, 1 skipped**.

## Historical validation

The sections below describe earlier revisions, including APIs that have since been removed. Their live results do not validate the current implementation.

## Direct backend configuration

Removed the deployment-specific Miles field-renaming table and Megatron `runtime/provider/optimizer/distributed` schema. Configs now use the existing `MilesBackendConfig` and `EngineModelConfig` field names. Megatron's existing config reader constructs its nested optimizer; extra Megatron constructor settings are explicit `*_overrides` dictionaries instead of being split by field name.

Replaced `native_options.py` with `argparse_config.py`: configured values become arguments and the actual backend parser validates types/choices. Explicit false boolean flags use defaults after removing the preset flag. Renamed the Miles passthrough dictionary to `cli_options`. Replaced `backend_options.py` with a small `reject_managed_options` check.

Validation: **619 CPU tests passed, 1 skipped**. All 14 examples produce the same effective trainer and inference settings as before (with the passthrough field renamed). CPU coverage includes Megatron constructor forwarding, Miles/SGLang overrides, boolean false, list/alias overrides, custom type converters and backend rejection of invalid types/choices. Ruff and whitespace checks passed. No apps were redeployed.

## Model configs contain infrastructure settings only

Removed the `deployment` section from `BaseConfig` and removed explicit model commits from the built-in examples. Frontend app/environment/region selection belongs to `lilo deploy`; platform metadata and automatically generated worker-release IDs live in saved records. Config authors do not need revision or runtime-version fields. The CLI resolves omitted model revisions from Hugging Face `main`.

Code-only updates use `--refresh-trainer CONFIG_NAME` or `--refresh-inference CONFIG_NAME`. Subsequent ordinary deploys retain those releases. The deployment script forwards these options.

Validation: **619 CPU tests passed, 1 skipped**. Added tests cover all 14 examples having no deployment/revision/runtime-version fields, automatic model-commit lookup, command-owned platform settings, and preservation of refreshed workers without changing configs. Existing update-isolation and worker-app construction tests still pass. Ruff, whitespace and script syntax checks passed. No apps were redeployed.

## Independent worker apps and config hashes

Removed the global source fingerprint and its code-upgrade rejection. The config hash identifies saved job settings; separate trainer and inference hashes identify worker apps. Runtime upgrades use explicit per-role `runtime_version` labels. The CLI skips existing worker apps, deploys changed ones before updating the frontend, and records successful workers for retry. Inference provisioners retain their source in an image so idle pools can restart using their original code.

CPU tests cover inference-only changes, Miles-only updates beside Megatron, runtime-version changes, adapter-shape changes, retained configurations, interrupted deploy recovery, frontend trainer references, remote spawn arguments, saved inference provisioners, and construction of both real Modal worker entrypoints without deploying. Trainer capacity remains enforced per saved definition by the existing control plane; there is no function-wide cap blocking new definitions behind retained jobs.

Validation: **615 CPU tests passed, 1 skipped**. Ruff and whitespace checks passed. Deployment-command isolation is tested with mocked Modal calls; no apps were redeployed and this architecture has not yet had a live GPU/deployment test. Existing manifests from the earlier shared-app draft require migration or a fresh frontend/registry. Worker API compatibility across future releases and cleanup of unused worker apps remain explicit operational concerns.

## Plain dictionary sections

Config files import only `BaseConfig`. Model, trainer, inference, resource, scaling, routing and deployment sections are plain dictionaries, consumed directly by backend and Modal code. The nested template classes have been removed. One defaults dictionary supplies omitted orchestration settings; inherited dotted overrides still work.

Validation: **604 CPU tests passed, 1 skipped**. All 14 computed configs match their previous settings, build backend/inference options and round-trip through saved JSON. Tests cover independent mutable values, omitted defaults, inherited overrides and native backend option forwarding. Ruff and whitespace checks passed. No apps were redeployed.

## Simple class defaults and inherited overrides

Config files now declare ordinary class defaults and dotted `overrides` dictionaries. They contain no dataclass decorators, default factories or `__post_init__` methods. `BaseConfig` copies values per instance and applies parent defaults/overrides before child defaults/overrides. The resulting settings and saved-record format are unchanged for all 14 configs.

Validation: **603 CPU tests passed, 1 skipped**. Added tests cover inherited overrides, child field replacement, false values, list/dictionary replacement, independent mutable values and override typo errors. All 14 computed configs were compared with the previous version and round-tripped through saved JSON. The CLI generated and validated the simplified config. Ruff and whitespace checks passed. No apps were redeployed.

## One config directory

Removed the three `deployments/` wrappers. The deployment script now lists `src/lilo/configs/` directly, and the validated model revisions live in the 9B LoRA and 4B FFT configs. Derived configs inherit those revisions. Authoring configs are excluded from runtime source fingerprints and shared-app/trainer/inference source mounts; workers receive computed settings as JSON.

Validation: **599 CPU tests passed, 1 skipped**; Ruff, whitespace and deployment-script syntax checks passed. The CLI loaded the three selected configs together. Regression tests verify that editing config source leaves the runtime fingerprint unchanged, editing backend source changes it, and worker source filtering retains runtime code while excluding authoring files. No apps were redeployed.

## Python dataclass configuration

Python `Config` subclasses replace the YAML presets and loader. All 14 configs were compared with the previous specifications: model, resources, backend options and inference options are unchanged. The three checked-in deployment files retain their pinned model revisions. Inheritance uses normal dataclass defaults and `__post_init__`; tests verify mutable defaults are independent.

Validation: **597 CPU tests passed, 1 skipped**; Ruff passed for changed Python files and whitespace/shell-syntax checks passed. The CLI generated a Python config and validated it. A clean wheel contains all 14 Python configs and no YAML presets or old YAML app modules; every config was loaded from that wheel, its backend settings built, and its record round-tripped through JSON. PyYAML was removed from Lilo's direct dependencies; the lockfile otherwise preserves package versions and sources.

This migration has not been redeployed or GPU-tested. The GPU results below describe the earlier YAML-based source at `13d2a31`, before the subsequent CPU-only refactors. Earlier cleanup counts below are historical.


## Deployment record simplification

The saved manifest wrapper is now named `DeploymentRecord`. Its factory copies the parsed specification and computes the configuration hash without a dictionary-to-model round trip. Model tag resolution and validation of the Hugging Face result stay in the CLI. The standalone `resolve()` function was removed. Serialized record fields and hash format are unchanged.

Validation: **595 CPU tests passed, 1 skipped**; Ruff passed for changed Python files and whitespace checks passed. Added coverage verifies model tag pinning, no lookup for an existing commit, record isolation from subsequent mutations, JSON round trips, and unchanged configuration hashes. No apps were redeployed.

## YAML loader simplification

`DeploymentSpec.compatible()` and `_load()` were removed. `load()` reads the inheritance chain, merges parent to child, and constructs the deployment fields once. Loading, deserializing and resolving a specification do not call backend readers. Backend integration checks run when building backend settings; the Modal provider checks reserved environment variables when configuring apps.

Validation: **592 CPU tests passed, 1 skipped**; Ruff passed for changed Python files and whitespace checks passed. Coverage includes partial parents, inheritance cycles, intermediate replacements, and ensuring loading/resolution never calls backend parsers. No apps were redeployed.

## Backend configuration passthrough

The deployment schema now uses `trainer.backend` / `trainer.config` and `inference.backend` / `inference.config`. Backend readers interpret those mappings outside the Modal provider. Megatron exposes native provider, optimizer and distributed-training settings alongside its Lilo runtime settings.

Validation: **588 CPU tests passed, 1 skipped**. Ruff passed for changed Python files, and whitespace checks passed. A comparison against the previous commit confirmed that all 14 migrated presets produce the same resolved backend and inference settings, apart from the new empty native-override fields. Tests cover native values reaching Megatron constructors, preserving nested values and false booleans through serialization, rejecting conflicting integration settings, and rejecting exact checkpoint resume when native optimizer/distributed settings differ.

Megatron constructor tests use CPU stubs; they verify forwarding and error propagation, not acceptance by the GPU image's installed Megatron version. The native backend remains responsible for validating its options at worker startup. This refactor has not been deployed or GPU-tested.

## YAML-only deployment cleanup

After the live checks below, the shared deployment's Python-catalog fallback and model-specific trainer/pool modules were removed. Existing recipes are available as YAML presets. The deployment script still selects only its three listed configurations; additional presets are opt-in.

Validation of this cleanup: **575 CPU tests passed, 1 skipped**; Ruff and whitespace checks passed. A wheel built with all 14 YAML presets and none of the removed Python definition modules or pool launchers. Coverage includes required manifests, configuration-ID sampling, both generic executors, custom checkpoint storage, generic pool deployment, migrated parallelism settings, and the deployed-configuration E2E helper.

No apps were redeployed for this cleanup. The GPU results below apply to the earlier source commit `13d2a31`; the additional migrated presets and DP-attention settings have CPU coverage only.

## GPU results

All checks passed on 2026-09-23. A step includes forward/backward, an optimizer update, publication of sampling weights, and a sample with a 16-token cap.

| Configuration | Region | Completed steps | Result |
| --- | --- | ---: | --- |
| 9B LoRA, 16K preset | us-west | 3 before deployment updates + 1 after | Passed; original trainer and inference container survived |
| 9B LoRA, changed 16K YAML | us-west | 3 on a fresh client | Passed; client selected the new configuration |
| 4B FFT, 64K preset | us-west | 4 across deployment updates | Passed; original trainer survived |
| 9B LoRA, 64K preset, 8×H200 | us-east | 3 | Passed; functional check in a separate frontend |

Every step returned 512 finite training log probabilities and 16 generated tokens with finite log probabilities. The test sends 512 training tokens and a 16-token sampling prompt; it does **not** fill a 16K or 64K context. It tests the configured processes and interfaces, not maximum-context capacity or learning quality.

All completed smoke-test clients unloaded without reported cleanup errors. The dedicated test frontends and their inference pools were stopped after validation.

## Deployment isolation checked on 2026-09-23

Source commit: `13d2a31`. CPU suite: **554 passed, 1 skipped**. The isolated shared app was `lilo-yaml-pr55-smoke-v3` in `modal-labs/kailash-dev`, initially deployed with all three packaged presets and pinned model revisions.

Modal reported 7.314 seconds for an unchanged deployment. A second apply changed only `qwen35-9b-lora-16k`'s `inference.sglang.options.max_running_requests` from 32 to 24; Modal reported 4.138 seconds for that deployment. The shared frontend was redeployed on both applies.

Both applies preserved the running 16K LoRA and FFT trainer boot IDs. Existing inference app IDs were unchanged, and the running LoRA inference container survived both applies. The original LoRA client completed three initial steps and another complete training/publication/sampling step after both applies, on the same trainer boot ID.

The changed YAML received a new definition; the previous definition was retained as inactive for existing clients. The unchanged FFT and 64K definitions retained their function IDs. The 64K trainer was still queued during these applies, so these observations do not establish continuity of an already-running 8×H200 trainer.

The FFT client also completed four training/publication/sampling steps on its original trainer boot ID. Its first sample was waiting for inference startup during the deployments, then completed successfully. A fresh 16K LoRA client selected the new configuration and passed three steps with the 24-request inference limit recorded in its resolved configuration.

These results establish continuity of these existing jobs. They do not mean `lilo deploy` skips deploying the shared app, or that an unrelated frontend request can never be retried during deployment.

## Runtime revisions and hardware

| Preset | Model revision | Trainer GPU request | Inference GPU request per replica |
| --- | --- | --- | --- |
| `qwen35-9b-lora-16k` | `68c46c4b3498877f3ef123c856ecfde50c39f404` | 4×H100, TP4 | H200 |
| `qwen35-9b-lora-64k` | `68c46c4b3498877f3ef123c856ecfde50c39f404` | 8×H200, TP8 | H200 |
| `qwen35-4b-fft-64k` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 4×H100, TP2/CP2 | H100 |

Miles was pinned to `f6d0257d83b7c14f4ce43ecfcd71d955112f6e0e`. GPU types in the table are requests: Modal can fulfill an H100 request with H200 hardware, which occurred during this test. Inference retained the presets' zero minimum and eight maximum replicas; these small requests did not exercise scaling to eight replicas.

The western 8×H200 trainer remained queued without a container for approximately 20 minutes. Its session and queued call were cancelled before retrying the same hardware/model configuration in a separate `us-east` test frontend. That retry passed three steps and is a functional check of the 64K configuration, separate from the western shared-app isolation test.

## Reproduce the smoke test

Use Python 3.12 and an authenticated Modal environment. Set `TINKER_API_KEY` locally to the value in the deployment's API secret. Generate Python config files from the packaged examples, give them the same isolated `deployment.frontend`, and pin model revisions plus `LILO_MILES_COMMIT` before deploying.

```bash
lilo deploy src/lilo/configs/qwen35_9b_lora_16k.py src/lilo/configs/qwen35_9b_lora_64k.py src/lilo/configs/qwen35_4b_fft_64k.py
python scripts/deployment_smoke.py \
  --frontend YOUR_TEST_FRONTEND \
  --name qwen35-9b-lora-16k \
  --steps 3 \
  --output /tmp/lora16-smoke.json \
  --continue-file /tmp/continue-after-redeploy
```

Run the script once per configuration. Each step performs cross-entropy forward/backward on 512 tokens, an Adam update, publication of sampling weights, and generation with a 16-token cap. It checks output lengths and finite trainer/sampler log probabilities. Miles does not support per-client seeds, so the test does not request one.

The optional `--continue-file` keeps the client alive after its initial steps. Apply the complete unchanged YAML set, record trainer boot IDs, then change one YAML and apply the complete set again. Create the file only after both applies finish:

```bash
touch /tmp/continue-after-redeploy
```

Each waiting client runs another training/publication/sampling step and checks that its trainer boot ID is unchanged. Run a fresh client against the changed configuration as well. Inspect inference app IDs and container IDs before and after applies to check pool continuity. The script unloads its model on exit; stop the dedicated frontend and its pools after completing the test.

This checks short-request functionality and continuity across deployment updates. It does not establish maximum-context memory fit, convergence, or performance under sustained load. The JSON reports are local test artifacts and are not committed to the documentation.

## Bugs found during live validation

- The frontend image applied environment settings after `add_local_python_source`; Modal rejected the image build. Environment settings now precede local source mounting.
- The YAML frontend used Python 3.11 to launch rollout deployment subprocesses while serialized GPU images used Python 3.12. The YAML frontend and deployment CLI now require the matching Python version.
- SGLang's integer parser expects strings and calls `.strip()`. YAML integers now go through backend type converters as command-line text would.
- Retiring a configuration changed data captured by its trainer function. The function now captures consistently ordered JSON with fixed admission/routing flags. A regression test checks that retiring a configuration preserves the serialized function and changing GPU resources changes it.
