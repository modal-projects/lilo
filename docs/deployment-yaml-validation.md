# YAML deployment validation

These checks exercise the shared-app deployment path in PR #55. Each deployment contains the frontend and generated trainer functions; inference pools are separate apps started on demand.

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

Use Python 3.12 and an authenticated Modal environment. Set `TINKER_API_KEY` locally to the value in the deployment's API secret. Generate the preset YAMLs, give them the same isolated `deployment.frontend`, and pin model revisions plus `LILO_MILES_COMMIT` before deploying.

```bash
lilo deploy qwen35-9b-lora-16k.yaml qwen35-9b-lora-64k.yaml qwen35-4b-fft-64k.yaml
python scripts/yaml_deployment_smoke.py \
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
