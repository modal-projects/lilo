# Scoped-run E2E results — 2026-09-15

Follow-up: [recovery fixes and the 2026-09-16 rerun](scoped-recovery-fixes.md).
The observations below describe the original PR head before those fixes.

PR head: `95e4c5542c96c13ad18e1fa1ac9301f1a7e04e88`.
Environment: `modal-labs/connor-dev-2`; test apps use the `lilo-e2e-` prefix.
Client: Python 3.12.13, Modal 1.5.5, Tinker 0.24.1.

These tests check startup, cleanup, and recovery alongside the existing long
training runs. The GPU scenario uses Qwen3.5-4B on four trainer H100s and individual
H100 sampler replicas, with short cross-entropy training steps.

## Outcome

**36 of 41 checks passed.** All scenarios reached their end and cleanup
completed. The main GPU run lasted 46.6 minutes and exited nonzero for five
failed checks: two checkpoint-API checks, replacement after container restart,
the SDK's occupied-slot error message, and numerical optimizer continuation.
The interrupted-download failure was reproduced in a separate setup run.

| Scenario | Passed / total | Duration |
| --- | ---: | ---: |
| Exception and child cleanup | 4 / 4 | 25.4 s |
| Simultaneous scope isolation | 7 / 7 | 41.5 s |
| Owner SIGKILL cleanup | 3 / 3 | 190.7 s |
| GPU lifecycle and recovery | 22 / 27 | 2795.2 s |

At this revision, normal training and sampling worked. Recovery after container
restart, checkpoint API wiring, incomplete asset downloads, and SDK error
messages needed fixes. The numerical difference after restore needed a
controlled repeatability test.

## Completed CPU lifecycle checks

All 14 checks passed:

- An exception in the context body propagates to the caller. Parent and pinned
  child apps stop with zero containers.
- A pinned request lease prevents idle eviction even when its last-used timestamp
  is aged beyond the idle threshold.
- Two simultaneous runs have independent API keys and session stores.
- `warm=False` creates no trainer invocation; the outer context remains usable
  after the inner context closes.
- SIGKILL of the owner process, bypassing cleanup code, causes both parent and
  pinned child apps to stop automatically. Observed cleanup latency was 176.7 s;
  both apps had zero remaining containers. This owner-loss scenario provisioned
  sampler definitions but did not start GPU sampler replicas.

Reports: `scripts/results/scoped-e2e/exception-final.json`, `isolation.json`, and
`hard-owner.json`.

## Confirmed defects

### Container restart leaves the training slot occupied

Stopping the trainer container caused Modal to restart the same function
invocation with a new boot ID. The old client correctly received HTTP 410
`model_lost`, but creating a replacement client failed. The assignment logic
rejects replacement whenever the previous slot has an active invocation, even
when that invocation has restarted and no longer owns the previous model.
Explicitly cancelling the invocation allowed creation of a replacement model.

Fix: distinguish a live model placement from a live trainer invocation. When the
placement's boot ID no longer matches, retire the lost model and allow a new
model to claim the restarted trainer. Keep rejection of a genuinely live second
model. Test this through ContainerStop, not just FunctionCall cancellation.

Evidence: `same_invocation_restarted`, `old_trainer_reports_loss`, and
`replacement_after_container_restart` in `gpu-validated-harness.json`.

### Saved checkpoint metadata is unavailable through the scoped API

After successfully saving an optimizer checkpoint, `/api/v1/weights_info` and
the create-from-checkpoint `/api/v1/load_weights` route both returned HTTP 409
`checkpoint is metadata unavailable`. The scoped control plane mounts the
checkpoint volume but does not pass the metadata reader used by the shared
deployment to `ControlPlane`. Reading the volume directly confirmed that both
`metadata.json` and `checkpoint_metadata.json` exist, alongside the optimizer
shards (81,777,649,526 bytes total).

Fix: wire scoped checkpoint storage callbacks, including metadata reads and
list/delete operations, using the configured checkpoint volume. Verify the
standard SDK create-from-state path after slot recovery as well as explicit
loading into an existing client.

Evidence: `checkpoint_metadata_endpoint` and
`create_from_checkpoint_reads_metadata` in `gpu-validated-harness.json`.

### Interrupted asset downloads are mistaken for complete checkpoints

An interrupted setup left `config.json`, tokenizer files and the weight index in
`/assets/Qwen3.5-4B`, but neither of its two weight shards. The next normal scoped
run skipped downloading because `config.json` existed and failed during trainer
warmup with a missing `model.language_model.norm.weight` tensor. Its app cleaned
up successfully.

This interruption occurred while correcting the test harness; it exposed a real
production recovery condition rather than a deliberately incomplete model mock.
The incomplete shared cache was subsequently repaired and both indexed shards
were verified present. The main GPU suite uses a separate complete asset path.

Fix: verify complete model assets, use immutable model/revision-specific paths,
and mark downloads complete only after all indexed files are committed.

Evidence: `gpu-final.json`, `gpu-final.log`, `repair-default-assets.log`.

### A second live client gets a misleading SDK connection error

The server correctly rejected creation of a second active model with HTTP 400 and
`a training model is already active in this deployment`. The live response used
chunked transfer encoding and omitted Content-Length. Tinker's
`ClientConnectionPool.aclient` special-cases exactly that response framing and
converts the API error to APIConnectionError. The Lilo helper therefore surfaced
only `Connection error.`

Fix: make this terminal rejection survive the actual Modal/Tinker transport path
(for example, use an appropriate terminal status not affected by Tinker's 400
special case), and retain an SDK-level regression test without Content-Length.

Evidence: `duplicate-model-http-probe.log` and the
`second_live_trainer_rejected` check in `gpu-validated-harness.json`.

## Numerical continuation discrepancy — cause unresolved

The restored model's forward log probabilities matched the saved model within
`1.1920565e-7`. Its next forward/backward loss also matched exactly, and the
optimizer step succeeded. However, forward log probabilities **after** that
update differed from uninterrupted continuation by as much as `0.12693596`,
exceeding the test's `0.003` absolute tolerance. Gradient norms were
`3158.2278` uninterrupted and `3156.0725` after restore.

This is a failed reproducibility check, not yet proof of corrupted optimizer
state or a regression introduced by this PR. The test crosses trainer processes
and does not control GPU kernel nondeterminism or restore random-generator
state. A same-process restore/repeat control and deterministic-backend comparison
are needed to attribute the difference. The toy update uses a learning rate of
`5e-5`; this is a recovery test, not a training-quality evaluation.

## GPU scenario

The complete GPU report is written to
`scripts/results/scoped-e2e/gpu-validated-harness.json`. It records numerical
comparisons, errors, timestamps and all observed owned app IDs. The corresponding
`.log` retains runtime diagnostics. `harness-run.py` and `harness-sha256.txt` record
the test harness used for that run.

Completed numerical and lifecycle checks include:

- Real forward/backward and optimizer steps, and publication to latest samplers.
- Base and pinned samples remain unchanged across training updates: identical
  greedy tokens and zero maximum difference in log probabilities.
- Idle pinned-app eviction reaches STOPPED; the same handle creates a different
  app and returns identical tokens and log probabilities.
- The restarted trainer reports the old model lost with HTTP 410.
- Explicit cancellation permits a replacement training client.
- Loading the optimizer checkpoint into that client restores forward log
  probabilities with maximum absolute error `1.1920565e-7` over 13 tokens
  (test tolerance `0.003`).
- The replacement model publishes and serves valid latest samples. Its greedy
  tokens match uninterrupted continuation; log probabilities differ.
- The retired latest-sampler handle returns HTTP 410 through the actual SDK.
- The old pinned handle recreates its app again after natural idle eviction
  during recovery and returns exactly the original tokens and log probabilities.

## Cleanup and artifacts

All 17 test apps, including setup attempts and three generations of the main
run's pinned app, were verified STOPPED with zero containers. The final child
(`ap-u3b8HIoBClDFuA80Hqh5xW`) was captured by an independent app inventory because
the original harness tracked only the first two generations. The harness now
also records the final pinned route.

Only test-created checkpoint/snapshot directories and the isolated asset cache
were deleted: 130,018,007,729 bytes total. The repaired shared model cache remains.
Checkpoint metadata and a file-size manifest are retained locally for diagnosis.

Local evidence under `scripts/results/scoped-e2e/` includes:

- `completed-summary.json`: completed-scenario counts and failed checks.
- `gpu-validated-harness.json` and `.log`: GPU observations and diagnostics.
- `final-app-inventory.json`: final app states and container counts.
- `artifact-cleanup.json`: exact test-owned paths removed.
- `checkpoint-metadata.json` and `checkpoint-files.json`: checkpoint evidence.
- `harness-run.py` and `harness-sha256.txt`: exact main-run harness snapshot.

After the live run, the harness was updated to synchronize report writes,
assert HTTP 410 for the lost trainer, and track the final recreated pinned app. Those small changes were checked locally; the full
GPU run was not repeated. Runtime source files were not modified.

## Reproduction

See [scoped-e2e-testing.md](scoped-e2e-testing.md) for commands and scenario
coverage. No existing training run is used as a fault-injection target.

## Coverage limits

This is one short-token Qwen3.5-4B GPU scenario plus independent CPU lifecycle
scenarios. It does not establish multi-replica load behavior, long-context
correctness, other model recipes, simultaneous GPU training in multiple scopes,
or isolation between different asset revisions. Idle-eviction and active-lease
tests accelerate the idle clock by aging only test-owned demand records. Owner
SIGKILL uses real parent/child apps with zero GPU replicas.
