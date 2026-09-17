# Scoped recovery fixes

Changes against PR #21 at `95e4c5542c96c13ad18e1fa1ac9301f1a7e04e88`.

## Model replacement after container restart

The scoped slot manager now distinguishes a live trainer invocation from a live
model placement. It permits replacement when the owning boot changed or the
control plane has already reported model loss and removed the model's demand.
An unplaced model with outstanding demand still owns its slot: another client
cannot displace a healthy or initializing model. Replacement retires the old
model, and retries of its creation cannot resurrect it.

The restarted trainer can accept the new model without cancelling its invocation
or increasing the trainer limit. Existing terminal-invocation recovery remains
supported.

## Checkpoint API wiring

`ModalCheckpointStorage` provides the existing metadata, listing, and deletion
operations for both shared and scoped deployments. The scoped API binds these
callbacks to its configured checkpoint volume. The existing shared API keeps its
callbacks and lock, delegating to the same implementation.

This enables the public Tinker
`create_training_client_from_state_with_optimizer()` method inside a scope. The
scoped control-plane subclass remains unchanged; these fixes do not change the
base control plane's model-creation callback semantics.

## Validation

- Full CPU suite: **306 passed, 2 skipped**.
- New regression cases cover a boot change before and after the old client's
  loss response; reuse of the existing trainer; retirement of the old client;
  and rejection of competing clients while the current model is healthy or
  initializing.
- Existing checkpoint read/list/delete and scope-lifecycle tests pass.

The live test is `scripts/scoped_restart_e2e.py`:

```sh
MODAL_ENVIRONMENT=your-test-environment PYTHONPATH=src:scripts \
  python scripts/scoped_restart_e2e.py --report /tmp/scoped-restart.json
```

It uses four H100s for Qwen3.5-4B, saves an optimizer checkpoint, sends ContainerStop
to the trainer belonging to its own app, verifies the same invocation restarts
with a different boot, and triggers recovery through the old client's HTTP 410.
It creates a second client through the public SDK checkpoint method and verifies
two subsequent training updates in the same scope and the same trainer invocation.
There is no invocation-cancellation fallback. It also tests metadata lookup,
checkpoint listing/deletion, old-client rejection, and final app cleanup.

### Live result — 2026-09-16

**Recovery and continued training succeeded, but one numerical check failed.**
Of 21 checks, 20 passed. The script exited nonzero because the log-probability
comparison exceeded its tolerance.

The test verified:

- A real ContainerStop restarted the same invocation with a different boot ID.
- The old client's normal forward request returned HTTP 410 `model_lost`.
- `create_training_client_from_state_with_optimizer()` created the second client
  in that scope. Its stored model specification has `restore_optimizer: true`.
- The second client reused the same invocation and new boot, with exactly one
  running trainer. No invocation cancellation or extra trainer was needed.
- Two resumed forward/backward and optimizer steps succeeded. Their mean losses
  were `0.3280329` and `0.0371729`; the first update changed the model's outputs,
  and a final forward returned finite log probabilities.
- The old client remained lost after replacement.
- SDK checkpoint metadata, listing, and deletion all worked.
- The app reached STOPPED with zero containers and zero tasks. An independent
  volume lookup verified checkpoint deletion; its empty parent was removed.

The diagnostic compared forward log probabilities before failure with those
immediately after restoration. Maximum absolute difference was `0.0047957003`
over 13 tokens, exceeding the existing `0.003` tolerance. Its cause is unresolved;
no tolerance was loosened to make the run pass. The test confirms that training resumes after recovery. It does not establish
numerical equivalence across trainer processes. Transient HTTP 500 responses during future retrieval
were retried by the SDK, and the scenario reached completion.

The [Modal run](https://modal.com/apps/modal-labs/connor-dev-2/ap-gIKuOfNPbED4miFTNdGDnG)
used only a new test app. Reports under `scripts/results/scoped-e2e/`:

- `restart-fix.json` and `.log`: checks, metrics, identities, and runtime output.
- `restart-fix-model-specs.json`: the replacement's checkpoint/optimizer settings.
- `restart-fix-cleanup.json`: independent app and checkpoint cleanup verification.
- `restart-fix-pytest.log`: full CPU suite result.
- `restart-fix-harness.py` and `restart-fix-runtime.patch`: reproduction snapshots.

## Issues still open at the time of this test

The earlier run also found that an incomplete model download was accepted as a
valid cache when `config.json` existed. It found a misleading SDK connection
error when a training slot was already occupied. Neither issue was addressed by
these recovery fixes. The numerical difference also needs a repeatability test
before it can be attributed to optimizer restoration.
