# Scoped-run E2E tests

`scripts/scoped_e2e.py` exercises real Modal provisioning and HTTP clients. Run it
from the PR checkout with Python 3.12 and the project's dependencies installed.
Set `MODAL_ENVIRONMENT` explicitly. The environment must contain the `lilo-proxy`
secret. Each run uses an automatically generated `lilo-e2e-*` app name.

```sh
export MODAL_ENVIRONMENT=your-test-environment
export PYTHONPATH=src
python scripts/scoped_e2e.py --mode exception --report /tmp/exception.json
python scripts/scoped_e2e.py --mode isolation --report /tmp/isolation.json
python scripts/scoped_e2e.py --mode hard-owner-controller --report /tmp/owner.json
python scripts/scoped_e2e.py --mode gpu \
  --asset-path /assets/lilo-e2e-complete/Qwen3.5-4B \
  --report /tmp/gpu.json
```

The first three scenarios register GPU server definitions but do not invoke GPU
workers. The GPU scenario uses the Qwen 4B recipe: four trainer H100s and one
replica each for base, latest and pinned sampling (up to seven H100s).

## Checks

- `exception`: pinned child ownership, active lease protection, body exception
  propagation, and zero containers after parent/child shutdown.
- `isolation`: two simultaneous contexts, independent API keys and session
  stores, no trainer invocation with `warm=False`, and an outer context that
  remains usable after the inner one exits.
- `hard-owner-controller`: creates a dedicated child process, waits for its
  parent and pinned app, sends SIGKILL, and checks automatic app/container
  cleanup. This tests owner death without running `finally` blocks.
- `gpu`: trainer warmup before model acceptance; authentication; real forward/
  backward and optimizer updates; base/latest/pinned sampling; base and pinned
  immutability across updates; rejection of a second active model; actual
  pinned-app shutdown followed by recreation with a different app ID;
  checkpoint metadata and create-from-checkpoint HTTP routes; trainer-container
  restart; stale training-client rejection; replacement after restart and after
  explicit terminal cancellation if necessary; optimizer checkpoint restore;
  numerical comparison with uninterrupted continuation; HTTP 410 for retired
  latest sampling clients; pinned sampling after trainer replacement; cleanup.

The pinned idle-eviction test ages only the test-owned demand record, avoiding a
10-minute idle wait. It waits for the old Modal app to reach STOPPED before
requesting the same pinned handle again. The active-lease probe also uses a
synthetic idle timestamp. Neither replaces a real long-running sample test.

Checkpoint continuation compares per-token forward log probabilities using a
0.003 maximum absolute tolerance. Deterministic sampling comparisons use greedy
12-token outputs plus log probabilities. The report retains observed errors and
metrics so a tolerance failure can be investigated.

## Results and cleanup

JSON reports are written atomically after each event. Checks distinguish
functional failures from fatal scenario errors. The test continues independent
checks after an expected capability failure; a report with any failed check has
`passed: false` and normally exits nonzero. No API keys are written to reports.

The GPU restart targets a trainer FunctionCall within the test app; it verifies
the target container belongs to that app before sending ContainerStop. It does
not target an existing training run. Parent/child apps are ephemeral, and normal
exit polls until they are stopped with zero containers. Named checkpoints and
model assets persist, as they do in the production feature.

If startup is interrupted before `lilo.run` yields, record the app ID printed by
Modal as well as the JSON report. A hard-killed runner cannot write its final
report; the hard-owner controller records cleanup independently.
