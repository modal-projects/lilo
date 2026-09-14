"""Adapt Stitch's run switching to a scoped deployment's current model."""
from contextvars import ContextVar

from stitch.service import create_app
from stitch.sync import Reconciler
from stitch.types import VersionRef
from starlette.responses import JSONResponse

from .full_bulletin import FFTSnapshotStore

_expected_run = ContextVar('scoped_sampling_run', default=None)


class AssignedSnapshotStore(FFTSnapshotStore):
    def __init__(self, bulletin, run_id, registry):
        super().__init__(bulletin, run_id)
        self.registry = registry

    def refresh(self):
        super().refresh()
        self.run_id = self.registry.get('slot:0')
        if not self.run_id:
            raise RuntimeError('latest sampler has no assigned model')

    def read_pointer(self):
        return self.bulletin.read_latest(self.run_id) or VersionRef(self.run_id, 0)

    def _check_run(self, ref):
        # Reconciliation may finish reading the previous run while assignment
        # changes. Immutable references select their own directories; admission
        # below prevents the wrong run from serving a request.
        if not ref.run_id:
            raise ValueError('snapshot run identity is required')

    def publish(self, *args, **kwargs):
        raise RuntimeError('assigned snapshot store is read-only')

    def advance_pointer(self, *args, **kwargs):
        raise RuntimeError('assigned snapshot store is read-only')

    def claim(self, *args, **kwargs):
        raise RuntimeError('assigned snapshot store is read-only')


async def retire_replica():
    """End this container through its init; Modal replenishes the same server."""
    import asyncio
    import os
    import signal
    os.kill(1, signal.SIGTERM)
    # Keep admission closed until the container exits, even if shutdown takes time.
    await asyncio.Future()


class AssignedReconciler(Reconciler):
    async def _switch_run(self, new_run):
        if (new_run != self.run_id
                and getattr(self.engine, 'delta_update_mode', None) == 'cpu'):
            # CPU-cache reset is unsupported. Do not relabel patched weights as
            # the new run, or call the disk-reset implementation by accident.
            await self.commit(apply=retire_replica, on_applied=lambda: None,
                              drain_all=True)
            return
        await super()._switch_run(new_run)

    def _rejection(self, constraint):
        # Called under Stitch's admission/commit lock. Middleware checks alone
        # would race a run switch between checking identity and admission.
        expected = _expected_run.get()
        if expected is not None and (self.applied is None or self.applied.run_id != expected):
            return {'type': 'WeightRunNotReady', 'message': 'replica has not switched to the requested model'}
        return super()._rejection(constraint)


def assigned_app(reconciler, engine, registry):
    app = create_app(reconciler, engine)

    @app.middleware('http')
    async def require_current_run(request, call_next):
        if request.url.path.strip('/') not in ('generate', 'v1/completions', 'v1/chat/completions'):
            return await call_next(request)
        body = await request.json()
        expected = body.get('weight_run_id')
        current = await registry.get.aio('slot:0')
        if not expected or expected != current:
            return JSONResponse({'error': 'sampling model was replaced; use a new sampling client'}, status_code=410)
        token = _expected_run.set(expected)
        try:
            return await call_next(request)
        finally:
            _expected_run.reset(token)
    return app


def serve_assigned(store, engine, *, run_id, registry, host, port):
    import uvicorn
    reconciler = AssignedReconciler(store=store, engine=engine, run_id=run_id)
    uvicorn.run(assigned_app(reconciler, engine, registry), host=host, port=port, log_level='info')
