from __future__ import annotations

import time

from tinker import ServiceClient, TrainingClient, types
from tinker.lib.api_future_impl import _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.queue_state_logger import QueueStateLogger


def _create_full_training_client_submit(
    service: ServiceClient,
    base_model: str,
    user_metadata: dict[str, str] | None,
    rollout: dict[str, int] | None,
) -> AwaitableConcurrentFuture[TrainingClient]:
    session_id = service.holder.get_session_id()
    model_seq_id = service.holder.get_training_client_id()

    async def create() -> TrainingClient:
        started_at = time.time()
        request = types.CreateModelRequest(
            session_id=session_id,
            model_seq_id=model_seq_id,
            base_model=base_model,
            user_metadata=user_metadata,
        )
        with service.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.create(
                request=request,
                extra_body={
                    "parameterization": {"type": "full"},
                    "rollout": rollout,
                },
            )
        response = await _APIFuture(
            types.CreateModelResponse,
            service.holder,
            future,
            request_start_time=started_at,
            request_type="CreateModel",
            queue_state_observer=QueueStateLogger(base_model, "Model creation"),
        ).result_async()
        return TrainingClient(
            service.holder,
            model_seq_id=model_seq_id,
            model_id=response.model_id,
        )

    return service.holder.run_coroutine_threadsafe(create())


def create_full_training_client(
    service: ServiceClient,
    base_model: str,
    *,
    user_metadata: dict[str, str] | None = None,
    rollout: dict[str, int] | None = None,
) -> TrainingClient:
    return _create_full_training_client_submit(
        service,
        base_model,
        user_metadata,
        rollout,
    ).result()


async def create_full_training_client_async(
    service: ServiceClient,
    base_model: str,
    *,
    user_metadata: dict[str, str] | None = None,
    rollout: dict[str, int] | None = None,
) -> TrainingClient:
    return await _create_full_training_client_submit(
        service,
        base_model,
        user_metadata,
        rollout,
    ).result_async()
