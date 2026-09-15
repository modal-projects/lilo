import hashlib


def _part(value: str) -> str:
    if not value or ":" in value:
        raise ValueError(f"invalid key component: {value!r}")
    return value


def _model_part(value: str) -> str:
    if not value:
        raise ValueError("model_id must be non-empty")
    return (
        value
        if ":" not in value
        else hashlib.sha256(value.encode()).hexdigest()
    )


def session_key(session_id: str) -> str:
    return f"session:{_part(session_id)}"


def session_last_seen_key(session_id: str) -> str:
    return f"session_last_seen:{_part(session_id)}"


def session_closed_key(session_id: str) -> str:
    return f"session_closed:{_part(session_id)}"


def model_creation_key(session_id: str, model_seq_id: int) -> str:
    if model_seq_id < 0:
        raise ValueError("model_seq_id must be non-negative")
    return f"model_creation:{_part(session_id)}:{model_seq_id}"


def model_key(model_id: str) -> str:
    return f"model:{_model_part(model_id)}"


def placement_key(model_id: str) -> str:
    return f"placement:{_model_part(model_id)}"


def placement_claim_key(model_id: str) -> str:
    return f"placement_claim:{_model_part(model_id)}"


def trainer_demand_key(model_id: str) -> str:
    return f"trainer_demand:{_model_part(model_id)}"


def sampling_session_creation_key(session_id: str, seq_id: int) -> str:
    if seq_id < 0:
        raise ValueError("seq_id must be non-negative")
    return f"sampling_session_creation:{_part(session_id)}:{seq_id}"


def sampling_session_key(sampling_session_id: str) -> str:
    return f"sampling_session:{_part(sampling_session_id)}"


def sample_task_key(sampling_session_id: str, seq_id: int) -> str:
    if seq_id < 0:
        raise ValueError("seq_id must be non-negative")
    return f"sample_task:{_part(sampling_session_id)}:{seq_id}"


def sampler_export_submission_key(model_id: str, seq_id: int) -> str:
    if seq_id <= 0:
        raise ValueError("seq_id must be positive")
    return f"sampler_export_submission:{_model_part(model_id)}:{seq_id}"


def sampler_export_result_key(model_id: str, seq_id: int) -> str:
    if seq_id <= 0:
        raise ValueError("seq_id must be positive")
    return f"sampler_export_result:{_model_part(model_id)}:{seq_id}"


def sampler_artifact_key(model_path: str) -> str:
    if not model_path:
        raise ValueError("model_path must be non-empty")
    digest = hashlib.sha256(model_path.encode()).hexdigest()
    return f"sampler_artifact:{digest}"
