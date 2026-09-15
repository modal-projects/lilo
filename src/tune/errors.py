class TinkerModalError(Exception):
    pass


class RecordNotFound(TinkerModalError):
    def __init__(self, kind: str, object_id: str) -> None:
        self.kind = kind
        self.object_id = object_id
        super().__init__(f"{kind} not found: {object_id}")


class RecordUnavailable(TinkerModalError):
    def __init__(self, kind: str, object_id: str, state: str) -> None:
        self.kind = kind
        self.object_id = object_id
        self.state = state
        super().__init__(f"{kind} is {state}: {object_id}")


class SequenceConflict(TinkerModalError):
    def __init__(self, object_id: str, seq_id: int) -> None:
        self.object_id = object_id
        self.seq_id = seq_id
        super().__init__(f"conflicting request for {object_id} sequence {seq_id}")


class EngineSaturated(TinkerModalError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"engine refused work: {reason}")


class ModelLost(TinkerModalError):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"engine owning model {model_id} terminated")
