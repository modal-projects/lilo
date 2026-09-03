import json
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from runtime_stubs import backend_runtime_imports

with backend_runtime_imports():
    from lilo.backends.megatron_runtime.common import checkpoint_io

commit_checkpoint_volume = checkpoint_io.commit_checkpoint_volume
rank_tag = checkpoint_io.rank_tag
reload_checkpoint_volume = checkpoint_io.reload_checkpoint_volume


def mock_volume_runtime(
    monkeypatch,
    *,
    rank: int,
) -> tuple[Mock, Mock, Mock]:
    volume = Mock()
    from_name = Mock(return_value=volume)
    modal = ModuleType("modal")
    modal.Volume = SimpleNamespace(from_name=from_name)

    distributed = ModuleType("torch.distributed")
    distributed.get_rank = Mock(return_value=rank)
    distributed.barrier = Mock()

    monkeypatch.setattr(checkpoint_io, "modal", modal)
    monkeypatch.setattr(checkpoint_io, "dist", distributed)
    return volume, from_name, distributed


def test_rank_tag_includes_active_parallel_dimensions(monkeypatch) -> None:
    parallel_state = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 1,
        get_pipeline_model_parallel_rank=lambda: 2,
        get_context_parallel_rank=lambda: 3,
        get_context_parallel_world_size=lambda: 4,
        get_expert_model_parallel_rank=lambda: 5,
        get_expert_model_parallel_world_size=lambda: 8,
        get_expert_tensor_parallel_rank=lambda: 6,
        get_expert_tensor_parallel_world_size=lambda: 1,
        get_data_parallel_rank=lambda: 7,
    )
    monkeypatch.setattr(checkpoint_io, "parallel_state", parallel_state)

    assert rank_tag() == "tp1_pp2_cp3_ep5_dp7"


def test_commit_prefers_checkpoint_volume_on_rank_zero(monkeypatch) -> None:
    volume, from_name, distributed = mock_volume_runtime(
        monkeypatch,
        rank=0,
    )
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "fft-checkpoints")
    monkeypatch.setenv("LILO_BULLETIN_VOLUME", "sampler-bulletin")

    commit_checkpoint_volume("persistence")

    distributed.barrier.assert_called_once_with(group="persistence")
    distributed.get_rank.assert_called_once_with(group="persistence")
    from_name.assert_called_once_with("fft-checkpoints")
    volume.commit.assert_called_once_with()


def test_commit_skips_volume_commit_on_nonzero_rank(monkeypatch) -> None:
    volume, from_name, distributed = mock_volume_runtime(
        monkeypatch,
        rank=1,
    )
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "fft-checkpoints")

    commit_checkpoint_volume("persistence")

    distributed.barrier.assert_called_once_with(group="persistence")
    distributed.get_rank.assert_called_once_with(group="persistence")
    from_name.assert_not_called()
    volume.commit.assert_not_called()


def test_single_rank_commit_uses_no_distributed_collective(monkeypatch) -> None:
    volume, from_name, distributed = mock_volume_runtime(
        monkeypatch,
        rank=0,
    )
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "fft-checkpoints")

    commit_checkpoint_volume(None)

    distributed.barrier.assert_not_called()
    distributed.get_rank.assert_not_called()
    from_name.assert_called_once_with("fft-checkpoints")
    volume.commit.assert_called_once_with()


def test_reload_checkpoint_volume_synchronizes_rank_zero(monkeypatch) -> None:
    volume, from_name, distributed = mock_volume_runtime(
        monkeypatch,
        rank=0,
    )
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "fft-checkpoints")

    reload_checkpoint_volume()

    assert distributed.barrier.call_count == 2
    from_name.assert_called_once_with("fft-checkpoints")
    volume.reload.assert_called_once_with()


def test_reload_checkpoint_volume_waits_on_nonzero_rank(monkeypatch) -> None:
    volume, from_name, distributed = mock_volume_runtime(
        monkeypatch,
        rank=1,
    )
    monkeypatch.setenv("LILO_CHECKPOINT_VOLUME", "fft-checkpoints")

    reload_checkpoint_volume()

    assert distributed.barrier.call_count == 2
    from_name.assert_not_called()
    volume.reload.assert_not_called()


def test_write_checkpoint_metadata_on_rank_zero(tmp_path, monkeypatch) -> None:
    mock_volume_runtime(monkeypatch, rank=0)
    metadata = {"schema_version": 1, "parameterization": {"type": "full"}}

    checkpoint_io.write_checkpoint_metadata(str(tmp_path), metadata)

    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata


def test_write_checkpoint_metadata_skips_nonzero_rank(tmp_path, monkeypatch) -> None:
    mock_volume_runtime(monkeypatch, rank=1)

    checkpoint_io.write_checkpoint_metadata(str(tmp_path), {"schema_version": 1})

    assert not (tmp_path / "metadata.json").exists()
