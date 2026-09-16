from dataclasses import replace
from unittest.mock import patch

import modal
import pytest

from lilo.engines import qwen3_5_4b_full_64k
from lilo.providers.modal.scoped import build_app
from lilo.run import Pool


def test_config_only_asset_cache_is_resumed_before_commit(tmp_path):
    """A failed download can leave config.json without weights or a tokenizer."""
    (tmp_path / "config.json").write_text("{}")
    engine = qwen3_5_4b_full_64k()
    engine = replace(engine, training=replace(engine.training, hf_checkpoint=str(tmp_path)))
    resources = build_app(
        engine, "asset-cache-test", "asset-cache-test", "test-key", 1,
        Pool(), Pool(), "test-checkpoints", modal.Secret.from_dict({}),
    )
    prepare = resources[4]
    with (
        patch("huggingface_hub.snapshot_download") as download,
        patch.object(modal.Volume, "commit") as commit,
    ):
        download.side_effect = OSError("interrupted download")
        with pytest.raises(OSError, match="interrupted"):
            prepare.local()
        commit.assert_not_called()

        download.side_effect = None
        prepare.local()
        assert download.call_count == 2
        download.assert_called_with(
            engine.model, revision=engine.revision, local_dir=str(tmp_path)
        )
        commit.assert_called_once()
