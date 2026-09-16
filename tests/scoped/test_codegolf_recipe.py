from lilo.engines import qwen3_5_9b_full_64k


def test_codegolf_preserves_existing_eight_gpu_recipe():
    engine = qwen3_5_9b_full_64k()
    engine.validate()
    assert engine.model == "Qwen/Qwen3.5-9B"
    assert engine.trainer_gpu == "H200:8"
    assert engine.sampler_gpu == "H200:1"
    config = engine.training
    assert config.tensor_model_parallel_size == config.context_parallel_size == 2
    assert config.seq_length == config.max_tokens_per_microbatch == 65536
    assert config.sequence_parallel and config.use_distributed_optimizer
    assert config.defer_fp32_logits and config.fp32_lm_head
    assert config.optimizer.loss_scale == 1.0
    assert config.provider_overrides == {
        "mtp_num_layers": 0,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
    }
