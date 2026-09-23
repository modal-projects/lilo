from dataclasses import replace

from lilo.configs.qwen35_4b_fft_64k import config as base

config = replace(
    base,
    name="qwen35-9b-fft-64k",
    model=replace(base.model, id="Qwen/Qwen3.5-9B"),
    trainer=replace(
        base.trainer,
        compute=replace(base.trainer.compute, gpu="H200"),
        env={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
    ),
    inference=replace(
        base.inference,
        compute=replace(base.inference.compute, gpu="H200"),
        config={**base.inference.config, "ep_size": 1},
    ),
)
