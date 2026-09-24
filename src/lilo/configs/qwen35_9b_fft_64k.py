from lilo.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-fft-64k"
    model = "Qwen/Qwen3.5-9B"
    overrides = {
        "trainer_gpu": "H200",
        "trainer_env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        "inference_gpu": "H200",
        "sglang_cfg.ep_size": 1,
    }


config = Config()
