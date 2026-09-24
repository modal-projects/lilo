from lilo.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-fft-64k"
    model = "Qwen/Qwen3.5-9B"
    trainer = {
        **Parent.trainer,
        "gpu": "H200",
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
    }
    inference = {
        **Parent.inference,
        "gpu": "H200",
        "config": {**Parent.inference["config"], "ep_size": 1},
    }


config = Config()
