"""SGLang settings that must agree with Lilo replica orchestration."""

from lilo.backend_options import native_options

SGLANG_MANAGED = {
    "model_path",
    "model",
    "host",
    "port",
    "context_length",
    "enable_lora",
    "max_lora_rank",
    "enable_cpu_weight_cache",
    "api_key",
    "pp_size",
    "lora_paths",
    "dist_init_addr",
    "nnodes",
    "node_rank",
    "tokenizer_path",
    "tokenizer_revision",
    "revision",
    "grpc_mode",
    "smg_grpc_mode",
    "encoder_only",
    "use_ray",
    "disaggregation_mode",
    "skip_tokenizer_init",
}


def build_config(spec):
    options = native_options(spec.inference.config, SGLANG_MANAGED)
    tp = options.get("tp_size", spec.inference.resources.gpu_count)
    ep = options.get("ep_size", 1)
    if not isinstance(tp, int) or tp != spec.inference.resources.gpu_count:
        raise ValueError("sglang.tp_size must equal the replica GPU allocation")
    if not isinstance(ep, int) or ep < 1 or tp % ep:
        raise ValueError("sglang.ep_size must divide the replica GPU allocation")
    dp = options.get("dp_size", 1)
    dp_attention = options.get("enable_dp_attention", False)
    if type(dp) is not int or dp < 1 or tp % dp:
        raise ValueError("sglang.dp_size must divide the replica GPU allocation")
    if not isinstance(dp_attention, bool):
        raise ValueError("sglang.enable_dp_attention must be a boolean")
    if dp > 1 and not dp_attention:
        raise ValueError("sglang.dp_size > 1 requires enable_dp_attention")
    for key in (
        "max_loaded_loras",
        "max_loras_per_batch",
        "max_running_requests",
        "max_queued_requests",
    ):
        if key in options and (not isinstance(options[key], int) or options[key] < 1):
            raise ValueError(f"sglang.{key} must be positive")
    if not 0 < options.get("mem_fraction_static", 0.8) < 1:
        raise ValueError("sglang.mem_fraction_static must be between zero and one")
    if options.get("max_loaded_loras", 64) < options.get("max_loras_per_batch", 8):
        raise ValueError("max_loaded_loras must be >= max_loras_per_batch")
    return options
