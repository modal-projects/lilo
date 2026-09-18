"""Keep summed Tinker gradients independent of the shared microbatch count."""

from functools import partial, wraps
from inspect import signature

from megatron.core.pipeline_parallel import schedules
from miles.backends.megatron_utils import model as miles_model
from miles.utils.multi_lora import is_multi_lora_enabled


_TINKER_LOSSES = {"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"}


def configure_deterministic_loss_scaling():
    original = schedules.forward_step_calc_loss
    if getattr(original, "_lilo_sum_loss", False):
        return

    @wraps(original)
    def calculate_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size=None,
        is_last_stage=None,
    ):
        if (
            not collect_non_loss_data
            and isinstance(loss_func, partial)
            and loss_func.func is miles_model.loss_function
        ):
            bound = signature(loss_func.func).bind_partial(
                *loss_func.args, **loss_func.keywords
            )
            args = bound.arguments["args"]
            batch = bound.arguments["batch"]
            if (
                getattr(args, "lilo_deterministic_attention", False)
                and is_multi_lora_enabled(args)
                and batch.get("loss_fn") in _TINKER_LOSSES
                and not args.calculate_per_token_loss
            ):
                # This helper also scales auxiliary losses. Their normalization
                # needs a separate policy before extending this path beyond the
                # dense, MTP-disabled models used by the deterministic definitions.
                if (
                    getattr(config, "num_moe_experts", None)
                    or getattr(config, "mtp_num_layers", None)
                    or getattr(config, "experimental_attention_variant", None)
                    or getattr(
                        config, "experimental_attention_variant_loss_scale_func", None
                    )
                ):
                    raise ValueError(
                        "Deterministic summed Tinker loss requires no auxiliary losses"
                    )
                # Miles multiplies by N and this Megatron helper divides by N.
                # FP32 backward computes round(round(1/N) * N), which differs
                # for N=55 and N=173. Use one at BOTH scalar scaling sites.
                # The outer schedule still executes the real microbatch count.
                bound.arguments["num_microbatches"] = 1
                loss_func = partial(loss_func.func, *bound.args, **bound.kwargs)
                num_microbatches = 1
        return original(
            model,
            output_tensor,
            loss_func,
            config,
            vp_stage,
            collect_non_loss_data,
            num_microbatches,
            forward_data_store,
            cp_group_size=cp_group_size,
            is_last_stage=is_last_stage,
        )

    calculate_loss._lilo_sum_loss = True
    schedules.forward_step_calc_loss = calculate_loss
