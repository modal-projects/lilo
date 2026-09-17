import modal

from .image_dependencies import CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.17"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "stitch-sglang-v0.5.17"
SGLANG_REVISION = "d050d06437d96196fc68d5b4e5c246408790d537"

# Narrow source fix for the pinned SGLang revision. git apply fails the build
# if an upstream update changes this context; keep until the fix is upstream.
SGLANG_LORA_ABORT_PATCH = (
    "--- a/python/sglang/srt/managers/tokenizer_manager.py\n"
    "+++ b/python/sglang/srt/managers/tokenizer_manager.py\n"
    "@@ -1650,9 +1650,8 @@\n"
    "             if state.obj.rid in self.rid_to_state:\n"
    "                 self._remove_req_state(state.obj.rid)\n"
    " \n"
    "-            # Mark ongoing LoRA request as finished.\n"
    "-            if self.enable_lora and state.obj.lora_path:\n"
    "-                await self.lora_registry.release(state.obj.lora_id)\n"
    "+            # Scheduler completion handlers own the LoRA reference release.\n"
    "+            # Releasing again here makes the counter negative, blocking eviction.\n"
    "             if not is_stream:\n"
    "                 raise fastapi.HTTPException(\n"
    '                     status_code=finish_reason["status_code"],\n'
    "@@ -3237,6 +3236,10 @@\n"
    '             "meta_info": meta_info,\n'
    "         }\n"
    "         self._remove_req_state(recv_obj.rid)\n"
    "+        # This scheduler completion path bypasses _handle_batch_output. Release\n"
    "+        # here even for aborts without an HTTP error code or a live consumer.\n"
    "+        if self.enable_lora and state.obj.lora_path:\n"
    "+            asyncio.create_task(self.lora_registry.release(state.obj.lora_id))\n"
    " \n"
    "         state.out_list.append(out)\n"
    "         state.event.set()\n"
)

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/stitch-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/stitch-sglang-overlay"
        f" && git -C /tmp/stitch-sglang-overlay checkout --detach {SGLANG_REVISION}",
        "cd /tmp/stitch-sglang-overlay && git apply --check - <<'PATCH'\n"
        + SGLANG_LORA_ABORT_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_LORA_ABORT_PATCH
        + "PATCH\n",
        "rm -rf /sgl-workspace/sglang/python/sglang"
        " && cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /tmp/stitch-sglang-overlay",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
        }
    )
    .add_local_python_source("lilo")
)
