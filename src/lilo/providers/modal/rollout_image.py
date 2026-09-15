from pathlib import Path
import shlex

import modal

from .image_dependencies import CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.17"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "stitch-sglang-v0.5.17"
SGLANG_REVISION = "d050d06437d96196fc68d5b4e5c246408790d537"

# Reference-lifetime fix for the pinned SGLang revision. git apply fails the build
# if an upstream update changes this context; keep until the fix is upstream.
SGLANG_LORA_LIFETIME_PATCH = (
    "--- a/python/sglang/srt/managers/tokenizer_manager.py\n"
    "+++ b/python/sglang/srt/managers/tokenizer_manager.py\n"
    "@@ -202,6 +202,14 @@\n"
    " \n"
    " \n"
    " @dataclasses.dataclass\n"
    "+class LoRARequestRef:\n"
    '+    """One registry acquisition shared by a logical request and its children."""\n'
    "+\n"
    "+    lora_id: str\n"
    "+    users: int = 1\n"
    "+\n"
    "+\n"
    "+@dataclasses.dataclass\n"
    " class ReqState:\n"
    '     """Store the state a request."""\n'
    " \n"
    "@@ -215,6 +223,7 @@\n"
    "     abort_requested: bool = False\n"
    "     lifecycle_id: object = dataclasses.field(default_factory=object)\n"
    "     dispatched: bool = False\n"
    "+    lora_ref: Optional[LoRARequestRef] = None\n"
    "     last_completion_tokens: int = 1\n"
    "     ttft_observed: bool = False\n"
    " \n"
    "@@ -1645,14 +1654,8 @@\n"
    "             HTTPStatus.INTERNAL_SERVER_ERROR,\n"
    "             CLIENT_CLOSED_REQUEST,\n"
    "         ):\n"
    "-            # Delete the key to prevent resending abort request to the scheduler and\n"
    "-            # to ensure aborted request state is cleaned up.\n"
    "-            if state.obj.rid in self.rid_to_state:\n"
    "-                self._remove_req_state(state.obj.rid)\n"
    "-\n"
    "-            # Mark ongoing LoRA request as finished.\n"
    "-            if self.enable_lora and state.obj.lora_path:\n"
    "-                await self.lora_registry.release(state.obj.lora_id)\n"
    "+            # Request-state removal owns the LoRA release. This consumer may\n"
    "+            # run after the RID has already been reused by a new request.\n"
    "             if not is_stream:\n"
    "                 raise fastapi.HTTPException(\n"
    '                     status_code=finish_reason["status_code"],\n'
    "@@ -2493,10 +2496,6 @@\n"
    "                     )\n"
    " \n"
    "                 self._remove_req_state(rid)\n"
    "-\n"
    "-                # Mark ongoing LoRA request as finished.\n"
    "-                if self.enable_lora and state.obj.lora_path:\n"
    "-                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))\n"
    " \n"
    "             if out_dict is not None:\n"
    "                 state.out_list.append(out_dict)\n"
    "@@ -3384,13 +3383,35 @@\n"
    '                     f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"\n'
    "                 )\n"
    " \n"
    "-        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.\n"
    "-        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)\n"
    "-        # Propagate lora_id to any sub-objects already cached by __getitem__.\n"
    '-        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():\n'
    "-            sub_obj.lora_id = (\n"
    "-                obj.lora_id[i] if isinstance(obj.lora_id, list) else obj.lora_id\n"
    "-            )\n"
    "+        # Acquire once per logical request, not once per expanded n-sample slot.\n"
    "+        # Children share the parent's reference until the final child finishes.\n"
    "+        states = [self.rid_to_state[rid] for rid in self._logical_rids(obj)]\n"
    "+        paths = [state.obj.lora_path for state in states]\n"
    "+\n"
    "+        async def acquire_and_attach():\n"
    "+            ids = await self.lora_registry.acquire(paths)\n"
    "+            obj.lora_id = ids[0] if obj.is_single else ids\n"
    "+            for state, lora_id in zip(states, ids):\n"
    "+                state.obj.lora_id = lora_id\n"
    "+                if lora_id is None:\n"
    "+                    continue\n"
    "+                if self.rid_to_state.get(state.obj.rid) is state:\n"
    "+                    state.lora_ref = LoRARequestRef(lora_id)\n"
    "+                else:\n"
    "+                    # Cancellation removed this state while acquisition waited.\n"
    "+                    self._track_lora_task(self.lora_registry.release(lora_id))\n"
    "+\n"
    "+        # Cancellation cannot interrupt a partially acquired batch or leave an\n"
    "+        # acquisition unattached. Cleanup can remove states while this finishes.\n"
    "+        await asyncio.shield(self._track_lora_task(acquire_and_attach()))\n"
    "+\n"
    "+    def _track_lora_task(self, coroutine):\n"
    "+        # Hold strong references to cleanup tasks after HTTP consumers disappear.\n"
    '+        tasks = self.__dict__.setdefault("_lora_tasks", set())\n'
    "+        task = asyncio.create_task(coroutine)\n"
    "+        tasks.add(task)\n"
    "+        task.add_done_callback(tasks.discard)\n"
    "+        return task\n"
    " \n"
    "     @staticmethod\n"
    "     def _logical_rids(obj) -> List[str]:\n"
    "@@ -3424,6 +3445,10 @@\n"
    "             request,\n"
    "             lifecycle_id=logical_state.lifecycle_id,\n"
    "         )\n"
    "+        child_state = self.rid_to_state[obj.rid]\n"
    "+        child_state.lora_ref = logical_state.lora_ref\n"
    "+        if child_state.lora_ref is not None:\n"
    "+            child_state.lora_ref.users += 1\n"
    "         try:\n"
    "             self._register_child_rid(logical_rid, obj.rid)\n"
    "         except BaseException:\n"
    "@@ -3442,6 +3467,12 @@\n"
    "         ):\n"
    "             return None\n"
    "         self.rid_to_state.pop(rid)\n"
    "+        ref, state.lora_ref = state.lora_ref, None\n"
    "+        if ref is not None:\n"
    "+            assert ref.users > 0\n"
    "+            ref.users -= 1\n"
    "+            if ref.users == 0:\n"
    "+                self._track_lora_task(self.lora_registry.release(ref.lora_id))\n"
    "         logical_rid = self.child_rid_to_logical_rid.pop(rid, None)\n"
    "         if logical_rid is not None:\n"
    "             children = self.logical_rid_to_child_rids.get(logical_rid)\n"
    "@@ -3536,9 +3567,9 @@\n"
    "     ):\n"
    '         """Drop all logical and child state owned by *obj*.\n'
    " \n"
    "-        Safe to call after a partial/failed dispatch: only requests known to have\n"
    "-        reached the scheduler are aborted, all owned state is removed, and a later\n"
    "-        output for a discarded RID is ignored by the scheduler-response path.\n"
    "+        Undispatched states release immediately. Dispatched LoRA states stay\n"
    "+        alive until the scheduler acknowledges their abort or normal completion;\n"
    "+        dropping them earlier either leaks the reference or unloads live weights.\n"
    '         """\n'
    "         if lifecycle_ids is None:\n"
    "             lifecycle_ids = {\n"
    "@@ -3577,9 +3608,14 @@\n"
    '                         "Failed to abort request rid=%s",\n'
    "                         target_rid,\n"
    "                     )\n"
    "-            for child_rid in child_rids:\n"
    "-                self._remove_req_state(child_rid, lifecycle_id)\n"
    "-            self._remove_req_state(logical_rid, lifecycle_id)\n"
    "+            for rid in (*child_rids, logical_rid):\n"
    "+                state = self.rid_to_state.get(rid)\n"
    "+                if state is None or state.lifecycle_id is not lifecycle_id:\n"
    "+                    continue\n"
    "+                if state.dispatched and state.lora_ref is not None:\n"
    "+                    state.abort_requested = True\n"
    "+                else:\n"
    "+                    self._remove_req_state(rid, lifecycle_id)\n"
    " \n"
    "     def _should_dispatch_to_encoder(\n"
    "         self, obj: Union[GenerateReqInput, EmbeddingReqInput]\n"
    "--- a/python/sglang/srt/lora/lora_registry.py\n"
    "+++ b/python/sglang/srt/lora/lora_registry.py\n"
    "@@ -142,27 +142,24 @@\n"
    "             self._registry.move_to_end(name)\n"
    "             return lora_ref.lora_id\n"
    " \n"
    "-        if isinstance(lora_name, str):\n"
    "-            async with self._registry_lock.writer_lock:\n"
    "-                lora_id = _lookup(lora_name)\n"
    "-\n"
    "-            await self._counters[lora_id].increment(notify_all=False)\n"
    "-            return lora_id\n"
    "-        elif isinstance(lora_name, list):\n"
    "-            async with self._registry_lock.writer_lock:\n"
    "-                lora_ids = [_lookup(name) for name in lora_name]\n"
    "-\n"
    "-            # Increment the counters only after all IDs are looked up.\n"
    "-            await asyncio.gather(\n"
    "-                *[\n"
    "-                    self._counters[id].increment(notify_all=False)\n"
    "-                    for id in lora_ids\n"
    "-                    if id is not None\n"
    "-                ]\n"
    "-            )\n"
    "-            return lora_ids\n"
    "-        else:\n"
    "+        if not isinstance(lora_name, (str, list)):\n"
    '             raise TypeError("lora_name must be either a string or a list of strings.")\n'
    "+        names = [lora_name] if isinstance(lora_name, str) else lora_name\n"
    "+        async with self._registry_lock.writer_lock:\n"
    "+            # Lookup and pin are atomic with respect to unregister. Otherwise an\n"
    "+            # eviction can observe zero and delete a counter before its increment.\n"
    "+            ids = [_lookup(name) for name in names]\n"
    "+            acquired = []\n"
    "+            try:\n"
    "+                for lora_id in ids:\n"
    "+                    if lora_id is not None:\n"
    "+                        await self._counters[lora_id].increment(notify_all=False)\n"
    "+                        acquired.append(lora_id)\n"
    "+            except BaseException:\n"
    "+                for lora_id in acquired:\n"
    "+                    await self._counters[lora_id].decrement()\n"
    "+                raise\n"
    "+        return ids[0] if isinstance(lora_name, str) else ids\n"
    " \n"
    "     async def release(self, lora_id: Union[str, List[str]]):\n"
    '         """\n'
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
        + SGLANG_LORA_LIFETIME_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_LORA_LIFETIME_PATCH
        + "PATCH\n",
        "rm -rf /sgl-workspace/sglang/python/sglang"
        " && cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /tmp/stitch-sglang-overlay",
    )
    .run_commands(
        "python -c "
        + shlex.quote(
            "exec("
            + repr(Path(__file__).with_name("sglang_determinism_patch.py").read_text())
            + ")"
        )
        + " /sgl-workspace/sglang/python/sglang"
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
