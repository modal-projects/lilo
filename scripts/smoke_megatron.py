import os
import time

import tinker
from tinker import types

BASE_URL = os.environ["TINKER_BASE_URL"]
API_KEY = os.environ["TINKER_API_KEY"]
BASE_MODEL = "Qwen/Qwen3-4B"

service = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
print(
    "capabilities:",
    [m.model_name for m in service.get_server_capabilities().supported_models],
)

start = time.time()
training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)
print(f"training client ready in {time.time() - start:.0f}s")

datum = types.Datum(
    model_input=types.ModelInput.from_ints([151644, 872, 198, 9707, 151645, 198]),
    loss_fn_inputs={
        "target_tokens": [872, 198, 9707, 151645, 198, 151643],
        "weights": [1.0] * 6,
    },
)
for step in range(3):
    fb = training.forward_backward([datum] * 4, "cross_entropy")
    opt = training.optim_step(types.AdamParams(learning_rate=1e-4))
    fb_result = fb.result(timeout=600)
    opt_result = opt.result(timeout=600)
    print(
        f"step {step}: fb metrics={fb_result.metrics}"
        f" optim metrics={opt_result.metrics}"
    )

saved = training.save_state(name="smoke").result(timeout=600)
print("save_state:", saved)
