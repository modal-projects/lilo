"""Allow the validated hdim256 FA3 prototype through TE's capability filter."""

import ast
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sysconfig


def patch_transformer_engine(expected_binary_hashes):
    distribution = importlib.metadata.distribution("flash_attn_3")
    binaries = [p for p in distribution.files if str(p).endswith(".so")]
    actual = {
        str(p): hashlib.sha256(distribution.locate_file(p).read_bytes()).hexdigest()
        for p in binaries
    }
    if actual != expected_binary_hashes:
        raise RuntimeError(
            f"Refusing TE capability exception for an unverified FA3 binary: {actual}"
        )
    path = (
        Path(sysconfig.get_paths()["purelib"])
        / "transformer_engine/pytorch/attention/dot_product_attention/utils.py"
    )
    source = path.read_text()
    old = "if is_training and max(head_dim_qk, head_dim_v) >= 256:"
    new = """if is_training and max(head_dim_qk, head_dim_v) >= 256 and not (
            max(head_dim_qk, head_dim_v) == 256
            and device_compute_capability == (9, 0)
            and os.getenv("LILO_PARITY_FA3_HD256") == "1"
        ):"""
    if source.count(old) != 1:
        raise RuntimeError(
            "Transformer Engine's hdim256 filter changed; inspect before patching"
        )
    patched = source.replace(old, new)
    ast.parse(patched)
    path.write_text(patched)
    os.environ["LILO_PARITY_FA3_HD256"] = "1"
    return {
        "binary_sha256": actual,
        "te_original_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "te_patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
    }


if __name__ == "__main__":
    import json
    import sys

    print(
        json.dumps(
            patch_transformer_engine(
                json.loads(Path(sys.argv[1]).read_text())["binary_sha256"]
            )
        )
    )
