"""Preserve FP32 GDN prefix states in our pinned SGLang deterministic build.

The chunk output matmul still consumes BF16 states. Prefix checkpoint storage
must retain the FP32 accumulator so resuming a cached prompt reproduces the
uninterrupted recurrence. Guard replacements against changes in the pinned source.
"""

from pathlib import Path
import sys


def apply(root: Path) -> None:
    def replace(path: Path, old: str, new: str) -> None:
        source = path.read_text()
        if source.count(old) != 1:
            raise RuntimeError(f"SGLang deterministic patch target changed: {path}")
        path.write_text(source.replace(old, new))

    kernel = root / "kernels/ops/attention/fla/chunk_delta_h.py"
    replace(
        kernel, "import torch\n", "import torch\nfrom sglang.srt.environ import envs\n"
    )
    replace(
        kernel,
        "h = k.new_empty(B, NT, H, V, K)",
        "h = torch.empty((B, NT, H, V, K), device=k.device, "
        "dtype=torch.float32 if envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get() else k.dtype)",
    )
    replace(
        kernel.with_name("chunk_o.py"),
        "b_h = tl.load(p_h, boundary_check=(0, 1))",
        "b_h = tl.load(p_h, boundary_check=(0, 1)).to(b_q.dtype)",
    )


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
