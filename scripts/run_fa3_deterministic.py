"""Build and test the upstream FA3 hdim256 deterministic prototype on one H100."""

from pathlib import Path
import argparse
import json
import modal

REPO = Path(__file__).resolve().parents[1]
REVISION = "8d3a3b80d4758ebde5a867c50d24d4351443cf2b"
app = modal.App("lilo-fa3-deterministic-prototype")


def compile_fa3():
    import os
    import subprocess

    env = dict(os.environ)
    for feature in (
        "SPLIT",
        "PAGEDKV",
        "APPENDKV",
        "PACKGQA",
        "FP8",
        "CLUSTER",
        "HDIM64",
        "HDIM96",
        "HDIM128",
        "HDIM192",
        "SM80",
        "HDIMDIFF64",
        "HDIMDIFF192",
    ):
        env[f"FLASH_ATTENTION_DISABLE_{feature}"] = "TRUE"
    env.update(MAX_JOBS="4", NVCC_THREADS="2", FLASH_ATTENTION_FORCE_BUILD="TRUE")
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--force-reinstall",
            ".",
            "-v",
        ],
        cwd="/root/flash-attention/hopper",
        env=env,
        check=True,
    )


image = (
    modal.Image.from_registry("radixark/miles:v0.1.0")
    .entrypoint([])
    .run_commands(
        "git init /root/flash-attention",
        f"git -C /root/flash-attention fetch --depth 1 https://github.com/Dao-AILab/flash-attention.git {REVISION}",
        "git -C /root/flash-attention checkout --detach FETCH_HEAD",
        "git -C /root/flash-attention submodule update --init --depth 1 csrc/cutlass",
    )
    .add_local_file(
        REPO / "scripts/patches/fa3_deterministic_hdim256.patch",
        "/root/fa3.patch",
        copy=True,
    )
    .run_commands("git -C /root/flash-attention apply /root/fa3.patch")
    .run_function(compile_fa3, cpu=16, memory=65536, timeout=1800)
    .add_local_file(
        REPO / "scripts/probe_fa3_deterministic.py", "/root/probe_fa3_deterministic.py"
    )
    .add_local_file(
        REPO / "scripts/probe_fa3_upstream.py", "/root/probe_fa3_upstream.py"
    )
    .add_local_file(
        REPO / "scripts/probe_fa3_forward_backward.py",
        "/root/probe_fa3_forward_backward.py",
    )
    .add_local_file(
        REPO / "scripts/patches/fa3_deterministic_hdim256_tests.patch",
        "/root/fa3-tests.patch",
    )
)


@app.function(image=image, gpu="H100!", cpu=4, timeout=900, max_containers=1)
def probe(upstream_tests=False, forward_backward=False):
    import subprocess
    from pathlib import Path
    import json

    if upstream_tests:
        subprocess.run(
            ["git", "apply", "/root/fa3-tests.patch"],
            cwd="/root/flash-attention",
            check=True,
        )
    script = (
        "/root/probe_fa3_upstream.py"
        if upstream_tests
        else "/root/probe_fa3_deterministic.py"
    )
    if forward_backward:
        script = "/root/probe_fa3_forward_backward.py"
    proc = subprocess.run(
        ["python", "-u", script, "--output", "/root/report.json"], timeout=720
    )
    report = (
        json.loads(Path("/root/report.json").read_text())
        if Path("/root/report.json").exists()
        else {}
    )
    return {"exit_code": proc.returncode, "revision": REVISION, **report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "scripts/results/fa3_deterministic/v1"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--upstream-tests", action="store_true")
    group.add_argument("--forward-backward", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), app.run():
        result = probe.remote(args.upstream_tests, args.forward_backward)
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    raise SystemExit(result["exit_code"])
