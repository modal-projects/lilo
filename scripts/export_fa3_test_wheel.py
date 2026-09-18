"""Build and export the pinned patched FA3 installation for Miles."""

from pathlib import Path
import hashlib
import json
import modal
from run_fa3_deterministic import image, REVISION

app = modal.App("lilo-fa3-export-test-wheel")


@app.function(image=image, cpu=2, timeout=180)
def export():
    import base64
    import csv
    import hashlib
    import importlib.metadata
    import io
    import zipfile

    distribution = importlib.metadata.distribution("flash_attn_3")
    output = io.BytesIO()
    records = []
    binary = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in distribution.files:
            name = str(relative)
            if "__pycache__" in name or name.endswith(
                ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")
            ):
                continue
            if name.startswith("../") or name.startswith("/"):
                raise RuntimeError(f"Unexpected installed path: {name}")
            data = distribution.locate_file(relative).read_bytes()
            digest = hashlib.sha256(data).digest()
            archive.writestr(name, data)
            records.append(
                (
                    name,
                    "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="),
                    len(data),
                )
            )
            if name.endswith(".so"):
                binary[name] = digest.hex()
        record_path = f"flash_attn_3-{distribution.version}.dist-info/RECORD"
        rows = io.StringIO()
        writer = csv.writer(rows, lineterminator="\n")
        writer.writerows(records + [(record_path, "", "")])
        archive.writestr(record_path, rows.getvalue())
    return {
        "name": f"flash_attn_3-{distribution.version}-cp310-abi3-linux_x86_64.whl",
        "data": output.getvalue(),
        "binary_sha256": binary,
    }


if __name__ == "__main__":
    destination = (
        Path(__file__).resolve().parent / "results/fa3_deterministic/artifacts"
    )
    destination.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), app.run():
        result = export.remote()
    data = result.pop("data")
    (destination / result["name"]).write_bytes(data)
    result.update(revision=REVISION, wheel_sha256=hashlib.sha256(data).hexdigest())
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
