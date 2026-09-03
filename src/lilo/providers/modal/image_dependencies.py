CORE_PACKAGES = (
    "fastapi>=0.141.1",
    "httpx>=0.28.1",
    "modal>=1.5.3",
    "protobuf>=5.29",
    "pydantic>=2.13.4",
    "uvicorn>=0.52.0",
    "xxhash>=3.8.1",
    "zstandard>=0.25.0",
)
STITCH_PACKAGE = (
    "stitch @ git+https://github.com/modal-projects/stitch.git"
    "@375a9396a7b05770dc4ed9cc5fe34fc4d5a472d5"
)

TINKER_PACKAGE = "tinker>=0.24,<0.25"
TINKER_CLIENT_PACKAGES = ("huggingface-hub", TINKER_PACKAGE)
MEGATRON_RUNTIME_PACKAGES = (
    *TINKER_CLIENT_PACKAGES,
    "numpy",
    "opentelemetry-api==1.43.0",
    "opentelemetry-exporter-prometheus==0.64b0",
    "opentelemetry-sdk==1.43.0",
    "safetensors",
)
MEGATRON_RUNTIME_CHECK = (
    'python -c "import importlib.metadata as m; import tinker; '
    "assert m.version('opentelemetry-api') == '1.43.0'; "
    "assert m.version('opentelemetry-sdk') == '1.43.0'; "
    "assert m.version('opentelemetry-exporter-prometheus') == '0.64b0'\""
)
