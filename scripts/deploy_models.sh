#!/usr/bin/env bash
# The complete set of configurations available to new clients on this frontend.
# Add a YAML under deployments/ and add its path to this list.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

deployment_files=(
  deployments/qwen35-9b-lora-16k.yaml
  deployments/qwen35-9b-lora-64k.yaml
  deployments/qwen35-4b-fft-64k.yaml
)

case "${1-}" in
  "") command_args=(deploy) ;;
  --check) command_args=(config validate) ;;
  *) echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

exec lilo "${command_args[@]}" "${deployment_files[@]}"
