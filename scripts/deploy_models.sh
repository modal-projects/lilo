#!/usr/bin/env bash
set -e

# Run from the repository root so the YAML paths below work from any directory.
cd "$(dirname "$0")/.."

# Add a model by creating its YAML in deployments/ and adding it here.
# Keep every configuration that should be available to new clients in this list.
deployment_files=(
  deployments/qwen35-9b-lora-16k.yaml
  deployments/qwen35-9b-lora-64k.yaml
  deployments/qwen35-4b-fft-64k.yaml
)

lilo deploy "${deployment_files[@]}"
