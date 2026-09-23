#!/usr/bin/env bash
set -e

# Run from the repository root so the Python config paths below work from any directory.
cd "$(dirname "$0")/.."

# Add a model by creating its Python config in deployments/ and adding it here.
# Keep every configuration that should be available to new clients in this list.
deployment_files=(
  deployments/qwen35_9b_lora_16k.py
  deployments/qwen35_9b_lora_64k.py
  deployments/qwen35_4b_fft_64k.py
)

lilo deploy "${deployment_files[@]}"
