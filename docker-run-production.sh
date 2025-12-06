#!/bin/bash
# Docker run command optimized for production (matching host behavior)
# Removed debugging flags that affect numerical precision
#
# Models are cached in ~/.cache/huggingface on the host and mounted to the container
# This prevents re-downloading models when the container restarts

# Default model (can be overridden via MODEL environment variable)
MODEL=${MODEL:-qwen/qwen2.5-1.5b-instruct}

# HuggingFace cache directory (defaults to ~/.cache/huggingface)
HF_CACHE_DIR=${HF_CACHE_DIR:-${HOME}/.cache/huggingface}

# Create cache directory if it doesn't exist
mkdir -p "${HF_CACHE_DIR}"

docker run -d --gpus all -p 8000:8000 --ipc=host --name vllm-test \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  vllm/vllm-openai:local \
  ${MODEL} \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype float32 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --generation-config vllm

echo "Container started with production settings (CUDA graphs enabled, no debugging flags)"
echo "Model: ${MODEL}"
echo "HuggingFace cache mounted from: ${HF_CACHE_DIR}"
echo "This should match host behavior and pass verification tests"
echo "Check logs: docker logs -f vllm-test"

