#!/bin/bash
# One-liner Docker build commands for vLLM
# CUDA 12.4.0, Python 3.12, Ubuntu 22.04
# NOTE: Using CUDA 12.4.0 for better stability and compatibility
# 
# IMPORTANT: This script skips wheel size check (RUN_WHEEL_CHECK=false)
#            Custom source code may produce wheels larger than 500MB limit
# 
# Adjust MAX_JOBS based on your CPU cores (default: 32, max recommended: 32)
# Adjust TORCH_CUDA_ARCH_LIST based on your GPU:
#   - 8.9 for L4, A10, RTX 3090/4090
#   - 8.0 for A100
#   - 9.0 for H100
#   - 7.5 for T4, V100

MAX_JOBS=${MAX_JOBS:-32}
NVCC_THREADS=${NVCC_THREADS:-8}
TORCH_CUDA_ARCH=${TORCH_CUDA_ARCH:-8.9}
TAG=${TAG:-vllm/vllm-openai:local}
TARGET=${TARGET:-vllm-openai}

# Option 1: Using docker buildx (builds from source - required for custom source)
echo "Building with docker buildx (compiling CUDA kernels from source)..."
echo "Note: Skipping wheel size check (custom source may produce larger wheels)"
DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 \
  --build-arg CUDA_VERSION=12.2.2 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.2.2-base-ubuntu22.04 \
  --build-arg max_jobs=${MAX_JOBS} \
  --build-arg nvcc_threads=${NVCC_THREADS} \
  --build-arg torch_cuda_arch_list=${TORCH_CUDA_ARCH} \
  --build-arg RUN_WHEEL_CHECK=false \
  --target ${TARGET} \
  --load \
  -t ${TAG} \
  -f docker/Dockerfile .

# Option 1b: Using docker buildx WITH precompiled (FASTEST - only if NO custom source)
# Uncomment ONLY if you have NO custom source modifications in csrc/:
# DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 \
#   --build-arg CUDA_VERSION=12.4.0 \
#   --build-arg PYTHON_VERSION=3.12 \
#   --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
#   --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04 \
#   --build-arg max_jobs=${MAX_JOBS} \
#   --build-arg nvcc_threads=${NVCC_THREADS} \
#   --build-arg torch_cuda_arch_list=${TORCH_CUDA_ARCH} \
#   --build-arg VLLM_USE_PRECOMPILED=1 \
#   --build-arg RUN_WHEEL_CHECK=false \
#   --target ${TARGET} \
#   --load \
#   -t ${TAG} \
#   -f docker/Dockerfile .

# Option 2: Using regular docker build (if buildx not available)
# Uncomment and use this if buildx is not available:
#
# DOCKER_BUILDKIT=1 docker build \
#   --build-arg CUDA_VERSION=12.4.0 \
#   --build-arg PYTHON_VERSION=3.12 \
#   --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 \
#   --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04 \
#   --build-arg max_jobs=${MAX_JOBS} \
#   --build-arg nvcc_threads=${NVCC_THREADS} \
#   --build-arg torch_cuda_arch_list=${TORCH_CUDA_ARCH} \
#   --build-arg RUN_WHEEL_CHECK=false \
#   --target ${TARGET} \
#   -t ${TAG} \
#   -f docker/Dockerfile .

# Option 3: True one-liner for CUSTOM SOURCE (builds from source, skips wheel check)
# DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 --build-arg CUDA_VERSION=12.2.0 --build-arg PYTHON_VERSION=3.12 --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04 --build-arg max_jobs=32 --build-arg nvcc_threads=8 --build-arg torch_cuda_arch_list=8.9 --build-arg RUN_WHEEL_CHECK=false --target vllm-openai --load -t vllm/vllm-openai:local -f docker/Dockerfile .

# Option 3b: True one-liner WITH precompiled (ONLY if NO custom source)
# DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 --build-arg CUDA_VERSION=12.2.0 --build-arg PYTHON_VERSION=3.12 --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04 --build-arg max_jobs=32 --build-arg nvcc_threads=8 --build-arg torch_cuda_arch_list=8.9 --build-arg VLLM_USE_PRECOMPILED=1 --build-arg RUN_WHEEL_CHECK=false --target vllm-openai --load -t vllm/vllm-openai:local -f docker/Dockerfile .

