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
#
# Optional components can be skipped for faster builds:
#   - SKIP_DEEPGEMM=true      : Skip DeepGEMM (GEMM optimization)
#   - SKIP_EP_KERNELS=true    : Skip EP kernels (pplx-kernels, DeepEP)
#   - SKIP_FLASHINFER=true    : Skip FlashInfer (attention optimization)
#   - SKIP_GDRCOPY=true       : Skip gdrcopy (GPU Direct RDMA)
#   - SKIP_ALL_OPTIONAL=true  : Skip all optional components at once
#
# Build mode:
#   - Always builds from source (no precompiled wheels)

MAX_JOBS=${MAX_JOBS:-32}
NVCC_THREADS=${NVCC_THREADS:-8}
TORCH_CUDA_ARCH=${TORCH_CUDA_ARCH:-8.9}
TAG=${TAG:-vllm/vllm-openai:local}
TARGET=${TARGET:-vllm-openai}

# Optional: Skip optional components for faster builds
# Set to "true" to skip: DeepGEMM, EP kernels, FlashInfer, gdrcopy
SKIP_DEEPGEMM=${SKIP_DEEPGEMM:-true}
SKIP_EP_KERNELS=${SKIP_EP_KERNELS:-true}
SKIP_FLASHINFER=${SKIP_FLASHINFER:-true}
SKIP_GDRCOPY=${SKIP_GDRCOPY:-true}

# Convenience: Skip all optional components at once
SKIP_ALL_OPTIONAL=${SKIP_ALL_OPTIONAL:-true}
if [ "$SKIP_ALL_OPTIONAL" = "true" ]; then
    SKIP_DEEPGEMM=true
    SKIP_EP_KERNELS=true
    SKIP_FLASHINFER=true
    SKIP_GDRCOPY=true
    echo "Skipping all optional components (DeepGEMM, EP kernels, FlashInfer, gdrcopy)"
fi

# Option 1: Using docker buildx (builds from source)
echo "Building with docker buildx (compiling CUDA kernels from source)..."
echo "Note: Skipping wheel size check (custom source may produce larger wheels)"
echo "Optional components: DeepGEMM=${SKIP_DEEPGEMM}, EP kernels=${SKIP_EP_KERNELS}, FlashInfer=${SKIP_FLASHINFER}, gdrcopy=${SKIP_GDRCOPY}"
DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 \
  --build-arg CUDA_VERSION=12.2.2 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.2.2-devel-ubuntu22.04 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.2.2-base-ubuntu22.04 \
  --build-arg max_jobs=${MAX_JOBS} \
  --build-arg nvcc_threads=${NVCC_THREADS} \
  --build-arg torch_cuda_arch_list=${TORCH_CUDA_ARCH} \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg SKIP_DEEPGEMM=${SKIP_DEEPGEMM} \
  --build-arg SKIP_EP_KERNELS=${SKIP_EP_KERNELS} \
  --build-arg SKIP_FLASHINFER=${SKIP_FLASHINFER} \
  --build-arg SKIP_GDRCOPY=${SKIP_GDRCOPY} \
  --target ${TARGET} \
  --load \
  -t ${TAG} \
  -f docker/Dockerfile .


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
#   --build-arg SKIP_DEEPGEMM=${SKIP_DEEPGEMM} \
#   --build-arg SKIP_EP_KERNELS=${SKIP_EP_KERNELS} \
#   --build-arg SKIP_FLASHINFER=${SKIP_FLASHINFER} \
#   --build-arg SKIP_GDRCOPY=${SKIP_GDRCOPY} \
#   --target ${TARGET} \
#   -t ${TAG} \
#   -f docker/Dockerfile .

# Option 3: True one-liner for CUSTOM SOURCE (builds from source, skips wheel check)
# DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 --build-arg CUDA_VERSION=12.2.0 --build-arg PYTHON_VERSION=3.12 --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04 --build-arg max_jobs=32 --build-arg nvcc_threads=8 --build-arg torch_cuda_arch_list=8.9 --build-arg RUN_WHEEL_CHECK=false --build-arg SKIP_DEEPGEMM=true --build-arg SKIP_EP_KERNELS=true --build-arg SKIP_FLASHINFER=true --build-arg SKIP_GDRCOPY=true --target vllm-openai --load -t vllm/vllm-openai:local -f docker/Dockerfile .

# Option 3b: True one-liner WITH precompiled (ONLY if NO custom source)
# DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 --build-arg CUDA_VERSION=12.2.0 --build-arg PYTHON_VERSION=3.12 --build-arg BUILD_BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04 --build-arg FINAL_BASE_IMAGE=nvidia/cuda:12.4.0-base-ubuntu22.04 --build-arg max_jobs=32 --build-arg nvcc_threads=8 --build-arg torch_cuda_arch_list=8.9 --build-arg VLLM_USE_PRECOMPILED=1 --build-arg RUN_WHEEL_CHECK=false --build-arg SKIP_DEEPGEMM=true --build-arg SKIP_EP_KERNELS=true --build-arg SKIP_FLASHINFER=true --build-arg SKIP_GDRCOPY=true --target vllm-openai --load -t vllm/vllm-openai:local -f docker/Dockerfile .

