#!/bin/bash
# Simple build script for Dockerfile.u22-cuda-12
# Builds vLLM for Ubuntu 22.04 with CUDA 12.2

set -e

# Default values
CUDA_VERSION=${CUDA_VERSION:-12.2.2}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
TAG=${TAG:-vllm/vllm-openai:u22-cuda12}
TARGET=${TARGET:-runtime}
TORCH_CUDA_ARCH=${TORCH_CUDA_ARCH:-8.9}
MAX_JOBS=${MAX_JOBS:-2}

echo "Building vLLM Docker image for Ubuntu 22.04 with CUDA 12.2"
echo "  CUDA Version: ${CUDA_VERSION}"
echo "  Python Version: ${PYTHON_VERSION}"
echo "  Tag: ${TAG}"
echo "  Target: ${TARGET}"
echo "  CUDA Arch: ${TORCH_CUDA_ARCH}"
echo "  Max Jobs: ${MAX_JOBS}"
echo ""

DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA_VERSION=${CUDA_VERSION} \
  --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
  --build-arg BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 \
  --build-arg FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 \
  --build-arg torch_cuda_arch_list=${TORCH_CUDA_ARCH} \
  --build-arg max_jobs=${MAX_JOBS} \
  --target ${TARGET} \
  -t ${TAG} \
  -f docker/Dockerfile.u22-cuda-12 \
  .

echo ""
echo "Build complete! Image tagged as: ${TAG}"
echo ""
echo "To run the server:"
echo "  docker run --gpus all -p 8000:8000 ${TAG} Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000"

