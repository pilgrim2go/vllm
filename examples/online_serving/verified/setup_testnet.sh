#!/bin/bash
set -e

# === 1. System Preparation ===
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms linux-headers-$(uname -r) \
    cmake ninja-build gcc g++ git wget curl

# === 2. Verify NVIDIA Driver ===
echo "Checking NVIDIA driver..."
nvidia-smi || echo "NVIDIA driver not found! Please use an AWS GPU AMI with drivers preinstalled."
sudo apt install -y nvidia-driver-535
# sudo reboot
# === 3. CUDA Dev Toolkit ===
# Remove duplicate repo entries if they exist
if [ -f /etc/apt/sources.list.d/cuda.list ]; then
  sudo rm /etc/apt/sources.list.d/cuda.list
fi

# Add NVIDIA CUDA repo keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit (runtime + dev)
sudo apt install -y nvidia-cuda-toolkit
sudo apt install -y cuda-compiler-12-2 cuda-cudart-dev-12-2 cuda-libraries-dev-12-2

# Export CUDA paths (fix nvcc not found)
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify nvcc
nvcc --version || true

# === 4. Python Environment (use default python3) ===
sudo apt install -y python3 python3-pip python3-venv gcc-12 g++-12
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip setuptools wheel build
# Install build dependencies required for editable install
pip install "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8.0" packaging jinja2

# === 5. PyTorch with CUDA ===
# Install PyTorch with CUDA support
# You can set PYTORCH_VERSION env var to pin a specific version
# Note: PyTorch 2.8.0+ requires CUDA 12.8 index (cu128), older versions use cu121
# The cu121 index only has versions up to 2.5.1
# Otherwise installs latest from cu121 (matching successful installation on g6.12xlarge)
if [ -n "$PYTORCH_VERSION" ]; then
  # Check if version is 2.8.0 or newer (requires cu128)
  PYTORCH_MAJOR=$(echo ${PYTORCH_VERSION} | cut -d. -f1)
  PYTORCH_MINOR=$(echo ${PYTORCH_VERSION} | cut -d. -f2)
  if [ "$PYTORCH_MAJOR" -gt 2 ] || ([ "$PYTORCH_MAJOR" -eq 2 ] && [ "$PYTORCH_MINOR" -ge 8 ]); then
    echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA 12.8 (version >= 2.8.0 requires cu128)..."
    echo "Note: CUDA 12.8 PyTorch is backward compatible with CUDA 12.2 runtime"
    pip install torch==${PYTORCH_VERSION} torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || {
      echo "Failed to install from cu128, trying main PyPI..."
      pip install torch==${PYTORCH_VERSION} torchvision==0.23.0 torchaudio==2.8.0
    }
  else
    echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA 12.1..."
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  fi
else
  echo "Installing latest PyTorch with CUDA 12.1 (available versions: up to 2.5.1)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Verify PyTorch installation and show version info
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || {
  echo "ERROR: PyTorch installation failed or verification failed"
  exit 1
}
echo "Note: If build fails with Float8_e8m0fnu errors, note the PyTorch version above"
echo "      and try installing that specific version on the failing VM."

# === 6. NumPy compatibility ===
pip install "numpy<2" --upgrade

# === 7. Clone & Build vLLM from Source ===
sudo mkdir -p /data/projects
sudo chown -R ubuntu: /data/
cd /data/projects

# Idempotent git clone/checkout - ensure we have verified6 branch
if [ -d "vllm" ]; then
  echo "vLLM directory already exists, checking branch..."
  cd vllm
  # Check if it's a git repository
  if [ -d ".git" ]; then
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
    echo "Current branch: ${CURRENT_BRANCH:-unknown}"
    # Fetch latest changes
    git fetch origin verified6 || true
    # Checkout verified6 branch (creates local tracking branch if needed)
    git checkout verified6 2>/dev/null || git checkout -b verified6 origin/verified6
    # Ensure we're on the right branch
    git checkout verified6
    # Pull latest changes
    git pull origin verified6 || true
  else
    echo "WARNING: vllm directory exists but is not a git repository. Removing and re-cloning..."
    cd ..
    rm -rf vllm
    git clone https://github.com/killerstorm/vllm
    cd vllm
    git checkout verified6
  fi
else
  echo "Cloning vLLM repository..."
  git clone https://github.com/killerstorm/vllm
  cd vllm
  git checkout verified6
fi

# Verify we're on verified6 branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "verified6" ]; then
  echo "ERROR: Failed to checkout verified6 branch. Current branch: $CURRENT_BRANCH"
  exit 1
fi
echo "Confirmed: On verified6 branch"

# Editable install (forces source build)
# Use --no-build-isolation since we've installed build deps manually
# Note: If you encounter Float8_e8m0fnu compilation errors, this may indicate
# a PyTorch version mismatch. Ensure PyTorch 2.8.0 is installed correctly.
echo "Building vLLM from source (this may take a while)..."
CUDAHOSTCXX=/usr/bin/g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
  pip install --no-build-isolation --no-cache-dir --editable . || {
  echo ""
  echo "ERROR: vLLM build failed!"
  echo "If you see 'Float8_e8m0fnu' errors, try:"
  echo "  1. Verify PyTorch version: python -c 'import torch; print(torch.__version__)'"
  echo "  2. Reinstall PyTorch 2.8.0: pip install --force-reinstall torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121"
  echo "  3. Check if you need PyTorch nightly build for Float8_e8m0fnu support"
  exit 1
}

# === 8. Verify Installation ===
python -c "import vllm; print('vLLM version:', vllm.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# === 9. Check GPU Memory and CUDA Info ===
echo "=== GPU Memory Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
echo "=== CUDA Driver Version ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "=== Checking for processes using GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "No processes found"

# === 10. Run vLLM Server ===
# Using V1 engine with float32 for accurate verification and better precision
# Note: float32 uses ~2x more GPU memory than bfloat16, so gpu-memory-utilization is set to 0.70

echo "Starting vLLM server with V1 engine and float32..."


VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
#   --tensor-parallel-size 4 \ # TP=1 for now
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --dtype float32 \
  --seed 42 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --generation-config vllm