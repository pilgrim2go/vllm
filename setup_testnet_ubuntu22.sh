#!/bin/bash
set -e

# Setup script for Ubuntu 22.04 (Jammy) with CUDA 12.2
# This script configures the environment for vLLM on Ubuntu 22.04 with CUDA 12.2

# Verify Ubuntu version and get codename
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ] || [ "$VERSION_ID" != "22.04" ]; then
        echo "WARNING: This script is designed for Ubuntu 22.04. Detected: $ID $VERSION_ID"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Confirmed: Ubuntu 22.04 detected"
    fi
    # Get codename for CMake repository (defaults to jammy for Ubuntu 22.04)
    UBUNTU_CODENAME=${VERSION_CODENAME:-jammy}
else
    # Fallback if /etc/os-release doesn't exist
    UBUNTU_CODENAME=jammy
fi

# === 1. System Preparation ===
echo "=== Step 1: System Preparation ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms linux-headers-$(uname -r) \
    ninja-build gcc g++ git wget curl

# Install CMake 3.26+ (required for vLLM build)
# Ubuntu 22.04 default CMake is 3.22.1, which is too old
echo "Installing CMake 3.26+ from Kitware repository..."
NEED_CMAKE_UPGRADE=false

if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version 2>/dev/null | head -n1 | cut -d' ' -f3)
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)
    
    if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 26 ]); then
        echo "Current CMake version: $CMAKE_VERSION (need 3.26+)"
        NEED_CMAKE_UPGRADE=true
    else
        echo "CMake version $CMAKE_VERSION is sufficient (>= 3.26)"
    fi
else
    echo "CMake not found, will install from Kitware repository"
    NEED_CMAKE_UPGRADE=true
fi

if [ "$NEED_CMAKE_UPGRADE" = true ]; then
    echo "Installing CMake from Kitware repository..."
    # Remove old cmake if installed
    sudo apt remove -y cmake cmake-data 2>/dev/null || true
    
    # Add Kitware's APT repository for CMake
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
        gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    echo "deb https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" | \
        sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
    sudo apt update
    sudo apt install -y cmake
fi

# Verify CMake version
echo "Verifying CMake installation..."
cmake --version

# === 2. Verify NVIDIA Driver ===
echo "=== Step 2: NVIDIA Driver Setup ==="
echo "Checking NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "NVIDIA driver not found or not working. Attempting to install/repair..."
    
    # Fix any broken package states first
    echo "Fixing broken package states..."
    sudo dpkg --configure -a || true
    sudo apt --fix-broken install -y || true
    
    # Remove ALL conflicting NVIDIA packages from Ubuntu repos
    # These packages conflict with NVIDIA repo versions
    echo "Removing conflicting NVIDIA packages from Ubuntu repositories..."
    sudo apt remove -y --purge \
        nvidia-kernel-common-* \
        nvidia-firmware-* \
        nvidia-compute-utils-* \
        libnvidia-extra-* \
        2>/dev/null || true
    
    # Force remove packages that might be stuck using dpkg
    echo "Force removing any stuck NVIDIA packages..."
    for pkg in nvidia-kernel-common-535 nvidia-firmware-535-535.274.02 nvidia-compute-utils-535 libnvidia-extra-535; do
        if dpkg -l | grep -q "^.i.*$pkg"; then
            echo "Force removing $pkg..."
            sudo dpkg --remove --force-all "$pkg" 2>/dev/null || true
            sudo dpkg --purge --force-all "$pkg" 2>/dev/null || true
        fi
    done
    
    # Find and remove packages that own conflicting files
    echo "Finding packages that own conflicting files..."
    GSP_OWNER=$(dpkg -S /lib/firmware/nvidia/535.274.02/gsp_ga10x.bin 2>/dev/null | cut -d: -f1 || echo "")
    POWERD_OWNER=$(dpkg -S /usr/bin/nvidia-powerd 2>/dev/null | cut -d: -f1 || echo "")
    
    if [ -n "$GSP_OWNER" ]; then
        echo "Removing package that owns gsp_ga10x.bin: $GSP_OWNER"
        sudo dpkg --remove --force-all "$GSP_OWNER" 2>/dev/null || true
        sudo dpkg --purge --force-all "$GSP_OWNER" 2>/dev/null || true
    fi
    
    if [ -n "$POWERD_OWNER" ] && [ "$POWERD_OWNER" != "$GSP_OWNER" ]; then
        echo "Removing package that owns nvidia-powerd: $POWERD_OWNER"
        sudo dpkg --remove --force-all "$POWERD_OWNER" 2>/dev/null || true
        sudo dpkg --purge --force-all "$POWERD_OWNER" 2>/dev/null || true
    fi
    
    # Clean up and update
    sudo apt autoremove -y || true
    sudo apt autoclean || true
    sudo apt update
    
    # Fix broken dependencies
    echo "Fixing broken dependencies..."
    sudo apt --fix-broken install -y || true
    
    # Install required dependencies first, then nvidia-driver-535
    echo "Installing NVIDIA driver and dependencies..."
    sudo apt install -y --allow-downgrades --allow-change-held-packages \
        -o Dpkg::Options::="--force-overwrite" \
        -o Dpkg::Options::="--force-confnew" \
        -o Dpkg::Options::="--force-depends" \
        nvidia-kernel-common-535=535.274.02-0ubuntu1 \
        libnvidia-extra-535=535.274.02-0ubuntu1 \
        nvidia-compute-utils-535=535.274.02-0ubuntu1 \
        nvidia-driver-535 || {
        echo "WARNING: Installation with version pins failed, trying without version pins..."
        sudo apt install -y --allow-downgrades --allow-change-held-packages \
            -o Dpkg::Options::="--force-overwrite" \
            -o Dpkg::Options::="--force-confnew" \
            -o Dpkg::Options::="--force-depends" \
            nvidia-kernel-common-535 \
            libnvidia-extra-535 \
            nvidia-compute-utils-535 \
            nvidia-driver-535 || {
            echo "ERROR: Failed to install NVIDIA driver."
            echo "You may need to manually resolve package conflicts or use an AWS GPU AMI with drivers preinstalled."
            echo "To manually fix, try running: ./fix_nvidia_driver_conflict.sh"
            exit 1
        }
    }
    
    echo "WARNING: NVIDIA driver installation completed."
    echo "You may need to reboot for the driver to work properly."
    echo "Run 'sudo reboot' and then continue with this script after reboot."
    # Uncomment the next line if you want to auto-reboot (not recommended in automated scripts)
    # sudo reboot
else
    echo "NVIDIA driver found and working:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "Driver version is sufficient, skipping installation."
fi

# === 3. CUDA 12.2 Toolkit Installation ===
echo "=== Step 3: CUDA 12.2 Toolkit Installation ==="
# Remove duplicate repo entries if they exist
if [ -f /etc/apt/sources.list.d/cuda.list ]; then
  sudo rm /etc/apt/sources.list.d/cuda.list
fi

# Add NVIDIA CUDA 12.2 repo keyring for Ubuntu 22.04
echo "Adding CUDA 12.2 repository for Ubuntu 22.04..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12.2 toolkit (runtime + dev)
echo "Installing CUDA 12.2 toolkit..."
sudo apt install -y cuda-toolkit-12-2
sudo apt install -y cuda-compiler-12-2 cuda-cudart-dev-12-2 cuda-libraries-dev-12-2

# Export CUDA 12.2 paths
echo "Setting up CUDA 12.2 environment variables..."
if ! grep -q "CUDA-12.2" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# CUDA 12.2 paths' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Verify nvcc
echo "Verifying CUDA installation..."
nvcc --version || {
    echo "WARNING: nvcc not found. CUDA may not be properly installed."
}

# === 4. Python Environment ===
echo "=== Step 4: Python Environment Setup ==="
sudo apt install -y python3 python3-pip python3-venv gcc-12 g++-12
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install --upgrade pip setuptools wheel build
# Install build dependencies required for editable install
pip install "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8.0" packaging jinja2

# === 5. PyTorch Installation ===
echo "=== Step 5: PyTorch Installation ==="
# Install PyTorch 2.8.0 (matching Dockerfile requirements/cuda.txt)
# PyTorch 2.8.0 requires cu128 index but is backward compatible with CUDA 12.2 runtime
# This version includes Float8_e8m0fnu support required for vLLM build
# You can set PYTORCH_VERSION env var to override, but 2.8.0 is recommended
if [ -n "$PYTORCH_VERSION" ]; then
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
    echo "WARNING: PyTorch ${PYTORCH_VERSION} may not have Float8_e8m0fnu support."
    echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA 12.1..."
    pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  fi
else
  echo "Installing PyTorch 2.8.0 with CUDA 12.8 (matching Dockerfile, compatible with CUDA 12.2 runtime)..."
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || {
    echo "Failed to install from cu128, trying main PyPI..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  }
fi

# Verify PyTorch installation and show version info
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || {
  echo "ERROR: PyTorch installation failed or verification failed"
  exit 1
}

# === 6. NumPy compatibility ===
echo "=== Step 6: NumPy Compatibility ==="
pip install "numpy<2" --upgrade

# === 7. Clone & Build vLLM from Source ===
echo "=== Step 7: vLLM Source Build ==="
sudo mkdir -p /data/projects
sudo chown -R $USER: /data/
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
echo "Building vLLM from source (this may take a while)..."
CUDAHOSTCXX=/usr/bin/g++-12 CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
  pip install --no-build-isolation --no-cache-dir --editable . || {
  echo ""
  echo "ERROR: vLLM build failed!"
  echo "Troubleshooting steps:"
  echo "  1. Verify CMake version (need >= 3.26): cmake --version"
  echo "  2. Verify CUDA 12.2 installation: nvcc --version"
  echo "  3. Verify PyTorch version (need 2.8.0+ for Float8_e8m0fnu): python -c 'import torch; print(torch.__version__)'"
  echo "  4. If you see 'Float8_e8m0fnu' errors, reinstall PyTorch 2.8.0:"
  echo "     pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
  echo "  5. Check CUDA paths: echo \$PATH and echo \$LD_LIBRARY_PATH"
  exit 1
}

# === 8. Verify Installation ===
echo "=== Step 8: Installation Verification ==="
python -c "import vllm; print('vLLM version:', vllm.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# === 9. Check GPU Memory and CUDA Info ===
echo "=== Step 9: GPU and CUDA Information ==="
echo "=== GPU Memory Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
echo "=== CUDA Driver Version ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "=== CUDA Runtime Version ==="
nvcc --version | grep "release" || echo "nvcc version check failed"
echo "=== Checking for processes using GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "No processes found"

# === 10. Run vLLM Server ===
# Using V1 engine with float32 for accurate verification and better precision
# Note: float32 uses ~2x more GPU memory than bfloat16, so gpu-memory-utilization is set to 0.70

echo "=== Step 10: Starting vLLM Server ==="
echo "Starting vLLM server with V1 engine and float32..."
echo "Configuration:"
echo "  - CUDA 12.2"
echo "  - Ubuntu 22.04"
echo "  - V1 engine"
echo "  - float32 dtype"
echo ""

VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --dtype float32 \
  --seed 42 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --generation-config vllm

