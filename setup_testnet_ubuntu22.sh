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
# Update package lists (may show broken dependencies warnings, but continue)
sudo apt update || {
    echo "WARNING: apt update showed errors. This may be due to broken dependencies."
    echo "Continuing - will be fixed in NVIDIA driver setup step..."
}
# Try to upgrade, but don't fail if there are broken dependencies
sudo apt upgrade -y || {
    echo "WARNING: apt upgrade failed. This may be due to broken dependencies."
    echo "Continuing - will be fixed in NVIDIA driver setup step..."
}
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

# Function to check if NVIDIA packages are from wrong repository (Ubuntu repo vs NVIDIA repo)
check_nvidia_package_source() {
    # Check if any NVIDIA packages are from Ubuntu repos (contain 0ubuntu0.22.04.1)
    # We want packages from NVIDIA repo (contain 0ubuntu1)
    if dpkg -l 2>/dev/null | grep -E "^.i.*nvidia-" 2>/dev/null | grep -q "0ubuntu0.22.04.1" 2>/dev/null; then
        return 0  # Found Ubuntu repo packages
    fi
    return 1  # No Ubuntu repo packages found
}

# Function to check for broken dependencies
check_broken_dependencies() {
    # Try to check for broken packages - if apt-get check fails or shows issues, we have broken deps
    CHECK_OUTPUT=$(sudo apt-get check 2>&1)
    if [ $? -ne 0 ] || echo "$CHECK_OUTPUT" | grep -qE "Unmet dependencies|Broken packages|You might want to run"; then
        return 0  # Found broken dependencies
    fi
    # Also check dpkg for any packages in broken state (r = reinst-required, p = purged but config remains)
    if dpkg -l 2>/dev/null | grep -qE "^.[rp].*"; then
        return 0  # Found broken/reinst-required packages
    fi
    return 1  # No broken dependencies
}

# Function to set up apt pinning to prefer NVIDIA repository
setup_nvidia_apt_pinning() {
    echo "Setting up apt pinning to prefer NVIDIA repository..."
    # Create apt preferences to prefer NVIDIA CUDA repository and deprioritize Ubuntu repo
    sudo mkdir -p /etc/apt/preferences.d
    cat <<EOF | sudo tee /etc/apt/preferences.d/nvidia-cuda-pin-600 > /dev/null
# Prefer NVIDIA repository for all packages
Package: *
Pin: release o=.*NVIDIA.*
Pin-Priority: 600

# Prefer NVIDIA repository for NVIDIA packages
Package: nvidia-*
Pin: release o=.*NVIDIA.*
Pin-Priority: 600

Package: libnvidia-*
Pin: release o=.*NVIDIA.*
Pin-Priority: 600

# Deprioritize Ubuntu repository NVIDIA packages (version 0ubuntu0.22.04.1)
Package: nvidia-kernel-common-535
Pin: version 535.274.02-0ubuntu0.22.04.1
Pin-Priority: 1

Package: libnvidia-extra-535
Pin: version 535.274.02-0ubuntu0.22.04.1
Pin-Priority: 1

Package: nvidia-compute-utils-535
Pin: version 535.274.02-0ubuntu0.22.04.1
Pin-Priority: 1

Package: nvidia-driver-535
Pin: version 535.274.02-0ubuntu0.22.04.1
Pin-Priority: 1

Package: nvidia-dkms-535
Pin: version 535.274.02-0ubuntu0.22.04.1
Pin-Priority: 1
EOF
    echo "Apt pinning configured to prefer NVIDIA repository and deprioritize Ubuntu repo versions"
    # Update package lists after setting up pinning
    sudo apt update || {
        echo "WARNING: apt update failed after setting pinning. Will continue..."
    }
}

# Function to hold Ubuntu repo versions to prevent apt from installing them
hold_ubuntu_repo_nvidia_packages() {
    echo "Checking for Ubuntu repository NVIDIA packages..."
    # This is now handled by apt preferences, but we can still check
    for pkg in nvidia-kernel-common-535 libnvidia-extra-535 nvidia-compute-utils-535 nvidia-driver-535 nvidia-dkms-535; do
        if apt-cache policy "$pkg" 2>/dev/null | grep -q "535.274.02-0ubuntu0.22.04.1"; then
            echo "Found Ubuntu repo version of $pkg (will be deprioritized by apt preferences)"
        fi
    done
}

# Function to comprehensively remove conflicting NVIDIA packages
remove_conflicting_nvidia_packages() {
    echo "Removing conflicting NVIDIA packages from Ubuntu repositories..."
    
    # Set up apt pinning first
    setup_nvidia_apt_pinning
    
    # Hold Ubuntu repo versions to prevent apt from selecting them
    hold_ubuntu_repo_nvidia_packages
    
    # Fix any broken package states first
    echo "Fixing broken package states..."
    sudo dpkg --configure -a || true
    sudo apt --fix-broken install -y || true
    
    # Find ALL NVIDIA packages with Ubuntu repo version (0ubuntu0.22.04.1)
    echo "Finding ALL NVIDIA packages from Ubuntu repository..."
    UBUNTU_REPO_PKGS=$(dpkg -l 2>/dev/null | grep -E "^.i.*nvidia-" 2>/dev/null | grep "0ubuntu0.22.04.1" 2>/dev/null | awk '{print $2}' || true)
    
    if [ -n "$UBUNTU_REPO_PKGS" ]; then
        echo "Found Ubuntu repository packages:"
        echo "$UBUNTU_REPO_PKGS"
        echo ""
        echo "Force removing all Ubuntu repository NVIDIA packages..."
        
        # Remove using dpkg with all force options
        for pkg in $UBUNTU_REPO_PKGS; do
            echo "Force removing $pkg..."
            sudo dpkg --remove --force-depends --force-remove-reinstreq --force-all "$pkg" 2>/dev/null || true
            sudo dpkg --purge --force-depends --force-remove-reinstreq --force-all "$pkg" 2>/dev/null || true
        done
    fi
    
    # Also find and remove ALL conflicting NVIDIA packages (any version)
    echo "Finding all NVIDIA packages..."
    CONFLICTING_PKGS=$(dpkg -l 2>/dev/null | grep -E "^.i.*nvidia-" 2>/dev/null | awk '{print $2}' || true)
    
    if [ -n "$CONFLICTING_PKGS" ]; then
        echo "Found all NVIDIA packages:"
        echo "$CONFLICTING_PKGS"
        echo ""
        echo "Removing all NVIDIA packages to ensure clean state..."
        
        # Remove using dpkg with all force options
        for pkg in $CONFLICTING_PKGS; do
            echo "Force removing $pkg..."
            sudo dpkg --remove --force-depends --force-remove-reinstreq --force-all "$pkg" 2>/dev/null || true
            sudo dpkg --purge --force-depends --force-remove-reinstreq --force-all "$pkg" 2>/dev/null || true
        done
    fi
    
    # Remove via apt with wildcards (more comprehensive)
    echo "Removing NVIDIA packages via apt with wildcards..."
    sudo apt remove -y --purge \
        nvidia-* \
        libnvidia-* \
        2>/dev/null || true
    
    # Force remove specific conflicting packages if they still exist
    echo "Force removing specific conflicting packages..."
    for pkg in nvidia-kernel-common-535 nvidia-firmware-535-535.274.02 nvidia-compute-utils-535 libnvidia-extra-535 libnvidia-gl-535 nvidia-dkms-535 nvidia-driver-535; do
        if dpkg -l 2>/dev/null | grep -q "^.i.*$pkg"; then
            echo "Force removing $pkg..."
            sudo dpkg --remove --force-all --force-depends --force-remove-reinstreq "$pkg" 2>/dev/null || true
            sudo dpkg --purge --force-all --force-depends --force-remove-reinstreq "$pkg" 2>/dev/null || true
        fi
    done
    
    # Find which packages own the conflicting files and remove them
    echo "Finding packages that own conflicting files..."
    GSP_OWNER=$(dpkg -S /lib/firmware/nvidia/535.274.02/gsp_ga10x.bin 2>/dev/null | cut -d: -f1 || echo "")
    POWERD_OWNER=$(dpkg -S /usr/bin/nvidia-powerd 2>/dev/null | cut -d: -f1 || echo "")
    
    if [ -n "$GSP_OWNER" ]; then
        echo "Removing package that owns gsp_ga10x.bin: $GSP_OWNER"
        sudo dpkg --remove --force-all --force-depends --force-remove-reinstreq "$GSP_OWNER" 2>/dev/null || true
        sudo dpkg --purge --force-all --force-depends --force-remove-reinstreq "$GSP_OWNER" 2>/dev/null || true
    fi
    
    if [ -n "$POWERD_OWNER" ] && [ "$POWERD_OWNER" != "$GSP_OWNER" ]; then
        echo "Removing package that owns nvidia-powerd: $POWERD_OWNER"
        sudo dpkg --remove --force-all --force-depends --force-remove-reinstreq "$POWERD_OWNER" 2>/dev/null || true
        sudo dpkg --purge --force-all --force-depends --force-remove-reinstreq "$POWERD_OWNER" 2>/dev/null || true
    fi
    
    # Clean up and update
    sudo apt autoremove -y || true
    sudo apt autoclean || true
    # Update package lists - may fail if broken deps, but we'll fix them next
    sudo apt update || {
        echo "WARNING: apt update failed due to broken dependencies. Will fix now..."
        sudo apt --fix-broken install -y || true
        sudo apt update || true
    }
    
    # Fix broken dependencies
    echo "Fixing broken dependencies..."
    sudo apt --fix-broken install -y || true
    
    # Verify no broken dependencies remain
    if check_broken_dependencies; then
        echo "WARNING: Some broken dependencies may still exist. Attempting additional cleanup..."
        sudo apt --fix-broken install -y || true
    fi
}

# Set up apt pinning early to prefer NVIDIA repository
setup_nvidia_apt_pinning

# Hold Ubuntu repo versions early to prevent apt from selecting them
hold_ubuntu_repo_nvidia_packages

# Check for broken dependencies first (even if nvidia-smi works)
if check_broken_dependencies; then
    echo "Detected broken dependencies. Running fix_nvidia_driver_conflict.sh..."
    FIX_SCRIPT="$(dirname "$0")/fix_nvidia_driver_conflict.sh"
    if [ -f "$FIX_SCRIPT" ]; then
        echo "Running $FIX_SCRIPT to fix conflicts..."
        # Temporarily disable set -e for the fix script in case it has issues
        set +e
        bash "$FIX_SCRIPT"
        FIX_EXIT_CODE=$?
        set -e
        
        if [ $FIX_EXIT_CODE -ne 0 ]; then
            echo "WARNING: fix_nvidia_driver_conflict.sh exited with code $FIX_EXIT_CODE"
        fi
        
        # After fix script, check again
        if check_broken_dependencies; then
            echo "WARNING: Broken dependencies still exist after fix script."
            echo "Attempting additional cleanup..."
            remove_conflicting_nvidia_packages
        else
            echo "Broken dependencies resolved by fix script."
        fi
        NEED_NVIDIA_INSTALL=true
    else
        echo "WARNING: fix_nvidia_driver_conflict.sh not found at $FIX_SCRIPT"
        echo "Attempting cleanup with built-in function..."
        remove_conflicting_nvidia_packages
        NEED_NVIDIA_INSTALL=true
    fi
elif ! nvidia-smi &>/dev/null; then
    echo "NVIDIA driver not found or not working. Attempting to install/repair..."
    # Check if we have broken deps first
    if check_broken_dependencies; then
        echo "Broken dependencies detected. Running fix_nvidia_driver_conflict.sh..."
        FIX_SCRIPT="$(dirname "$0")/fix_nvidia_driver_conflict.sh"
        if [ -f "$FIX_SCRIPT" ]; then
            bash "$FIX_SCRIPT" || true
        fi
    fi
    remove_conflicting_nvidia_packages
    NEED_NVIDIA_INSTALL=true
elif check_nvidia_package_source; then
    echo "NVIDIA driver is working but packages are from Ubuntu repository."
    echo "Removing Ubuntu repo packages and installing from NVIDIA repository..."
    remove_conflicting_nvidia_packages
    NEED_NVIDIA_INSTALL=true
else
    echo "NVIDIA driver found and working:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "Driver version is sufficient, skipping installation."
    NEED_NVIDIA_INSTALL=false
fi

# Install NVIDIA driver if needed
if [ "$NEED_NVIDIA_INSTALL" = true ]; then
    # Check if driver packages are already installed (even if nvidia-smi doesn't work yet)
    if dpkg -l 2>/dev/null | grep -q "^.i.*nvidia-driver-535"; then
        if ! check_broken_dependencies; then
            echo "NVIDIA driver packages are already installed and dependencies are resolved."
            if nvidia-smi &>/dev/null; then
                echo "NVIDIA driver is working. Continuing with setup..."
                NEED_NVIDIA_INSTALL=false
            else
                echo "NVIDIA driver packages installed but nvidia-smi doesn't work yet."
                echo "This usually means a reboot is required."
                echo "After reboot, rerun this script to continue."
                read -p "Reboot now? (y/N) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "Rebooting in 5 seconds..."
                    sleep 5
                    sudo reboot
                else
                    echo "Please reboot manually and then rerun this script."
                    exit 0
                fi
            fi
        else
            echo "NVIDIA driver packages installed but broken dependencies detected."
            echo "This should not happen. Attempting to fix..."
        fi
    fi
    
    # Only proceed with installation if still needed
    if [ "$NEED_NVIDIA_INSTALL" = true ]; then
        # Verify no broken dependencies before installing
        if check_broken_dependencies; then
            echo "ERROR: Broken dependencies still exist. Cannot proceed with installation."
            echo "Please run: ./fix_nvidia_driver_conflict.sh"
            echo "Then reboot and rerun this script."
            exit 1
        fi
        
        # Check what versions apt would install
    echo "Checking available package versions..."
    echo "nvidia-kernel-common-535 versions:"
    apt-cache policy nvidia-kernel-common-535 | head -15 || true
    echo "libnvidia-extra-535 versions:"
    apt-cache policy libnvidia-extra-535 | head -15 || true
    echo "nvidia-compute-utils-535 versions:"
    apt-cache policy nvidia-compute-utils-535 | head -15 || true
    
    # Check if Ubuntu repo versions would be selected
    if apt-cache policy nvidia-kernel-common-535 | grep -q "535.274.02-0ubuntu0.22.04.1"; then
        echo "WARNING: Ubuntu repo version detected. Will force install NVIDIA repo version."
    fi
    
    # Install required dependencies individually first (like fix_nvidia_driver_conflict.sh)
    echo "Installing NVIDIA driver dependencies individually from NVIDIA repository..."
    DEPENDENCIES="nvidia-kernel-common-535=535.274.02-0ubuntu1 libnvidia-extra-535=535.274.02-0ubuntu1 nvidia-compute-utils-535=535.274.02-0ubuntu1"
    
    for dep in $DEPENDENCIES; do
        pkg_name=$(echo $dep | cut -d= -f1)
        pkg_version=$(echo $dep | cut -d= -f2)
        echo "Installing $pkg_name=$pkg_version..."
        sudo apt install -y --allow-downgrades --allow-change-held-packages \
            -o Dpkg::Options::="--force-overwrite" \
            -o Dpkg::Options::="--force-confnew" \
            -o Dpkg::Options::="--force-depends" \
            "$pkg_name=$pkg_version" || {
            echo "Warning: Failed to install $pkg_name=$pkg_version, trying without version pin..."
            sudo apt install -y --allow-downgrades --allow-change-held-packages \
                -o Dpkg::Options::="--force-overwrite" \
                -o Dpkg::Options::="--force-confnew" \
                -o Dpkg::Options::="--force-depends" \
                "$pkg_name" || {
                echo "ERROR: Failed to install $pkg_name"
                echo "To manually fix, try running: ./fix_nvidia_driver_conflict.sh"
                exit 1
            }
        }
    done
    
    # Now install nvidia-driver-535 (without recommends to avoid i386 packages)
    echo "Installing nvidia-driver-535 from NVIDIA repository..."
    sudo apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        -o Dpkg::Options::="--force-overwrite" \
        -o Dpkg::Options::="--force-confnew" \
        -o Dpkg::Options::="--force-depends" \
        nvidia-driver-535 || {
        echo "WARNING: Installation failed, trying with all packages together..."
        sudo apt install -y --allow-downgrades --allow-change-held-packages \
            -o Dpkg::Options::="--force-overwrite" \
            -o Dpkg::Options::="--force-confnew" \
            -o Dpkg::Options::="--force-depends" \
            nvidia-kernel-common-535=535.274.02-0ubuntu1 \
            libnvidia-extra-535=535.274.02-0ubuntu1 \
            nvidia-compute-utils-535=535.274.02-0ubuntu1 \
            nvidia-driver-535 || {
            echo "ERROR: Failed to install NVIDIA driver."
            echo "You may need to manually resolve package conflicts or use an AWS GPU AMI with drivers preinstalled."
            echo "To manually fix, try running: ./fix_nvidia_driver_conflict.sh"
            exit 1
        }
    }
    
    # Verify installation succeeded and no broken dependencies
    if check_broken_dependencies; then
        echo "ERROR: Broken dependencies still exist after installation."
        echo "This usually means the system needs a reboot or manual intervention."
        echo "Try running: ./fix_nvidia_driver_conflict.sh"
        echo "Then reboot and rerun this script."
        exit 1
    fi
    
    echo "NVIDIA driver installation completed."
    
    # Check if nvidia-smi works now
    if nvidia-smi &>/dev/null; then
        echo "NVIDIA driver is working. Continuing with setup..."
    else
        echo "WARNING: NVIDIA driver installed but nvidia-smi doesn't work yet."
        echo "You may need to reboot for the driver to work properly."
        echo "After reboot, rerun this script to continue."
        echo ""
        read -p "Reboot now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Rebooting in 5 seconds..."
            sleep 5
            sudo reboot
        else
            echo "Please reboot manually and then rerun this script."
            exit 0
        fi
    fi
    fi
fi

# === 3. CUDA 12.2 Toolkit Installation ===
echo "=== Step 3: CUDA 12.2 Toolkit Installation ==="

# Check if CUDA 12.2 is already installed
CUDA_INSTALLED=false
CUDA_VERSION=""
if [ -d "/usr/local/cuda-12.2" ] && command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo "")
    if [ "$CUDA_VERSION" = "12.2" ]; then
        echo "CUDA 12.2 is already installed. Skipping installation."
        CUDA_INSTALLED=true
        # Still set up environment variables if not already set
        if ! grep -q "CUDA-12.2" ~/.bashrc; then
            echo '' >> ~/.bashrc
            echo '# CUDA 12.2 paths' >> ~/.bashrc
            echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        fi
        export PATH=/usr/local/cuda-12.2/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    else
        echo "CUDA $CUDA_VERSION found, but need 12.2. Proceeding with installation..."
    fi
fi

# Only install if CUDA 12.2 is not already installed
if [ "$CUDA_INSTALLED" = false ]; then
    # Remove duplicate repo entries if they exist
    if [ -f /etc/apt/sources.list.d/cuda.list ]; then
      sudo rm /etc/apt/sources.list.d/cuda.list
    fi

    # Add NVIDIA CUDA 12.2 repo keyring for Ubuntu 22.04
    echo "Adding CUDA 12.2 repository for Ubuntu 22.04..."
    if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    fi
    sudo dpkg -i cuda-keyring_1.1-1_all.deb || true
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
fi

# Verify nvcc
echo "Verifying CUDA installation..."
nvcc --version || {
    echo "WARNING: nvcc not found. CUDA may not be properly installed."
}

# === 4. Python Environment ===
echo "=== Step 4: Python Environment Setup ==="
sudo apt install -y python3 python3-pip python3-venv gcc-12 g++-12

# Create venv only if it doesn't exist
if [ ! -d ~/vllm-env ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ~/vllm-env
else
    echo "Python virtual environment already exists. Skipping creation."
fi

source ~/vllm-env/bin/activate

# Upgrade pip and install build dependencies (idempotent - pip will skip if already installed)
echo "Installing/upgrading Python build dependencies..."
pip install --upgrade pip setuptools wheel build
# Install build dependencies required for editable install
pip install "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8.0" packaging jinja2

# === 5. PyTorch Installation ===
echo "=== Step 5: PyTorch Installation ==="

# Check if PyTorch is already installed
PYTORCH_INSTALLED=false
PYTORCH_VERSION_INSTALLED=""
if python -c "import torch" 2>/dev/null; then
    PYTORCH_INSTALLED=true
    PYTORCH_VERSION_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    echo "PyTorch is already installed: $PYTORCH_VERSION_INSTALLED"
fi

# Determine target PyTorch version
if [ -n "$PYTORCH_VERSION" ]; then
    TARGET_PYTORCH_VERSION=$PYTORCH_VERSION
else
    TARGET_PYTORCH_VERSION="2.8.0"
fi

# Check if we need to install/upgrade PyTorch
NEED_PYTORCH_INSTALL=true
if [ "$PYTORCH_INSTALLED" = true ]; then
    if [ "$PYTORCH_VERSION_INSTALLED" = "$TARGET_PYTORCH_VERSION" ]; then
        echo "PyTorch $TARGET_PYTORCH_VERSION is already installed. Skipping installation."
        NEED_PYTORCH_INSTALL=false
    else
        echo "PyTorch version mismatch: installed $PYTORCH_VERSION_INSTALLED, need $TARGET_PYTORCH_VERSION"
        echo "Reinstalling PyTorch..."
    fi
fi

# Install PyTorch if needed
# PyTorch 2.8.0 requires cu128 index but is backward compatible with CUDA 12.2 runtime
# This version includes Float8_e8m0fnu support required for vLLM build
if [ "$NEED_PYTORCH_INSTALL" = true ]; then
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
fi

# Verify PyTorch installation and show version info
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || {
    echo "ERROR: PyTorch installation failed or verification failed"
    exit 1
}

# === 6. NumPy compatibility ===
echo "=== Step 6: NumPy Compatibility ==="
# Install numpy<2 (idempotent - pip will upgrade if needed)
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

# Check if vLLM is already installed
if python -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    echo "vLLM is already installed: $VLLM_VERSION"
    echo "Skipping build. If you need to rebuild, uninstall first: pip uninstall vllm"
else
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
fi

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

