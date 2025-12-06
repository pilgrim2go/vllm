#!/bin/bash
# Recovery script to fix NVIDIA driver package conflicts
# Run this before continuing with setup_testnet_ubuntu22.sh

set -e

echo "=== Fixing NVIDIA Driver Package Conflicts ==="

# Step 1: Configure any partially installed packages
echo "Step 1: Configuring partially installed packages..."
sudo dpkg --configure -a || true

# Step 2: Fix broken dependencies
echo "Step 2: Fixing broken dependencies..."
sudo apt --fix-broken install -y || true

# Step 3: Find and remove ALL conflicting NVIDIA packages
echo "Step 3: Finding conflicting NVIDIA packages..."
CONFLICTING_PKGS=$(dpkg -l | grep -E "nvidia-(kernel-common|firmware|compute-utils|driver)" | awk '{print $2}' || true)

if [ -n "$CONFLICTING_PKGS" ]; then
    echo "Found conflicting packages:"
    echo "$CONFLICTING_PKGS"
    echo ""
    echo "Force removing all conflicting packages..."
    
    # Remove using dpkg with all force options
    for pkg in $CONFLICTING_PKGS; do
        echo "Force removing $pkg..."
        sudo dpkg --remove --force-depends --force-remove-reinstreq "$pkg" 2>/dev/null || true
        sudo dpkg --purge --force-depends --force-remove-reinstreq "$pkg" 2>/dev/null || true
    done
else
    echo "No conflicting packages found via dpkg"
fi

# Step 4: Remove via apt with wildcards
echo "Step 4: Removing NVIDIA packages via apt..."
sudo apt remove -y --purge \
    nvidia-kernel-common-* \
    nvidia-firmware-* \
    nvidia-compute-utils-* \
    libnvidia-extra-* \
    nvidia-driver-* \
    2>/dev/null || true

# Step 5: Remove the specific conflicting packages if they still exist
echo "Step 5: Force removing specific conflicting packages..."
for pkg in nvidia-kernel-common-535 nvidia-firmware-535-535.274.02 nvidia-compute-utils-535 libnvidia-extra-535; do
    if dpkg -l | grep -q "^.i.*$pkg"; then
        echo "Force removing $pkg..."
        sudo dpkg --remove --force-all "$pkg" 2>/dev/null || true
        sudo dpkg --purge --force-all "$pkg" 2>/dev/null || true
    fi
done

# Step 6: Clean up package cache
echo "Step 6: Cleaning up..."
sudo apt autoremove -y || true
sudo apt autoclean || true

# Step 7: Find which packages own the conflicting files and remove them
echo "Step 7: Finding packages that own conflicting files..."
# Find package owning gsp_ga10x.bin
GSP_OWNER=$(dpkg -S /lib/firmware/nvidia/535.274.02/gsp_ga10x.bin 2>/dev/null | cut -d: -f1 || echo "")
# Find package owning nvidia-powerd
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

# Step 8: Update package lists
echo "Step 8: Updating package lists..."
sudo apt update

# Step 9: Fix broken dependencies first
echo "Step 9: Fixing broken dependencies..."
sudo apt --fix-broken install -y || true

# Step 10: Install required dependencies individually with force options
echo "Step 10: Installing required dependencies with force options..."
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
        echo "Warning: Failed to install $pkg_name, trying without version pin..."
        sudo apt install -y --allow-downgrades --allow-change-held-packages \
            -o Dpkg::Options::="--force-overwrite" \
            -o Dpkg::Options::="--force-confnew" \
            -o Dpkg::Options::="--force-depends" \
            "$pkg_name" || true
    }
done

# Step 11: Install nvidia-driver-535 with force overwrite
echo "Step 11: Installing nvidia-driver-535..."
sudo apt install -y --allow-downgrades --allow-change-held-packages \
    -o Dpkg::Options::="--force-overwrite" \
    -o Dpkg::Options::="--force-confnew" \
    -o Dpkg::Options::="--force-depends" \
    nvidia-driver-535 || {
    echo "Installation failed. Trying to install all packages together..."
    sudo apt install -y --allow-downgrades --allow-change-held-packages \
        -o Dpkg::Options::="--force-overwrite" \
        -o Dpkg::Options::="--force-confnew" \
        -o Dpkg::Options::="--force-depends" \
        nvidia-kernel-common-535=535.274.02-0ubuntu1 \
        libnvidia-extra-535=535.274.02-0ubuntu1 \
        nvidia-compute-utils-535=535.274.02-0ubuntu1 \
        nvidia-driver-535
}

echo ""
echo "=== Recovery Complete ==="
echo "If installation succeeded, you may need to reboot:"
echo "  sudo reboot"
echo ""
echo "After reboot, continue with:"
echo "  ./setup_testnet_ubuntu22.sh"

