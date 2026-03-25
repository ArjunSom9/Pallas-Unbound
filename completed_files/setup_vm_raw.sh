#!/bin/bash
# scripts/setup_vm.sh
#
# Pallas-Flash-v5e Environment Setup Script
#
# Actions:
# 1. Installs system dependencies (zsh, htop, git).
# 2. Creates a Python virtual environment (.venv).
# 3. Installs JAX[TPU], Flax, and other project dependencies.
# 4. Configures TPU environment variables (PJRT_DEVICE).

set -e  # Exit on error

echo "============================================================"
echo "   Pallas-Flash-v5e: VM Setup"
echo "============================================================"

# 1. System Dependencies
echo "[1/4] Updating system and installing utilities..."
sudo apt-get update -qq
sudo apt-get install -y -qq zsh htop git build-essential python3-venv vim

# 2. Python Virtual Environment
echo "[2/4] Configuring Python environment..."
if [ ! -d ".venv" ]; then
    echo "      Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "      Found existing .venv."
fi

# Activate venv for dependency installation
source .venv/bin/activate

# Upgrade pip to ensure smooth installation of wheels
pip install --upgrade pip -q

# 3. Install Dependencies
echo "[3/4] Installing Python dependencies..."

# Install JAX with TPU support
# Using the libtpu_releases.html to ensure compatibility with v5e
echo "      Installing JAX[TPU]..."
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install Flax and other scientific stack libraries
# Including libtpu-nightly as specified in the roadmap
echo "      Installing Flax, Optax, Einops, Matplotlib, libtpu-nightly..."
pip install flax optax einops matplotlib pandas tqdm libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 4. Environment Variables
echo "[4/4] Configuring Environment Variables..."

# Helper to add to shell config
add_to_shell() {
    local rc_file="$1"
    if [ -f "$rc_file" ] && ! grep -q "PJRT_DEVICE=TPU" "$rc_file"; then
        echo "" >> "$rc_file"
        echo "# Pallas-Flash TPU Config" >> "$rc_file"
        echo "export PJRT_DEVICE=TPU" >> "$rc_file"
        echo "      Added PJRT_DEVICE to $rc_file"
    fi
}

add_to_shell "$HOME/.bashrc"
add_to_shell "$HOME/.zshrc"

echo "============================================================"
echo "   Setup Complete!"
echo "   To start working, run: source .venv/bin/activate"
echo "============================================================"