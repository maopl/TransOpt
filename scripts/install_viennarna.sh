#!/bin/bash

# install_viennarna.sh
# Script to install ViennaRNA on Debian/Ubuntu or macOS by compiling from source if necessary

# Set ViennaRNA version and URL as variables
VIENNA_VERSION="2.7.0"
VIENNA_URL="https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_7_x/ViennaRNA-${VIENNA_VERSION}.tar.gz"


# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if ViennaRNA is already installed
is_viennarna_installed() {
    if command_exists RNAfold; then
        echo "ViennaRNA is already installed."
        return 0
    else
        return 1
    fi
}

# Install dependencies for building from source
install_dependencies() {
    echo "Installing build dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential autoconf automake libtool pkg-config curl
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
}

# Function to install ViennaRNA from source on Debian/Ubuntu
install_on_ubuntu_from_source() {
    echo "Detected Debian/Ubuntu system. Installing ViennaRNA from source..."

    # Check if ViennaRNA is already installed
    if is_viennarna_installed; then
        echo "Skipping ViennaRNA installation."
        return 0
    fi

    install_dependencies

    CPU_CORES=$(nproc || echo 1)

    # Download and compile ViennaRNA in a temporary directory
    TEMP_DIR=$(mktemp -d)
    echo "Created temporary directory at $TEMP_DIR"

    cd "$TEMP_DIR"
    echo "Downloading ViennaRNA version ${VIENNA_VERSION}..."
    curl -LO "${VIENNA_URL}" || {
        echo "Failed to download ViennaRNA. Exiting."
        exit 1
    }

    echo "Extracting ViennaRNA source code..."
    tar -xzf "ViennaRNA-${VIENNA_VERSION}.tar.gz"
    cd "ViennaRNA-${VIENNA_VERSION}"

    echo "Configuring the build..."
    ./configure

    echo "Building ViennaRNA..."
    make -j"$CPU_CORES"

    echo "Installing ViennaRNA..."
    sudo make -j"$CPU_CORES" install || {
        echo "Failed to install ViennaRNA. Exiting."
        exit 1
    }
    
    echo "ViennaRNA version ${VIENNA_VERSION} installed successfully!"

    # Cleanup
    echo "Cleaning up..."
    rm -rf "$TEMP_DIR"
}

# Install for macOS
install_on_macos() {
    echo "Detected macOS system. Installing ViennaRNA..."

    # Check if ViennaRNA is already installed
    if is_viennarna_installed; then
        echo "Skipping ViennaRNA installation."
        return 0
    fi

    if ! command_exists brew; then
        echo "Homebrew is not installed. Please install Homebrew first: https://brew.sh"
        exit 1
    fi
    brew install viennarna
    if [ $? -eq 0 ]; then
        echo "ViennaRNA installed successfully!"
    else
        echo "Failed to install ViennaRNA on macOS."
        exit 1
    fi
}

# Detect OS and install accordingly
if [ "$(uname)" == "Darwin" ]; then
    install_on_macos
elif [ -f /etc/debian_version ]; then
    install_on_ubuntu_from_source
else
    echo "Unsupported OS. This script supports only Debian/Ubuntu and macOS."
    exit 1
fi

# Install Python bindings for ViennaRNA
echo "Installing Python bindings for ViennaRNA..."
pip install ViennaRNA

if [ $? -eq 0 ]; then
    echo "Python bindings for ViennaRNA installed successfully!"
else
    echo "Failed to install Python bindings for ViennaRNA."
    exit 1
fi

echo "Installation complete."