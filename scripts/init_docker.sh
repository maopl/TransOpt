#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCKER_ROOT_DIR="$SCRIPT_DIR/../resources/docker"

build_docker_image() {
    local image_name=$1
    local docker_dir=$2
    
    if [ -f "$docker_dir/Dockerfile" ]; then
        echo "Building Docker image '$image_name'..."
        docker build -t "$image_name" "$docker_dir"
        echo "Docker image '$image_name' created successfully."
    else
        echo "Dockerfile not found in $docker_dir"
        exit 1
    fi
}

# 构建 absolut_image
build_docker_image "absolut_image" "$DOCKER_ROOT_DIR/absolut_image"
