# Base image for GPU runs with headless EGL rendering of LIBERO.
# Build from the repo root:  docker build -t ckpt-selection env/
# The image supplies the CUDA 12 user space. The experiments use their own Python 3.11 .venv,
# which scripts/setup.sh makes.
# Pin the base image by digest when building for a fixed environment.
FROM nvcr.io/nvidia/pytorch:24.10-py3

# The NVIDIA runtime mounts the EGL driver libraries only with the "graphics" capability.
# On a hosted GPU service, set this variable in the container template too: it applies at
# container creation, and exporting it inside a running container has no effect.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

# LIBERO needs libglib2.0-0 to render. scripts/setup.sh installs the EGL loader and GLES.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Point the GLVND loader at the NVIDIA EGL driver.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace
