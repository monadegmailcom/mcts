# Browser Lab image for NVIDIA GPUs (RunPod): candle and libtorch backends on CUDA, an SSH server
# and the lab on port 8080. Built and pushed by .github/workflows/docker.yml; see RUNPOD.md.
#
# The image is also the development environment: it keeps the toolchain (Rust, nvcc, libtorch with
# headers) and the build in /src, so code synced into /src rebuilds on the pod in a minute or two
# (`mcts-build`) instead of a new image.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU}

# SSH host keys are generated when the container starts, not baked into the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git htop \
        openssh-server pkg-config rsync tini tmux unzip \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\n' > /etc/ssh/sshd_config.d/mcts.conf

ARG RUST_VERSION=1.99.1
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_VERSION"

# libtorch for tch 0.22, the release start_lab.sh pins, in its CUDA build (CUDA libraries included).
ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.1%2Bcu128.zip
RUN curl -fsSL -o /tmp/libtorch.zip "$LIBTORCH_URL" \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

# tch 0.22 asks for libtorch 2.9.0 exactly; 2.9.x patch releases are compatible. There is no GPU
# while building, so candle compiles its CUDA kernels to PTX for compute capability 8.0 (Ampere);
# newer GPUs compile that PTX when it is loaded. Builds on the pod keep the same value, so they
# reuse the image's compiled kernels.
ENV LIBTORCH=/opt/libtorch LIBTORCH_BYPASS_VERSION_CHECK=1 CUDA_COMPUTE_CAP=80

COPY docker/entrypoint.sh docker/mcts-lab docker/mcts-build /usr/local/bin/
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src
COPY examples examples
RUN mcts-build

# Lab state (checkpoints, experiments, Elo pools) lives in MCTS_DATA_DIR: on RunPod the volume
# mounted at /workspace, so it survives pod restarts.
ENV MCTS_DATA_DIR=/workspace/mcts MCTS_BIND_ADDR=0.0.0.0:8080
LABEL org.opencontainers.image.source=https://github.com/monadegmailcom/mcts
EXPOSE 22 8080
# tini as PID 1 reaps the processes of restarted labs.
ENTRYPOINT ["tini", "--", "entrypoint.sh"]
