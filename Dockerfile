# syntax=docker/dockerfile:1

# NInfer-all (iamwavecut/ninfer-all @ 2172a598, the commit the Ternary-Bonsai-2-27B
# NInfer-v3 card benchmarked), built for consumer GeForce Blackwell: sm_120a
# runs the ternary t2_g128_fp16 route on the mma.sync compatibility path that
# consumer Blackwell needs. The repo's stock Dockerfile builds with the
# default 86 and would refuse this artifact on a 5090. CMAKE_CUDA_ARCHITECTURES
# is an ARG: local builds default to 120a (5090 only); the CI build passes
# 86;89;120a so the published image also runs on RTX 3090 / 4090.
# The pinned engine's CMake guard admits a single arch only, so a multi-arch
# ARG also relaxes that guard's regex with a one-line sed - a build-metadata
# change only; kernel sources are untouched and the 120a cubin stays
# identical to a solo-120a build.

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
# Overridable build knobs; defaults reproduce the local host build (RTX 5090).
# CI passes CMAKE_CUDA_ARCHITECTURES=86;89;120a (one fat binary for
# RTX 3090/4090/5090) and smaller NINFER_BUILD_PARALLEL / ccache size to fit
# a 4-core / 16 GB runner.
ARG CMAKE_CUDA_ARCHITECTURES=120a
ARG NINFER_BUILD_PARALLEL=16
ARG NINFER_CCACHE_MAXSIZE=20G
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ccache \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
        rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Same cache-key mechanism as the upstream Dockerfile: CMake scripts, package
# versions and this build's own flags identify the configuration; removed flags
# and changed defaults must not reuse CMakeCache.txt. Lock the mutable Ninja
# tree and copy deliverables out of the transient cache mount.
# NINFER_BUILD_PARALLEL defaults to 16: the local build host has 48 cores but
# only ~40 GB usable RAM (SGLang is running), and heavy concurrent nvcc+ptxas
# frontends would risk OOM. CI (4-core / 16 GB runner) passes 4. Split-
# compile keeps ptxas off the critical path.
# The pinned engine's CMakeLists refuses any multi-arch value ("NInfer
# supports CMAKE_CUDA_ARCHITECTURES=80, 86, 89 or 120a"). For the fat-binary
# list the guard's regex is widened to admit exactly 86;89;120a before the
# hash gate, so the configuration hash covers the patched tree and solo vs
# multi builds never share a build dir. Only the guard changes - same
# sources, same NINFER_SM8X_COMPAT compatibility path, same per-arch cubins.
RUN --mount=type=cache,id=ninfer-build-120a,target=/build,sharing=locked \
    --mount=type=cache,target=/ccache \
    export CCACHE_DIR=/ccache CCACHE_MAXSIZE=${NINFER_CCACHE_MAXSIZE} \
    && if [ "${CMAKE_CUDA_ARCHITECTURES}" = "86;89;120a" ]; then \
      sed -i 's/\^(80|86|89|120a)\$"/^(80|86|89|120a|86;89;120a)$"/' CMakeLists.txt; \
    fi \
    && find . -type f \( -path ./Dockerfile -o -name CMakeLists.txt -o -name '*.cmake' \) \
        -exec sha256sum {} + > /build/configuration \
    && dpkg-query -W >> /build/configuration \
    && echo "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} NINFER_NVCC_SPLIT_COMPILE=2 PARALLEL=${NINFER_BUILD_PARALLEL} CCACHE=${NINFER_CCACHE_MAXSIZE}" >> /build/configuration \
    && LC_ALL=C sort -o /build/configuration /build/configuration \
    && build_dir="/build/$(sha256sum /build/configuration | cut -d ' ' -f 1)" \
    && rsync --recursive --links --checksum --delete /src/ /build/src/ \
    && cmake -S /build/src -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DNINFER_NVCC_SPLIT_COMPILE=2 \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build "$build_dir" --parallel ${NINFER_BUILD_PARALLEL} --target ninfer ninfer-serve \
    && mkdir -p /out \
    && cp "$build_dir/apps/ninfer" "$build_dir/apps/ninfer-serve" /out/ \
    && ccache --show-stats

FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

# The CUDA runtime image ships forward-compatibility libraries in
# /usr/local/cuda*/compat (a newer libcuda.so than the host driver). Forward
# compatibility is supported only on datacenter GPUs; on any GeForce card the
# loader picks these up and every CUDA call fails at startup with
#   cudaErrorCompatNotSupportedOnDevice: forward compatibility was attempted
#   on non supported HW
# Removing them lets the container use the host driver through ordinary CUDA
# minor-version compatibility, which is what an RTX 5090 needs.
RUN rm -rf /usr/local/cuda-13.1/compat /usr/local/cuda-13/compat /usr/local/cuda/compat

COPY --from=build /out/ninfer /usr/local/bin/ninfer
COPY --from=build /out/ninfer-serve /usr/local/bin/ninfer-serve

# Default: the card's all-round profile from the model card - 262,144-token
# window, rk4v4 KV, GDN state in fp16, DFlash2 drafter with five drafts.
# Every knob is overridable through the environment; mount the .ninfer artifact
# where MODEL_PATH points (the default assumes the snapshot directory mounted
# at /workspace).
RUN cat > /entrypoint.sh <<'EOF'
#!/bin/sh
set -eu

: "${MODEL_PATH:=/workspace/Ternary-Bonsai-2-27B-ninfer-v3.ninfer}"
: "${MODEL_ID:=bonsai2-27b}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8080}"
: "${MAX_CONTEXT:=262144}"
: "${KV_CAPACITY:=${MAX_CONTEXT}}"
: "${KV_DTYPE:=rk4v4}"
: "${SPEC:=dflash2}"
: "${DRAFT_TOKENS:=5}"
: "${EXTRA_ARGS:=}"

args="$MODEL_PATH --model-id $MODEL_ID --host $HOST --port $PORT"
args="$args --max-context $MAX_CONTEXT --kv-capacity $KV_CAPACITY"
args="$args --kv-dtype $KV_DTYPE --gdn-state-fp16"
if [ -n "$SPEC" ]; then
  args="$args --spec $SPEC --draft-tokens $DRAFT_TOKENS"
fi
args="$args $EXTRA_ARGS"

# shellcheck disable=SC2086
exec ninfer-serve $args
EOF
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

ENTRYPOINT ["/entrypoint.sh"]
CMD []
