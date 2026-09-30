#!/usr/bin/env bash
# Start the Ternary-Bonsai-2-27B NInfer server (901,120-token window with
# --rope-yarn, rk4v4 KV, DFlash2 drafter, vision tower in overlay residency,
# 32k default output cap) on one RTX 5090. The model artifact is hard-linked
# from the local HuggingFace cache into ./model and mounted read-only; nothing
# is baked into the image.
#
# Usage:
#   ./run.sh                      # detached server on GPU 0, host port 8080
#   GPU=1 HOST_PORT=8081 ./run.sh # different card / port
#   SPEC= ./run.sh                # no drafter (frees ~1.5 GiB, slower decode)
#   ./run.sh --foreground         # run attached, Ctrl-C stops it
set -euo pipefail

HF_SNAP="/root/.cache/huggingface/hub/models--WaveCut--Ternary-Bonsai-2-27B-NInfer-v3/snapshots/b85b33627b27b9757a5a094fc74785a53e99f8ee"
MODEL_FILE="Ternary-Bonsai-2-27B-ninfer-v3.ninfer"
MODEL_DIR="/root/ninfer-bonsai/model"

IMAGE="${IMAGE:-ninfer-bonsai2-27b:sm120a-2172a598}"
NAME="${NAME:-bonsai2-27b-ninfer}"
GPU="${GPU:-0}"
HOST_PORT="${HOST_PORT:-8080}"

# HuggingFace stores the artifact as a chain of symlinks (snapshot entry ->
# blobs/<sha> -> sharded blob store). Those link targets live outside the
# snapshot directory, so a Docker bind-mount of the snapshot dir leaves the
# name dangling inside the container. Instead, mount a small dir holding a
# hardlink to the resolved blob (same filesystem => no data copy, and it is
# re-created whenever the cache is refreshed).
mkdir -p "$MODEL_DIR"
REAL_MODEL="$(readlink -f "${HF_SNAP}/${MODEL_FILE}" 2>/dev/null || true)"
if [ -n "$REAL_MODEL" ] && [ -f "$REAL_MODEL" ]; then
  ln -f "$REAL_MODEL" "${MODEL_DIR}/${MODEL_FILE}"
fi
[ -f "$MODEL_DIR/$MODEL_FILE" ] || {
  echo "model artifact not found: ${HF_SNAP}/${MODEL_FILE}" >&2
  exit 1
}

# Deployment defaults for this box; each still yields to an exported
# variable of the same name. KV coding (rk4v4) and drafter (DFlash2, 5
# drafts) stay at the image defaults.
MAX_CONTEXT="${MAX_CONTEXT:-901120}"
KV_CAPACITY="${KV_CAPACITY:-$MAX_CONTEXT}"
if [ -z "${EXTRA_ARGS:-}" ]; then
  EXTRA_ARGS="--rope-yarn --vision --vision-residency overlay --vision-max-merged 12288 --default-max-tokens 32768"
fi

# Tunables: MAX_CONTEXT, KV_CAPACITY, KV_DTYPE, SPEC, DRAFT_TOKENS,
# MODEL_ID, EXTRA_ARGS
envs=()
for v in MAX_CONTEXT KV_CAPACITY KV_DTYPE SPEC DRAFT_TOKENS MODEL_ID HOST PORT EXTRA_ARGS; do
  if [ "${!v:-}" != "" ]; then envs+=("-e" "$v=${!v}"); fi
done

if [ "${1:-}" = "--foreground" ]; then
  exec docker run --rm -it \
    --name "${NAME}-fg" \
    --gpus "device=${GPU}" \
    -p "${HOST_PORT}:8080" \
    -v "${MODEL_DIR}:/workspace:ro" \
    "${envs[@]}" \
    "$IMAGE"
fi

docker rm -f "$NAME" 2>/dev/null || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --gpus "device=${GPU}" \
  -p "${HOST_PORT}:8080" \
  -v "${MODEL_DIR}:/workspace:ro" \
  "${envs[@]}" \
  "$IMAGE"

echo "container '$NAME' started on GPU ${GPU}, host port ${HOST_PORT}"
echo "watch: docker logs -f $NAME"
echo "chat:  curl -s http://127.0.0.1:${HOST_PORT}/v1/chat/completions -H 'content-type: application/json' \\"
echo "        -d '{\"model\":\"bonsai2-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}]}'"
