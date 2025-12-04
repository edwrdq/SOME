#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/steam/uv-cache/}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"

cd "${REPO_ROOT}"

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m src.dashboard.app "$@"
else
  echo "[info] uv not found; falling back to python" >&2
  exec python -m src.dashboard.app "$@"
fi

