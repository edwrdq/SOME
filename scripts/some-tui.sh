#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script is in scripts/)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Defaults that you can override in env
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/steam/uv-cache/}"
# Prefer CPU unless you have CUDA configured properly
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"

cd "${REPO_ROOT}"

if command -v uv >/dev/null 2>&1; then
  exec uv run python src/tui.py "$@"
else
  echo "[info] uv not found; falling back to python" >&2
  exec python src/tui.py "$@"
fi

