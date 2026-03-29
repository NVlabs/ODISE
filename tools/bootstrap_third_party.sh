#!/usr/bin/env bash
set -euo pipefail

FORCE_REINIT=false
if [[ "${1-}" == "--force" ]]; then
  FORCE_REINIT=true
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_CLONE_FALLBACK=true
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  USE_CLONE_FALLBACK=false
fi

boot_dep() {
  local name="$1"
  local url="$2"
  local path="$3"

  if [ -f "$path/.git" ]; then
    echo "[odise] $name already initialized as submodule link ($path)"
    return
  fi

  if [ -d "$path/.git" ]; then
    if [ "$FORCE_REINIT" = "true" ]; then
      echo "[odise] Replacing nested git checkout at $path with submodule/clone..."
      rm -rf "$path"
    else
    echo "[odise] $name already has a nested git checkout ($path/.git)."
    echo "[odise] Keeping as-is; remove that directory and rerun this script for a clean submodule checkout."
    return
    fi
  fi

  if [ -d "$path" ]; then
    echo "[odise] $name directory exists without git metadata; skipping auto-bootstrap."
    echo "[odise] Ensure this directory comes from a clean git checkout before running installs that depend on it."
    return
  fi

  if [ "$USE_CLONE_FALLBACK" = "true" ]; then
    echo "[odise] Cloning $name (non-git context)..."
    git clone --depth 1 "$url" "$path"
  else
    echo "[odise] Adding $name as submodule..."
    git submodule add --depth 1 "$url" "$path" || git submodule update --init --recursive "$path"
  fi
}

boot_dep "latent-diffusion" "https://github.com/CompVis/latent-diffusion.git" "third_party/latent-diffusion"
boot_dep "taming-transformers" "https://github.com/CompVis/taming-transformers.git" "third_party/taming-transformers"

if [ "$USE_CLONE_FALLBACK" = "false" ]; then
  git submodule update --init --recursive third_party/latent-diffusion third_party/taming-transformers || true
  echo "[odise] Submodule records refreshed."
fi

echo "[odise] Third_party bootstrap complete."
