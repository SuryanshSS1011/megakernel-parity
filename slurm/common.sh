#!/usr/bin/env bash
# Shared environment for megakernel-parity on ROAR Collab.
# Sourced by slurm/setup.sh, slurm/submit.sh and every slurm/jobs/*.sbatch.
# Override any variable in ~/.mkparity.env (template: env.example).
set -eo pipefail

if [[ -f "$HOME/.mkparity.env" ]]; then
  set -a; source "$HOME/.mkparity.env"; set +a
fi

export MKP_WORK="${MKP_WORK:-/storage/work/$USER}"
export MKP_SCRATCH="${MKP_SCRATCH:-/scratch/$USER}"
export MKP_REPO="${MKP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MKP_ENV="${MKP_ENV:-$MKP_WORK/envs/mkparity}"
export MIRAGE_ROOT="${MIRAGE_ROOT:-$MKP_WORK/mirage}"
export MIRAGE_HOME="$MIRAGE_ROOT"
export MIRAGE_COMMIT="${MIRAGE_COMMIT:-ed771bda1b1143e660eb36d0f82e6e24e9df7b9c}"
export HAZY_ROOT="${HAZY_ROOT:-$MKP_WORK/Megakernels}"
export HAZY_COMMIT="${HAZY_COMMIT:-7309cec801537b61fea3b50d7dfe454a6cde578e}"
export HF_HOME="${HF_HOME:-$MKP_SCRATCH/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MKP_SCRATCH/pip-cache}"
export CARGO_HOME="${CARGO_HOME:-$MKP_WORK/rust/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$MKP_WORK/rust/rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$MKP_REPO/slurm/logs"

# Module names differ between clusters and change over time. Override them in
# ~/.mkparity.env after checking `module avail python` and `module avail cuda`.
_mkp_module() {
  command -v module >/dev/null 2>&1 || return 1
  set +u; module load "$1" >/dev/null 2>&1; local rc=$?; return $rc
}
_mkp_module "${MKP_PYTHON_MODULE:-python/3.11.2}" || true
if ! command -v nvcc >/dev/null 2>&1; then
  _mkp_module "${MKP_CUDA_MODULE:-cuda}" || true
fi
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi

if [[ -f "$MKP_ENV/bin/activate" ]]; then
  source "$MKP_ENV/bin/activate"
fi

# Compute nodes may not reach the internet, so jobs read models from the cache
# that slurm/setup.sh filled. Set MKP_HF_OFFLINE=0 to allow downloads in jobs.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export HF_HUB_OFFLINE="${MKP_HF_OFFLINE:-1}"
  export CARGO_NET_OFFLINE="${CARGO_NET_OFFLINE:-true}"
fi

MKP_GPU_TAG="none"
if command -v nvidia-smi >/dev/null 2>&1; then
  _name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [[ -n "$_name" ]]; then
    MKP_GPU_TAG="$(echo "$_name" | sed 's/^NVIDIA //; s/[^A-Za-z0-9._-]/_/g')"
  fi
fi
export MKP_GPU_TAG

mkp_banner() {
  echo "=== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$MKP_GPU_TAG"
  echo "repo=$MKP_REPO commit=$(git -C "$MKP_REPO" rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "python=$(command -v python || echo missing) nvcc=$(command -v nvcc || echo missing)"
  echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
  if [[ -n "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN is set"; else echo "HF_TOKEN is not set"; fi
}
