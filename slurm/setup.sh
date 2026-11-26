#!/usr/bin/env bash
# One-time setup on a ROAR Collab login node (internet, no GPU).
# Creates the Python environment, clones MPK and Hazy at pinned commits,
# prefetches Rust crates and model weights, and builds the prompt set.
# The heavy MPK build runs later on a GPU node (slurm/jobs/build.sbatch).
#
#   bash slurm/setup.sh            # everything
#   bash slurm/setup.sh models     # only (re)download model weights
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"
mkp_banner
step="${1:-all}"

pick_torch() {
  # Match the torch CUDA build to the loaded nvcc (major.minor).
  local v; v="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  case "$v" in
    12.1|12.2|12.3) echo "torch==2.5.1 https://download.pytorch.org/whl/cu121" ;;
    12.4|12.5)      echo "torch==2.5.1 https://download.pytorch.org/whl/cu124" ;;
    12.6|12.7)      echo "torch==2.7.1 https://download.pytorch.org/whl/cu126" ;;
    12.8|12.9|13.*) echo "torch==2.7.1 https://download.pytorch.org/whl/cu128" ;;
    *) echo "unsupported nvcc $v; set MKP_CUDA_MODULE to a CUDA 12.x module" >&2; return 1 ;;
  esac
}

do_env() {
  command -v nvcc >/dev/null || { echo "nvcc not found; set MKP_CUDA_MODULE (see: module avail cuda)" >&2; exit 1; }
  python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "need Python >= 3.10; set MKP_PYTHON_MODULE (see: module avail python)" >&2; exit 1; }
  if [[ ! -f "$MKP_ENV/bin/activate" ]]; then
    python3 -m venv "$MKP_ENV"
  fi
  source "$MKP_ENV/bin/activate"
  python -m pip install -U pip setuptools wheel
  read -r spec index <<<"${MKP_TORCH:-$(pick_torch)}"
  python -m pip install "$spec" --index-url "$index"
  python -m pip install "cmake>=3.24" "cython>=0.28" ninja modal
  python -m pip install -e "$MKP_REPO[dev]"
}

do_mirage() {
  if [[ ! -d "$MIRAGE_ROOT/.git" ]]; then
    git clone --recursive https://github.com/mirage-project/mirage "$MIRAGE_ROOT"
  fi
  git -C "$MIRAGE_ROOT" fetch --quiet origin
  git -C "$MIRAGE_ROOT" checkout --quiet "$MIRAGE_COMMIT"
  git -C "$MIRAGE_ROOT" submodule update --init --recursive
  # Everything mirage needs at build time, so the GPU-node build can run offline.
  # torch is already installed; cuda-python 13 targets CUDA 13, so cap it to the toolkit's major.
  local major; major="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
  grep -v '^torch' "$MIRAGE_ROOT/requirements.txt" \
    | sed "s/^cuda-python\$/cuda-python<$((major + 1))/" > "$MKP_SCRATCH/mirage-reqs.txt"
  python -m pip install -r "$MKP_SCRATCH/mirage-reqs.txt"
  if ! command -v cargo >/dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
  fi
  for crate in src/search/abstract_expr/abstract_subexpr src/search/verification/formal_verifier_equiv; do
    (cd "$MIRAGE_ROOT/$crate" && cargo fetch)
  done
}

do_hazy() {
  # Only for the optional A100 build attempt; the H100 runs build inside Modal.
  if [[ ! -d "$HAZY_ROOT/.git" ]]; then
    git clone https://github.com/HazyResearch/Megakernels "$HAZY_ROOT"
  fi
  git -C "$HAZY_ROOT" checkout --quiet "$HAZY_COMMIT"
  git -C "$HAZY_ROOT" submodule update --init --recursive
}

do_models() {
  python - <<'EOF'
import os
from huggingface_hub import snapshot_download
models = os.environ.get("MKP_MODELS",
    "Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B meta-llama/Llama-3.2-1B-Instruct").split()
for m in models:
    if m.startswith("meta-llama/") and not os.environ.get("HF_TOKEN"):
        print(f"skip {m}: HF_TOKEN is not set"); continue
    p = snapshot_download(m, ignore_patterns=["original/*", "*.pth", "*.gguf"])
    print(f"ok {m} -> {p}")
EOF
}

do_data() {
  (cd "$MKP_REPO" && python scripts/build_prompts.py)
}

case "$step" in
  all) do_env; do_mirage; do_hazy; do_models; do_data ;;
  env) do_env ;;
  mirage) do_mirage ;;
  hazy) do_hazy ;;
  models) do_models ;;
  data) do_data ;;
  *) echo "unknown step $step (all|env|mirage|hazy|models|data)" >&2; exit 2 ;;
esac
echo "setup step '$step' finished"
