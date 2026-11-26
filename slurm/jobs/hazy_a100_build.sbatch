#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
# O1: try the Hazy low-latency build with GPU=A100. It is expected to fail on Hopper
# TMA instructions; the log is the result either way.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
mkdir -p results/hazy_a100
log="$MKP_REPO/results/hazy_a100/build_${MKP_GPU_TAG}.log"
cd "$HAZY_ROOT"
python -m pip install -e . --no-build-isolation --no-deps >"$log" 2>&1 || true
cd demos/low-latency-llama
pyver="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
set +e
THUNDERKITTENS_ROOT="$HAZY_ROOT/ThunderKittens" MEGAKERNELS_ROOT="$HAZY_ROOT" \
  PYTHON_VERSION="$pyver" GPU=A100 make -j"${SLURM_CPUS_PER_TASK:-12}" >>"$log" 2>&1
rc=$?
echo "make exit code $rc" | tee -a "$log"
grep -m 20 -E "error|Error" "$log" || true
exit 0
