#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
# T3: build MPK (one build serves the A40 and the A100), then check it sees the GPU.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
cd "$MIRAGE_ROOT"
test "$(git rev-parse HEAD)" = "$MIRAGE_COMMIT" || { echo "mirage is not at $MIRAGE_COMMIT" >&2; exit 1; }
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}" CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-16}"
time python -m pip install -e . -v --no-build-isolation --no-deps
cd "$MKP_REPO"
python - <<'PY'
import mirage as mi, torch
print("mirage", mi.__file__)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("num_workers, num_schedulers =", mi.get_configurations_from_gpu(0))
PY
mkdir -p results/env
python scripts/env_report.py --mirage-root "$MIRAGE_ROOT" > "results/env/${MKP_GPU_TAG}.json"
echo "build ok"
