#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
# T8: the A100 side of the SM-count comparison, plus kernel checks on sm_80.
# Results land next to the A40 results on the shared filesystem, so analyze.py
# picks up the cross-GPU rows without copying anything.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
status=0
python experiments/mpk/run_mpk.py --mirage-root "$MIRAGE_ROOT" --model "${MKP_MODEL:-Qwen/Qwen3-1.7B}" \
  --profile cross --retry-failed ${MKP_RUN_ARGS:-} || status=1
python experiments/mpk/kernel_parity.py --out "results/mpk_kernels/${MKP_GPU_TAG}.json" || status=1
python scripts/analyze.py results/mpk --out results/summary || status=1
exit $status
