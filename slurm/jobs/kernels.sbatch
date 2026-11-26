#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
# T5 and T8: MPK test-mode kernel checks (row count, task count, token count).
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
python experiments/mpk/kernel_parity.py --out "results/mpk_kernels/${MKP_GPU_TAG}.json" ${MKP_RUN_ARGS:-}
