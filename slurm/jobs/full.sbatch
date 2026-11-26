#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
# T7: the main MPK sweep. Resumable: resubmit the same job if it hits the time limit.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
status=0
python experiments/mpk/run_mpk.py --mirage-root "$MIRAGE_ROOT" --model "${MKP_MODEL:-Qwen/Qwen3-1.7B}" \
  --profile full --retry-failed ${MKP_RUN_ARGS:-} || status=1
python scripts/analyze.py results/mpk --out results/summary || status=1
exit $status
