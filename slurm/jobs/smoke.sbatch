#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
# T4: one prompt through every MPK arm for two model sizes. This is the go/no-go gate.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
status=0
for model in Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B; do
  python experiments/mpk/run_mpk.py --mirage-root "$MIRAGE_ROOT" --model "$model" \
    --profile smoke --retry-failed ${MKP_RUN_ARGS:-} || status=1
done
python scripts/analyze.py results/mpk --out "results/summary_smoke_${MKP_GPU_TAG}" || status=1
exit $status
