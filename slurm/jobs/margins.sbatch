#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
# T10: fp32 top-2 margin at each first divergence. MKP_PAIRS lists "a:b" arm pairs
# that results/summary.md shows diverging; every GPU directory holding both arms is used.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
model="${MKP_MODEL:-Qwen/Qwen3-1.7B}"; m="$(echo "$model" | tr '/' '_')"
status=0
for pair in ${MKP_PAIRS:-ns:spec_k3 ns:spec_k5}; do
  a="${pair%%:*}"; b="${pair##*:}"
  for runs in results/mpk/*/"$m"; do
    [[ -d "$runs/$a" && -d "$runs/$b" ]] || continue
    gpu="$(basename "$(dirname "$runs")")"
    python experiments/margins/margin_at_divergence.py --runs "$runs" --a "$a" --b "$b" \
      --model "$model" --out "results/margins/${gpu}__${m}__${a}_vs_${b}.json" || status=1
  done
done
exit $status
