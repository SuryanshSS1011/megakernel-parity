#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
# T6: verify(M=k+1) vs decode(M=1) on stock Hugging Face kernels, the non-megakernel
# baseline. Llama-3.2-1B-Instruct is the model Hazy runs, so it links the two systems.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
out="results/hf_parity/${MKP_GPU_TAG}"; mkdir -p "$out"
status=0
for model in ${MKP_MODELS_HF:-meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3-1.7B}; do
  m="$(echo "$model" | tr '/' '_')"
  for variant in "bfloat16 sdpa" "bfloat16 eager" "float32 sdpa"; do
    read -r dtype attn <<<"$variant"
    python -I harness/parity_hf.py --model "$model" --k 1 2 4 7 --dtype "$dtype" --attn "$attn" \
      --prompts-file harness/prompts.txt --out "$out/${m}__${dtype}_${attn}.jsonl" \
      > "$out/${m}__${dtype}_${attn}.log" 2>&1 || { echo "FAILED $model $variant"; status=1; }
    tail -n 5 "$out/${m}__${dtype}_${attn}.log"
  done
done
exit $status
