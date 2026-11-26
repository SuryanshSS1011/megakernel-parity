#!/usr/bin/env bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:a40:1
# T1: record what a GPU node offers before anything is built.
source "${MKP_REPO:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
mkp_banner
set +e
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
echo "--- cuda modules"; module avail cuda 2>&1 | tail -n 30
echo "--- python modules"; module avail python 2>&1 | tail -n 30
echo "--- partition limits"; sinfo -o "%P %l %G" 2>/dev/null | sort -u | head -n 40
echo "--- internet from compute node"
curl -sS -o /dev/null -w "pypi %{http_code}\n" --max-time 10 https://pypi.org/simple/ || echo "pypi unreachable"
curl -sS -o /dev/null -w "hf %{http_code}\n" --max-time 10 https://huggingface.co || echo "hf unreachable"
echo "--- storage"; df -h "$MKP_WORK" "$MKP_SCRATCH" 2>/dev/null
exit 0
