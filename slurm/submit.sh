#!/usr/bin/env bash
# Submit one job from slurm/jobs/ with the account, partition and GPU filled in.
#
#   slurm/submit.sh <job> [--gpu a40|a100] [--after JOBID] [--env KEY=VALUE]... [--dry-run] [-- extra sbatch args]
#
# Examples:
#   slurm/submit.sh probe --gpu a40
#   B=$(slurm/submit.sh build --gpu a40); slurm/submit.sh smoke --gpu a40 --after "$B"
#   slurm/submit.sh margins --gpu a40 --env MKP_PAIRS="ns:spec_k3 ns:ns_B4"
#   slurm/submit.sh full --gpu a40 -- --time=12:00:00
#
# Prints only the job id on success, so it can be captured for dependencies.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh" >/dev/null

job="${1:?usage: slurm/submit.sh <job> [--gpu a40|a100] [--after JOBID] [--dry-run] [-- sbatch args]}"
shift
gpu="a40"; after=""; dry=0; extra=(); exports="ALL"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --after) after="$2"; shift 2 ;;
    --env) [[ "$2" == *=* ]] || { echo "--env needs KEY=VALUE" >&2; exit 2; }
           exports+=",$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    --) shift; extra=("$@"); break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

script="$here/jobs/$job.sbatch"
[[ -f "$script" ]] || { echo "no such job: $script" >&2; ls "$here/jobs" >&2; exit 2; }
case "$gpu" in a40|a100) ;; *) echo "--gpu must be a40 or a100" >&2; exit 2 ;; esac
if [[ -z "${ROAR_ACCOUNT:-}" ]]; then
  echo "ROAR_ACCOUNT is not set; add it to .env in the repository root" >&2; exit 2
fi

cmd=(sbatch --parsable
     --account="$ROAR_ACCOUNT"
     --partition="${ROAR_PARTITION:-standard}"
     --gres="gpu:$gpu:1"
     --job-name="mkp-$job-$gpu"
     --output="$MKP_REPO/slurm/logs/%x-%j.out"
     --chdir="$MKP_REPO"
     --export="$exports,MKP_REPO=$MKP_REPO")
[[ -n "$after" ]] && cmd+=(--dependency="afterok:$after" --kill-on-invalid-dep=yes)
for a in "${extra[@]}"; do
  [[ "$a" == --export* ]] && { echo "use --env KEY=VALUE instead of --export" >&2; exit 2; }
done
cmd+=("${extra[@]}" "$script")

if [[ $dry -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"; echo
else
  "${cmd[@]}"
fi
