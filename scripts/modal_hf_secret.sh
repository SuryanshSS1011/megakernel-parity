#!/usr/bin/env bash
# Create or refresh the Modal secret "huggingface" from HF_TOKEN in ~/.mkparity.env.
# The token goes through a private temp file, never the command line, so it does not
# show up in `ps` on a shared login node.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../slurm/common.sh" >/dev/null
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set in ~/.mkparity.env" >&2; exit 1; }
command -v modal >/dev/null || { echo "modal CLI not found; run slurm/setup.sh env" >&2; exit 1; }
umask 077
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > "$tmp"
modal secret create huggingface --from-dotenv "$tmp" --force >/dev/null
echo "Modal secret 'huggingface' updated"
