# Instructions for coding agents

This repository measures numerical divergence in LLM megakernels. `docs/plan.md` holds the study design and `docs/evaluation.md` holds the metrics and pass criteria. `RUNBOOK.md` lists the GPU tasks in order, and it is the task list to work through. Read all three before submitting anything.

## Where things run

1. The repository lives on ROAR Collab, a Slurm cluster. Login nodes have internet but no GPU, and compute nodes have GPUs but may have no internet.
2. Run every GPU workload through `slurm/submit.sh`, never directly on a login node. Use login nodes only for git, pip, downloads, editing and CPU analysis such as `scripts/analyze.py`.
3. The cluster offers A40 (sm_86, 84 SMs) and A100 40GB (sm_80, 108 SMs) GPUs. It has no H100, so the Hazy runs go to Modal (task T9).
4. Paths, modules and the Slurm account come from `~/.mkparity.env`, which follows `env.example`. Fix a wrong module name or path there, not in the scripts.

## Secrets

1. Never print, `cat`, `echo` or log `HF_TOKEN`, `~/.mkparity.env`, `~/.modal.toml` or `~/.cache/huggingface/token`. Check whether a token is set with `[[ -n "$HF_TOKEN" ]]` and nothing more.
2. Never pass a token as a command-line argument, because other users on a shared node can see process arguments. `scripts/modal_hf_secret.sh` shows the safe pattern.
3. Never commit a token or a file that holds one. If a token appears in any output, log or file, stop and tell the user so they can revoke it.

## Keeping the experiment intact

1. Do not change the experiment definition to make a run pass. That covers the prompt set, the arm list and arm settings in `run_mpk.py`, token counts, models, seeds, the pinned mirage commit `ed771bd` and the pinned Hazy commit `7309cec`.
2. Fixes that only make the code build or run are allowed. Commit each one separately and explain it in `docs/run_log.md`. If an exact-match edit in `experiments/mpk/patch_demo.py` stops applying, update it so it keeps the same intent.
3. Never delete or overwrite results. Resume interrupted or failed runs with `--retry-failed` and use `--force` only if the user asks for it.
4. Treat a divergence as a result, not a bug to fix. The study exists to measure them.

## When to stop and report

Stop, record the finding in `docs/run_log.md`, commit and report to the user in any of these cases.

1. The MPK build fails twice for the same reason.
2. Every MPK arm fails in the smoke test (task T4). This is the project's go/no-go gate.
3. Any job fails twice in a row for the same reason.
4. Modal spending would pass $15 for task T9 or $25 in total.
5. A storage quota is exceeded. `quota_check` reports usage.
6. A task needs a decision the RUNBOOK does not cover, such as changing the experiment definition.

## Committing results

1. Work on a branch named `roar-runs`, push it and open a pull request once T4 finishes. Then push again after each later task.
2. Commit `results/**/*.json`, `results/**/*.jsonl`, `results/**/*.md`, the run `.log` files, `slurm/logs/*.out` and `docs/run_log.md`.
3. Never commit `results/**/_kernels/`, `.so` or `.cu` files, model weights or anything under `data/`. `.gitignore` covers these, so do not force-add anything it ignores.
4. Gzip any log over 5 MB before committing it, and leave out any other file over 5 MB.
5. Commits are authored by the user's git identity. Keep the agent co-author trailer.

## Reporting format

After each task, add an entry to `docs/run_log.md` with the date, task id, job ids, GPU and node, wall time, whether each acceptance check passed and the headline numbers. Then tell the user the same in a few sentences, including the next task you would start.

## Useful commands

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName%24,State,Elapsed,MaxRSS,ExitCode
tail -n 50 slurm/logs/mkp-<job>-<gpu>-<jobid>.out
python scripts/analyze.py results/mpk --out results/summary
```
