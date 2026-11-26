"""Static checks for the ROAR job scripts: shell syntax, submit wrapper, no secrets."""
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted([*REPO.glob("slurm/*.sh"), *REPO.glob("slurm/jobs/*.sbatch"), *REPO.glob("scripts/*.sh")])


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_jobs_have_time_and_gpu_and_no_account():
    for job in REPO.glob("slurm/jobs/*.sbatch"):
        text = job.read_text()
        assert "#SBATCH --time=" in text, job.name
        assert "#SBATCH --gres=gpu:" in text, job.name
        assert "--account" not in text, f"{job.name}: the account comes from slurm/submit.sh"
        assert "slurm/common.sh" in text, job.name


def _submit(tmp_path, *args, account="abc123_gpu"):
    env = dict(os.environ, HOME=str(tmp_path), MKP_WORK=str(tmp_path / "w"),
               MKP_SCRATCH=str(tmp_path / "s"), ROAR_ACCOUNT=account)
    return subprocess.run(["bash", str(REPO / "slurm" / "submit.sh"), *args], env=env,
                          capture_output=True, text=True)


def test_submit_dry_run(tmp_path):
    r = _submit(tmp_path, "smoke", "--gpu", "a100", "--after", "42", "--env", "MKP_PAIRS=ns:spec_k3", "--dry-run")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--account=abc123_gpu" in out and "--gres=gpu:a100:1" in out
    assert "afterok:42" in out and "MKP_PAIRS=ns:spec_k3" in out and "MKP_REPO=" in out


def test_submit_refuses_bad_input(tmp_path):
    assert _submit(tmp_path, "smoke", "--dry-run", account="").returncode == 2
    assert _submit(tmp_path, "nope", "--dry-run").returncode == 2
    assert _submit(tmp_path, "smoke", "--gpu", "h100", "--dry-run").returncode == 2
    assert _submit(tmp_path, "smoke", "--dry-run", "--", "--export=X").returncode == 2


def test_no_tokens_in_tracked_text():
    token = re.compile(r"hf_[A-Za-z0-9]{30,}")
    files = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True).stdout.split()
    files += [str(p.relative_to(REPO)) for p in SCRIPTS] + ["env.example", "CLAUDE.md", "RUNBOOK.md"]
    for f in sorted(set(files)):
        p = REPO / f
        if p.is_file() and p.stat().st_size < 2_000_000:
            assert not token.search(p.read_text(errors="ignore")), f"possible token in {f}"
    assert "HF_TOKEN=\n" in (REPO / "env.example").read_text()
