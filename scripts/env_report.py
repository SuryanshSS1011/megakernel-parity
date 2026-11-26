"""Record the software and hardware environment of a run as JSON.

  python scripts/env_report.py --mirage-root ~/mirage > results/env_$(hostname).json
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys


def sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_rev(path: str | None) -> str | None:
    if not path:
        return None
    return sh(["git", "-C", path, "rev-parse", "HEAD"])


def gpu_info() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append(dict(index=i, name=p.name, capability=f"{p.major}.{p.minor}",
                        sm_count=p.multi_processor_count, total_memory_gb=round(p.total_memory / 2**30, 1)))
    return out


def report(mirage_root: str | None = None, hazy_root: str | None = None) -> dict:
    try:
        import torch
        torch_v, cuda_v = torch.__version__, torch.version.cuda
    except ImportError:
        torch_v = cuda_v = None
    mpk_workers = None
    try:
        import mirage
        mpk_workers = [list(mirage.get_configurations_from_gpu(g["index"])) for g in gpu_info()]
    except Exception:  # mirage absent or no GPU
        pass
    return dict(
        host=platform.node(), python=sys.version.split()[0], platform=platform.platform(),
        torch=torch_v, torch_cuda=cuda_v,
        nvcc=sh(["nvcc", "--version"]).splitlines()[-1] if shutil.which("nvcc") else None,
        driver=sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        gpus=gpu_info(), mpk_workers_schedulers=mpk_workers,
        mirage_commit=git_rev(mirage_root), hazy_commit=git_rev(hazy_root),
        mkparity_commit=git_rev("."),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirage-root")
    ap.add_argument("--hazy-root")
    a = ap.parse_args()
    print(json.dumps(report(a.mirage_root, a.hazy_root), indent=1))
