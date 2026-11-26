"""Run E3 (Hazy run-to-run determinism) on a Modal H100.

One-time setup on your machine:
  pip install modal && modal setup
  modal secret create huggingface HF_TOKEN=hf_...   # token with Llama-3.2 access
  python scripts/build_prompts.py                    # writes data/prompts/eval64.jsonl

Cheap first run, about 5 GPU-minutes:
  modal run experiments/hazy/modal_app.py --limit 2 --reps 3 --ntok 64

Full E3, an estimated 20 to 40 GPU-minutes:
  modal run experiments/hazy/modal_app.py --limit 32 --reps 20 --ntok 256

The image builds the megakernel with nvcc and needs no GPU. Image builds are
billed as CPU time, and the GPU is only attached while hazy_repeat.py runs.
Results land in results/hazy/<gpu>/ on your machine.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HAZY_COMMIT = "7309cec801537b61fea3b50d7dfe454a6cde578e"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "python3-dev")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone https://github.com/HazyResearch/Megakernels /opt/Megakernels",
        f"cd /opt/Megakernels && git checkout {HAZY_COMMIT} && git submodule update --init --recursive",
        "cd /opt/Megakernels && pip install -e .",
        # -lcuda resolves against the driver stub at build time; the real
        # libcuda.so.1 is used at run time on the GPU host.
        "cd /opt/Megakernels/demos/low-latency-llama && "
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "THUNDERKITTENS_ROOT=/opt/Megakernels/ThunderKittens MEGAKERNELS_ROOT=/opt/Megakernels "
        "PYTHON_VERSION=3.12 GPU=H100 make",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file(str(HERE / "hazy_repeat.py"), "/root/hazy_repeat.py")
)

app = modal.App("mkparity-hazy", image=image)
hf_cache = modal.Volume.from_name("mkparity-hf-cache", create_if_missing=True)


@app.function(gpu="H100", volumes={"/cache": hf_cache}, timeout=3 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def run_e3(prompts_jsonl: str, limit: int, reps: int, ntok: int, model: str) -> dict:
    import subprocess

    Path("/tmp/prompts.jsonl").write_text(prompts_jsonl)
    env_info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    subprocess.run(["python", "/root/hazy_repeat.py", "--hazy-root", "/opt/Megakernels",
                    "--model", model, "--prompts", "/tmp/prompts.jsonl", "--limit", str(limit),
                    "--reps", str(reps), "--ntok", str(ntok), "--out", "/tmp/out/e3.json"], check=True)
    hf_cache.commit()
    files = {p.name: p.read_text() for p in Path("/tmp/out/e3").glob("*.json")}
    return dict(summary=Path("/tmp/out/e3.json").read_text(), files=files, nvidia_smi=env_info,
                hazy_commit=HAZY_COMMIT)


@app.local_entrypoint()
def main(limit: int = 2, reps: int = 3, ntok: int = 64, model: str = "meta-llama/Llama-3.2-1B-Instruct"):
    prompts = REPO / "data" / "prompts" / "eval64.jsonl"
    if not prompts.exists():
        raise SystemExit("run `python scripts/build_prompts.py` first")
    res = run_e3.remote(prompts.read_text(), limit, reps, ntok, model)
    gpu = json.loads(res["summary"])["gpu"]["name"].replace(" ", "_")
    out = REPO / "results" / "hazy" / gpu / f"e3_l{limit}_r{reps}_n{ntok}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in res["files"].items():
        (out / name).write_text(text)
    (out / "summary.json").write_text(json.dumps(dict(json.loads(res["summary"]), nvidia_smi=res["nvidia_smi"],
                                                      hazy_commit=res["hazy_commit"]), indent=1))
    print(f"saved {len(res['files'])} runs to {out}")
