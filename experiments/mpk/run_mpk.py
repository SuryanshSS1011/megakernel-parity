"""Run MPK's Qwen3 demo across prompts and experiment arms, one launch per run.

Each arm changes one thing relative to the reference arm `ns`, so any token
difference can be traced to that change. Every launch writes one
`mkparity.run.v1` JSON file. Finished files are skipped, so an interrupted
sweep can be resumed by rerunning the same command.

  python experiments/mpk/run_mpk.py --mirage-root ~/mirage --model Qwen/Qwen3-1.7B \\
      --profile smoke --dry-run
  python experiments/mpk/run_mpk.py --mirage-root ~/mirage --model Qwen/Qwen3-1.7B --profile full

Arms (all greedy, --ignore-eos, same prompts, same max lengths):
  ns          plain decode, the reference
  ns_nocache  plain decode compiled fresh for every prompt, to check kernel reuse is exact
  ns_rep      plain decode repeated 5 times, the run-to-run noise floor
  spec_k3     prompt-lookup speculative decoding, 3 draft tokens
  spec_k5     prompt-lookup speculative decoding, 5 draft tokens
  ns_T16      plain decode built with max_num_batched_tokens=16 instead of 8, which also
              changes the prefill chunk size
  ns_splitkv  plain decode with split-KV attention
  ns_B4       plain decode with 4 identical requests in one batch
  torch       MPK's PyTorch reference model, no megakernel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
from patch_demo import PATCHED_NAME, write_patched_demo  # noqa: E402


@dataclass
class Arm:
    name: str
    extra: list = field(default_factory=list)
    tokens: int = 8          # --max-num-batched-tokens
    requests: int = 1        # --max-num-batched-requests
    mirage: bool = True
    reuse: bool = True       # reuse one compiled kernel across prompts
    reps: int = 1
    limit: int | None = None  # only the first N prompts


ARMS = {a.name: a for a in [
    Arm("ns"),
    Arm("ns_nocache", reuse=False, limit=4),
    Arm("ns_rep", reps=5, limit=8),
    Arm("spec_k3", extra=["--spec-decode", "promptlookup", "--spec-length", "3"]),
    Arm("spec_k5", extra=["--spec-decode", "promptlookup", "--spec-length", "5"]),
    Arm("ns_T16", tokens=16),
    Arm("ns_splitkv", extra=["--split-kv-cache"]),
    Arm("ns_B4", requests=4, limit=16),
    Arm("torch", mirage=False, reuse=False),
]}

PROFILES = {
    # one prompt through every arm, to find build or API problems quickly
    "smoke": dict(arms=["ns", "ns_nocache", "spec_k3", "ns_T16", "ns_splitkv", "ns_B4", "torch"],
                  limit=1, new_tokens=64),
    # the main A40 study
    "full": dict(arms=["ns", "ns_nocache", "ns_rep", "spec_k3", "spec_k5", "ns_T16",
                       "ns_splitkv", "ns_B4", "torch"], limit=None, new_tokens=512),
    # the A100 side of the cross-GPU comparison
    "cross": dict(arms=["ns", "ns_rep", "spec_k3"], limit=None, new_tokens=512),
}


def gpu_tag() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return dict(name=p.name, capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count)
    except ImportError:
        pass
    return dict(name="unknown", capability=None, sm_count=None)


def slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def kernel_key(arm: Arm, args) -> str:
    cfg = [args.model, arm.tokens, arm.requests, args.max_seq_length, sorted(arm.extra)]
    return hashlib.sha1(json.dumps(cfg).encode()).hexdigest()[:10]


def build_cmd(arm: Arm, prompt: str, demo: Path, save: Path, kdir: Path, args) -> list[str]:
    cmd = [args.python, str(demo), "--model", args.model,
           "--max-new-tokens", str(args.new_tokens), "--ignore-eos",
           "--max-seq-length", str(args.max_seq_length),
           "--max-num-batched-tokens", str(arm.tokens),
           "--max-num-batched-requests", str(arm.requests),
           "--save-tokens", str(save), "--prompt", prompt]
    if arm.mirage:
        cmd += ["--use-mirage", "--output-dir", str(kdir)]
    return cmd + arm.extra


def run(args) -> int:
    prof = PROFILES[args.profile]
    arms = args.arms.split(",") if args.arms else prof["arms"]
    args.new_tokens = args.new_tokens or prof["new_tokens"]
    limit = args.limit if args.limit is not None else prof["limit"]
    prompts = load_prompts(Path(args.prompts), limit)
    demo = Path(args.mirage_root) / "demo" / "qwen3" / PATCHED_NAME
    if not args.dry_run:
        demo = write_patched_demo(args.mirage_root)
    gpu = gpu_tag()
    base = Path(args.out) / slug(gpu["name"]) / slug(args.model)
    commit = None
    try:
        commit = subprocess.run(["git", "-C", args.mirage_root, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or None
    except OSError:
        pass

    n_done = n_fail = n_skip = 0
    for an in arms:
        arm = ARMS[an]
        kdir = base / "_kernels" / f"{an}_{kernel_key(arm, args)}"
        ps = prompts[: arm.limit] if arm.limit else prompts
        for p in ps:
            for rep in range(arm.reps):
                final = base / an / f"{p['prompt_id']}__r{rep}.json"
                if final.exists() and not args.force:
                    n_skip += 1
                    continue
                final.parent.mkdir(parents=True, exist_ok=True)
                save = final.with_suffix(".demo.json")
                this_kdir = kdir if arm.reuse else base / "_kernels" / f"{an}_{p['prompt_id']}_r{rep}"
                cmd = build_cmd(arm, p["prompt"], demo, save, this_kdir, args)
                env = dict(os.environ)
                env.setdefault("MIRAGE_HOME", str(Path(args.mirage_root).resolve()))
                env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # as in MPK's own CI
                if arm.mirage and arm.reuse and kdir.exists():
                    env["MKP_REUSE_KERNEL_DIR"] = str(kdir)
                else:
                    env.pop("MKP_REUSE_KERNEL_DIR", None)
                if args.dry_run:
                    reuse = ("torch" if not arm.mirage else
                             "reuse" if "MKP_REUSE_KERNEL_DIR" in env else "compile")
                    print(f"[{an} {p['prompt_id']} r{rep} {reuse}] " + shlex.join(cmd[:14]) + " ...")
                    n_done += 1
                    continue
                log = final.with_suffix(".log")
                t0 = time.time()
                with open(log, "w") as lf:
                    try:
                        rc = subprocess.run(cmd, cwd=str(final.parent), env=env, stdout=lf,
                                            stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                    except subprocess.TimeoutExpired:
                        rc = -9
                wall = time.time() - t0
                rec = dict(schema="mkparity.run.v1", system="mpk", arm=an, prompt_id=p["prompt_id"],
                           task=p.get("task"), rep=rep, model=args.model, gpu=gpu, mirage_commit=commit,
                           config=dict(max_new_tokens=args.new_tokens, max_seq_length=args.max_seq_length,
                                       max_num_batched_tokens=arm.tokens,
                                       max_num_batched_requests=arm.requests, extra=arm.extra,
                                       mirage=arm.mirage, reuse=arm.reuse),
                           returncode=rc, wall_s=round(wall, 2), log=log.name)
                if rc == 0 and save.exists():
                    demo_out = json.loads(save.read_text())
                    rec.update(token_ids=demo_out.get("token_ids"),
                               all_token_ids=demo_out.get("all_token_ids"),
                               prompt_length=demo_out.get("prompt_length"),
                               latency_ms_per_token=demo_out.get("latency_ms_per_token"),
                               num_workers=demo_out.get("num_workers"),
                               kernel_source=demo_out.get("kernel_source"),
                               kernel_seconds=demo_out.get("kernel_seconds"),
                               run_time_ms=demo_out.get("run_time_ms"))
                    save.unlink()
                    n_done += 1
                else:
                    n_fail += 1
                    print(f"FAILED rc={rc} {an} {p['prompt_id']} r{rep}; see {log}", file=sys.stderr)
                final.write_text(json.dumps(rec))
                print(f"[{an} {p['prompt_id']} r{rep}] rc={rc} {wall:.0f}s "
                      f"{rec.get('kernel_source', '')}", flush=True)
                if rc != 0 and args.stop_on_fail:
                    return 1
    print(f"done={n_done} failed={n_fail} skipped={n_skip}")
    return 1 if n_fail else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mirage-root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--prompts", default=str(REPO / "data" / "prompts" / "eval64.jsonl"))
    ap.add_argument("--profile", choices=list(PROFILES), default="smoke")
    ap.add_argument("--arms", help="comma-separated arm names, overriding the profile")
    ap.add_argument("--limit", type=int, help="number of prompts, overriding the profile")
    ap.add_argument("--new-tokens", type=int, help="generated tokens, overriding the profile")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--out", default=str(REPO / "results" / "mpk"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--force", action="store_true", help="rerun finished launches")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
