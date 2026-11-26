"""E3: run-to-run determinism of the Hazy low-latency Llama megakernel.

Runs inside a HazyResearch/Megakernels checkout that has the low-latency demo
built (see RUNBOOK.md, or run it through modal_app.py). For each prompt it
does one torch prefill and then repeats the megakernel decode `--reps` times
from the same state. Each megakernel step is a separate launch, so the order
in which the four down-projection partial sums reach the residual can change
between repetitions. Every step's logits are hashed and its top-2 margin is
recorded.

  python hazy_repeat.py --hazy-root /opt/Megakernels --prompts eval64.jsonl \\
      --limit 32 --reps 20 --ntok 256 --out results/hazy/h100.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def build(hazy_root: Path, model_name: str, setting: str = "latency"):
    sys.path.insert(0, str(hazy_root))
    from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
    from megakernels.llama import LlamaForCausalLM
    from megakernels.model_types import ExtraModelConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, device="cuda:0",
        extra_config=ExtraModelConfig(interleave_rope=True, max_len_override=16384, max_batch_size=1))
    interp = make_mk_interpreter(setting, hazy_root / "demos" / "low-latency-llama")
    return tok, model, interp, make_schedule_builder(setting)


def run_prompt(tok, model, interp, builder, prompt: str, ntok: int, reps: int, with_torch: bool):
    import torch
    from megakernels.generators import MK_Generator, PyTorchGenerator
    from megakernels.model_types import BatchState
    from megakernels.scheduler import assign_to_sms, tensorize_instructions

    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    plen = ids.shape[-1]
    pre = model(BatchState(input_ids=ids, position_ids=torch.arange(plen, device=model.device)))
    first = pre.output_ids[:, -1:]

    schedule = builder.build(model)
    tensorize_instructions(schedule.globs, assign_to_sms("rr", schedule=schedule))

    class Hashing(MK_Generator):
        def run(self, input_ids, pos_id):
            out = super().run(input_ids, pos_id)
            lg = self.schedule.globs.logits.detach()
            raw = lg.contiguous().view(torch.int16) if lg.dtype in (torch.bfloat16, torch.float16) else lg
            self.hashes.append(hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()[:16])
            t2 = lg.float().reshape(-1).topk(2).values
            self.margins.append(round(float(t2[0] - t2[1]), 6))
            return out

    gen = Hashing(model, interp, schedule)
    reps_out = []
    for _ in range(reps):
        gen.hashes, gen.margins = [], []
        out = torch.zeros(1, ntok, device=model.device, dtype=torch.long)
        out[:, 0] = first
        t0 = time.time()
        gen.generate(out, plen, ntok - 1)
        torch.cuda.synchronize()
        reps_out.append(dict(token_ids=out[0].tolist(), logit_hashes=gen.hashes,
                             top2_margins=gen.margins, seconds=round(time.time() - t0, 3)))
    torch_ids = None
    if with_torch:
        out = torch.zeros(1, ntok, device=model.device, dtype=torch.long)
        out[:, 0] = first
        PyTorchGenerator(model).generate(out, plen, ntok - 1)
        torch_ids = out[0].tolist()
    return dict(prompt_length=plen, reps=reps_out, torch_token_ids=torch_ids)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hazy-root", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ntok", type=int, default=256)
    ap.add_argument("--no-torch", action="store_true", help="skip the torch reference decode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    p = torch.cuda.get_device_properties(0)
    gpu = dict(name=p.name, capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count)
    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    prompts = prompts[: args.limit]
    tok, model, interp, builder = build(Path(args.hazy_root), args.model)
    runs = []
    for i, pr in enumerate(prompts):
        t0 = time.time()
        r = run_prompt(tok, model, interp, builder, pr["prompt"], args.ntok, args.reps, not args.no_torch)
        for rep, x in enumerate(r["reps"]):
            runs.append(dict(schema="mkparity.run.v1", system="hazy", arm="mk", prompt_id=pr["prompt_id"],
                             task=pr.get("task"), rep=rep, model=args.model, gpu=gpu, returncode=0,
                             prompt_length=r["prompt_length"], **x))
        if r["torch_token_ids"] is not None:
            runs.append(dict(schema="mkparity.run.v1", system="hazy", arm="torch", prompt_id=pr["prompt_id"],
                             task=pr.get("task"), rep=0, model=args.model, gpu=gpu, returncode=0,
                             prompt_length=r["prompt_length"], token_ids=r["torch_token_ids"]))
        distinct = len({tuple(x["logit_hashes"]) for x in r["reps"]})
        print(f"[{i + 1}/{len(prompts)}] {pr['prompt_id']} distinct_hash_streams={distinct}/{args.reps} "
              f"{time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # one file per run keeps the format identical to the MPK runner
    for r in runs:
        (out.parent / out.stem).mkdir(exist_ok=True)
        (out.parent / out.stem / f"{r['arm']}__{r['prompt_id']}__r{r['rep']}.json").write_text(json.dumps(r))
    out.write_text(json.dumps(dict(gpu=gpu, model=args.model, n_runs=len(runs), args=vars(args)), indent=1))
    print(f"wrote {len(runs)} runs under {out.parent / out.stem}")


if __name__ == "__main__":
    main()
