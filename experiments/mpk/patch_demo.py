"""Runtime patch for MPK's demo/qwen3/demo.py.

The upstream demo saves only the first 100 tokens of request 0 and always
recompiles the megakernel. The runner needs every generated token of every
request, and it needs to reuse a compiled kernel across prompts, so it writes
a patched copy of demo.py next to the original and runs that copy.

Each edit is an exact string replacement that must match once. If upstream
changes those lines, patching fails loudly instead of silently running the
unpatched demo.
"""
from __future__ import annotations

from pathlib import Path

PATCHED_NAME = "_mkparity_demo.py"

EDITS = [
    (
        "MAX_SAVE_TOKENS = 100\n",
        'MAX_SAVE_TOKENS = int(os.environ.get("MKP_MAX_SAVE_TOKENS", "1000000"))\n',
    ),
    (
        "        mpk.compile(output_dir=args.output_dir)\n",
        "        import time as _mkp_time\n"
        "        _mkp_t0 = _mkp_time.time()\n"
        '        _mkp_reuse = os.environ.get("MKP_REUSE_KERNEL_DIR")\n'
        "        if _mkp_reuse and os.path.isdir(_mkp_reuse) and any(\n"
        '                f.startswith("mpk_launcher_rank") for f in os.listdir(_mkp_reuse)):\n'
        "            mpk.load_mpk_kernel(output_dir=_mkp_reuse)\n"
        '            _mkp_kernel_source = "load"\n'
        "        else:\n"
        "            mpk.compile(output_dir=args.output_dir)\n"
        '            _mkp_kernel_source = "compile"\n'
        "        _mkp_kernel_seconds = _mkp_time.time() - _mkp_t0\n",
    ),
    (
        '                "mode": "mpk",\n'
        "            }\n"
        '            with open(save_path, "w") as f:\n',
        '                "mode": "mpk",\n'
        "            }\n"
        '            out["all_token_ids"] = [\n'
        "                tokens[r, prompt_lengths[r].item(): step[r].item() + 1].tolist()\n"
        "                for r in range(total_num_requests)]\n"
        '            out["num_workers"] = int(mpk.num_workers)\n'
        '            out["kernel_source"] = _mkp_kernel_source\n'
        '            out["kernel_seconds"] = float(_mkp_kernel_seconds)\n'
        '            out["run_time_ms"] = float(run_time)\n'
        '            with open(save_path, "w") as f:\n',
    ),
]


class PatchError(RuntimeError):
    pass


def patch_source(src: str) -> str:
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise PatchError(
                f"expected exactly one match for {old.splitlines()[0].strip()!r}, found {n}. "
                "The upstream demo changed; update experiments/mpk/patch_demo.py.")
        src = src.replace(old, new)
    return src


def write_patched_demo(mirage_root: str | Path) -> Path:
    demo = Path(mirage_root) / "demo" / "qwen3" / "demo.py"
    if not demo.exists():
        raise FileNotFoundError(f"{demo} not found; pass --mirage-root pointing at a mirage checkout")
    out = demo.with_name(PATCHED_NAME)
    out.write_text(patch_source(demo.read_text()))
    return out


if __name__ == "__main__":
    import sys

    print(write_patched_demo(sys.argv[1] if len(sys.argv) > 1 else "."))
