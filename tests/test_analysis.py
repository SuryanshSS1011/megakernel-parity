"""CPU tests for the analysis helpers, the MPK demo patcher, the MPK runner's
dry run and the summary script, using synthetic run files."""
import json
import os
import sys
from pathlib import Path

import pytest

from mkparity.analysis import compare_hashes, compare_streams, survival, wilson

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "mpk"))
sys.path.insert(0, str(REPO / "scripts"))
import analyze  # noqa: E402
import patch_demo  # noqa: E402
import run_mpk  # noqa: E402


def test_wilson_matches_known_values():
    lo, hi = wilson(32, 64)
    assert abs(lo - 0.3816) < 1e-3 and abs(hi - 0.6184) < 1e-3
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(64, 64)[1] == 1.0


def test_survival_counts_never_diverged():
    s = survival([-1, 10, 200, -1], length=256, points=(5, 64, 256))
    assert s == {5: 1.0, 64: 0.75, 256: 0.5}


def test_compare_streams():
    ref = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 9, 9, 9]}
    test = {"a": [1, 2, 3, 4], "b": [5, 0, 7, 8], "d": [1]}
    r = compare_streams(ref, test)
    assert r["prompts"] == 2 and r["exact"] == 1
    assert r["first_divergence"] == {"a": -1, "b": 1}


def test_compare_hashes_counts_steps_before_divergence():
    r = compare_hashes({"a": ["x", "y", "z"]}, {"a": ["x", "q", "z"]})
    assert r["steps"] == 3 and r["bitwise_equal_steps"] == 1


DEMO_FIXTURE = """import os, json
MAX_SAVE_TOKENS = 100

        mpk.compile(output_dir=args.output_dir)

            out = {
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
"""


def test_patch_applies_and_compiles():
    out = patch_demo.patch_source(DEMO_FIXTURE)
    assert "load_mpk_kernel" in out and "all_token_ids" in out and "MKP_MAX_SAVE_TOKENS" in out


def test_patch_fails_loudly_on_upstream_change():
    with pytest.raises(patch_demo.PatchError):
        patch_demo.patch_source(DEMO_FIXTURE.replace("MAX_SAVE_TOKENS = 100", "MAX_SAVE = 50"))


def test_real_demo_patches_if_present():
    root = os.environ.get("MIRAGE_ROOT")
    demo = Path(root or "") / "demo" / "qwen3" / "demo.py"
    if not root or not demo.exists():
        pytest.skip("set MIRAGE_ROOT to a mirage checkout to run this check")
    compile(patch_demo.patch_source(demo.read_text()), "demo", "exec")


def test_runner_dry_run_counts(tmp_path, capsys):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps(dict(prompt_id=f"p{i}", prompt=f"q{i}")) + "\n" for i in range(20)))
    rc = run_mpk.main(["--mirage-root", str(tmp_path), "--prompts", str(prompts), "--profile", "full",
                       "--dry-run", "--out", str(tmp_path / "out")])
    assert rc == 0
    # ns, spec_k3, spec_k5, ns_T16, ns_splitkv, torch: 20 each; ns_nocache 4; ns_rep 8x5; ns_B4 16
    assert "done=180 " in capsys.readouterr().out


def _run(arm, pid, tokens, rep=0, gpu="G1", system="mpk", **kw):
    return dict(schema="mkparity.run.v1", system=system, arm=arm, prompt_id=pid, rep=rep,
                model="M", gpu=dict(name=gpu), returncode=0, token_ids=tokens, **kw)


def test_analyze_end_to_end(tmp_path):
    runs = [
        _run("ns", "a", [1, 2, 3]), _run("ns", "b", [4, 5, 6]),
        _run("spec_k3", "a", [1, 2, 3]), _run("spec_k3", "b", [4, 9, 6]),
        _run("ns_B4", "a", [1, 2, 3], all_token_ids=[[1, 2, 3], [1, 2, 7]]),
        _run("ns", "a", [1, 2, 3], gpu="G2"), _run("ns", "b", [4, 5, 0], gpu="G2"),
        _run("mk", "a", [1, 2], system="hazy", logit_hashes=["h1", "h2"]),
        _run("mk", "a", [1, 2], rep=1, system="hazy", logit_hashes=["h1", "h3"]),
    ]
    for i, r in enumerate(runs):
        (tmp_path / f"{i}.json").write_text(json.dumps(r))
    rows, failures = analyze.analyze([json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.json"))])
    by = {r["name"]: r["result"] for r in rows}
    assert by["mpk G1 M: spec_k3 vs ns"]["exact"] == 1
    assert by["mpk G1 M: ns_B4 row 1 vs row 0"]["first_divergence"] == {"a": 2}
    assert by["mpk M: ns on G2 vs G1"]["exact"] == 1
    hz = by["hazy G1 M: mk r1 vs r0"]
    assert hz["exact"] == 1 and hz["logit_hashes"]["bitwise_equal_steps"] == 1
    assert failures == {}
