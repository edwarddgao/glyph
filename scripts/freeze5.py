#!/usr/bin/env python3
"""Freeze five (#84): one configuration, one script, every number the headline quotes.

    python scripts/freeze5.py --stages train,mmi,eval,export,fused --reads val
    python scripts/freeze5.py --post-prediction "test joint 95.1-95.3, hws 83-84"
    python scripts/freeze5.py --stages export,fused --reads test,hws
    python scripts/freeze5.py --stages report

The configuration is the dict below and nothing else. Freeze four (#53) was
the AR-MMI encoder (#46/#48) -> 24-deep trie beam -> fused sentence beam with
delta-form gpt2-xl at mu=0.8, joint commitment. Freeze five keeps that
recipe and adds the three post-freeze levers that cleared validation:

  * #82  the encoder trains on `futo_clean/train` — FUTO minus the 1.6% of
         gestures that do not trace their label (#81's decoder-independent
         filter). Null in-domain, +1.0 on How We Swipe.
  * #78  the prior algebra: external unigram at alpha=0.6 with the
         encoder's internal LM subtracted at lambda=0.25 (mean-memory
         ablation). Null in-domain, +0.6 on How We Swipe, replicated on a
         disjoint slice.
  * #66  the LM's own prior estimated over neutral contexts (`marginal`)
         rather than the start token — a no-op for gpt2-xl (-0.01), kept
         because it is the corrected convention.

Not in the freeze, deliberately: the geometry channel (#73) — its in-domain
weight (gamma=0.1, +0.09 n.s.) and its cross-corpus weight (0.5, +2.1)
differ, and it has never been composed with #78/#82; the mixed FUTO+HWS
encoder (#82b) — it trains on 70% of How We Swipe's users, which would
change what the cross-corpus row means. Both are the next experiments, not
part of this anchor.

Stages are idempotent: each skips when its output exists, so a killed queue
resumes where it stopped. The test and hws reads refuse to run until a
prediction has been posted with --post-prediction — the protocol every
freeze has followed (#19/#28/#35/#53) — and refuse to run twice.

Everything lands under runs/freeze5/: n-best dumps, the MMI checkpoint,
bundles, fused logs, hypothesis arrays, and manifest.json with the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "runs" / "freeze5"

FREEZE = {
    # encoder: the #82 clean-data AR decoder, seed 1, then one MMI epoch
    # over its own beam's lists (#48's recipe on the same training split)
    "train": {"train_path": "futo_clean/train", "seed": 1,
              "out": "runs/ar_clean_s1"},
    "mmi": {"train_split": "futo_clean/train", "train_limit": 150000,
            "val_split": "futo/validation", "val_limit": 20000,
            "epochs": 1, "lr": 5e-5, "out": "runs/freeze5/ar_clean_s1_mmi"},
    "checkpoint": "runs/freeze5/ar_clean_s1_mmi/ar_decoder_ep0.pt",
    # first pass: trie-constrained AR beam, deep lists for the fused search
    "lists": {"beam_width": 64, "max_cands": 64, "lexicon": "train+wf320k"},
    # fused sentence beam: run_fused_local.py, BETA=1.2 and BEAM=8 fixed there
    "fused": {"lm": "gpt2-xl", "mu": 0.8, "m": 24, "alpha": 0.6,
              "lam": 0.25, "mode": "mean", "uncond": "marginal",
              "delta": True, "lags": "0,1,joint"},
}

# name -> (split, limit). None = the whole split.
READS = {
    "val": ("futo/validation", 20000),       # the tuning slice; predicts the rest
    "test": ("futo/test", None),             # 48,711 swipes — read once
    "hws": ("how_we_swipe/test", None),      # 85,459 swipes — read once, zero-shot
}
ONCE = ("test", "hws")
# The cross-corpus row quotes joint; lookahead-1 is the shipping policy. The
# streaming pass costs a third of the read and is not quoted for hws.
LAGS = {"hws": "1,joint"}
ALL_SWIPES = 10**9


def log(msg: str) -> None:
    print(f"[freeze5 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], logfile: Path) -> None:
    """Run one stage command, tee-ing to a log; a non-zero exit stops the queue."""
    log(" ".join(cmd) + f"  > {logfile.relative_to(ROOT)}")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}")
    with open(logfile, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"stage failed (exit {rc}); see {logfile}")


def done(path: Path, marker: str | None = None) -> bool:
    if not path.exists():
        return False
    return marker is None or marker in path.read_text()


# --- stages -----------------------------------------------------------------

def stage_train() -> None:
    c = FREEZE["train"]
    ckpt = ROOT / c["out"] / "ar_decoder.pt"
    if done(ckpt):
        return log(f"train: {ckpt.relative_to(ROOT)} exists")
    run([PY, "scripts/train_ar_decoder.py", "--train-path", c["train_path"],
         "--seed", str(c["seed"]), "--out", c["out"]], OUT / "train.log")


def stage_mmi() -> None:
    c = FREEZE["mmi"]
    base = ROOT / FREEZE["train"]["out"] / "ar_decoder.pt"
    nbest = OUT / "nbest"
    for split, limit in ((c["train_split"], c["train_limit"]),
                         (c["val_split"], c["val_limit"])):
        npz = nbest / f"{split.replace('/', '_')}.npz"
        if done(npz):
            log(f"mmi: {npz.relative_to(ROOT)} exists")
            continue
        run([PY, "scripts/dump_ar_nbest.py", "--checkpoint", str(base),
             "--split", split, "--limit", str(limit), "--out", str(nbest),
             "--lexicon", FREEZE["lists"]["lexicon"]],
            OUT / f"dump_{split.replace('/', '_')}.log")
    ckpt = ROOT / FREEZE["checkpoint"]
    if done(ckpt):
        return log(f"mmi: {ckpt.relative_to(ROOT)} exists")
    run([PY, "scripts/finetune_ar_mmi.py", "--checkpoint", str(base),
         "--nbest", str(nbest),
         "--train", f"{c['train_split'].replace('/', '_')}.npz",
         "--val", f"{c['val_split'].replace('/', '_')}.npz",
         "--epochs", str(c["epochs"]), "--lr", str(c["lr"]),
         "--out", c["out"]], OUT / "mmi.log")


def stage_eval() -> None:
    """Beam eval of the base and MMI checkpoints on the #82 protocol
    (val 20k + hws 20k). #48's rule: judge MMI by beam top-1, never greedy."""
    for name, ckpt in (("base", ROOT / FREEZE["train"]["out"] / "ar_decoder.pt"),
                       ("mmi", ROOT / FREEZE["checkpoint"])):
        logfile = OUT / f"beam_eval_{name}.log"
        if done(logfile, "oracle over n-best"):
            log(f"eval: {logfile.relative_to(ROOT)} exists")
            continue
        run([PY, "scripts/eval_ar_decoder.py", "--checkpoint", str(ckpt),
             "--lexicon", FREEZE["lists"]["lexicon"]], logfile)


def bundle_path(read: str) -> Path:
    return OUT / f"bundle_{read}.pkl"


def stage_export(read: str) -> None:
    split, limit = READS[read]
    out = bundle_path(read)
    if done(out):
        return log(f"export {read}: {out.relative_to(ROOT)} exists")
    c = FREEZE["lists"]
    run([PY, "scripts/export_fused_bundle.py",
         "--checkpoint", FREEZE["checkpoint"], "--split", split,
         "--limit", str(ALL_SWIPES if limit is None else limit),
         "--beam-width", str(c["beam_width"]),
         "--max-cands", str(c["max_cands"]), "--lexicon", c["lexicon"],
         "--out", str(out)], OUT / f"export_{read}.log")


def stage_fused(read: str) -> None:
    logfile = OUT / f"fused_{read}.log"
    if done(logfile, "ceiling@"):
        return log(f"fused {read}: {logfile.relative_to(ROOT)} exists"
                   + (" — a once-only read is never repeated" if read in ONCE
                      else ""))
    if read in ONCE and not done(OUT / "prediction.json"):
        raise SystemExit(f"fused {read}: post a prediction first "
                         f"(--post-prediction) — the split is read once")
    c = FREEZE["fused"]
    cmd = [PY, "scripts/run_fused_local.py", "--bundle", str(bundle_path(read)),
           "--lm", c["lm"], "--mu", str(c["mu"]), "--m", str(c["m"]),
           "--alpha", str(c["alpha"]), "--lam", str(c["lam"]),
           "--mode", c["mode"], "--uncond", c["uncond"],
           "--lags", LAGS.get(read, c["lags"]),
           "--save-hyps", str(OUT / f"hyps_{read}.npz")]
    if c["delta"]:
        cmd.append("--delta")
    run(cmd, logfile)


# --- report -----------------------------------------------------------------

def parse_fused(logfile: Path) -> dict[str, float]:
    if not logfile.exists():
        return {}
    text = logfile.read_text()
    out = {k: float(v) for k, v in
           re.findall(r"mu=[\d.]+ (\w+): ([\d.]+)", text)}
    m = re.search(r"ceiling@\d+: ([\d.]+)", text)
    if m:
        out["ceiling"] = float(m.group(1))
    return out


def parse_beam(logfile: Path) -> dict[str, dict[str, float]]:
    if not logfile.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    split = None
    for line in logfile.read_text().splitlines():
        m = re.match(r"== (\S+)\s+n=", line)
        if m:
            split = m.group(1)
            out[split] = {}
        elif split and "greedy" in line:
            out[split]["greedy"] = float(line.split("top-1")[1])
        elif split and "beam top-1" in line:
            out[split]["beam"] = float(re.search(r": ([\d.]+)", line).group(1))
        elif split and "truth among surviving" in line:
            out[split]["in_list"] = float(re.search(r"candidates: ([\d.]+)",
                                                    line).group(1))
    return out


def stage_report() -> None:
    manifest = {
        "config": FREEZE, "reads": READS, "written": time.ctime(),
        "prediction": (json.loads((OUT / "prediction.json").read_text())
                       if (OUT / "prediction.json").exists() else None),
        "beam": {n: parse_beam(OUT / f"beam_eval_{n}.log")
                 for n in ("base", "mmi")},
        "fused": {r: parse_fused(OUT / f"fused_{r}.log") for r in READS},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nfreeze five  ({FREEZE['checkpoint']}; {FREEZE['fused']['lm']} "
          f"mu={FREEZE['fused']['mu']} M={FREEZE['fused']['m']} "
          f"alpha={FREEZE['fused']['alpha']} lam={FREEZE['fused']['lam']})")
    print("\n| stage | " + " | ".join(READS) + " |")
    print("|---|" + "---|" * len(READS))
    for stage in ("streaming", "lookahead1", "joint", "ceiling"):
        cells = [f"{manifest['fused'][r][stage]:.2%}"
                 if stage in manifest["fused"][r] else "—" for r in READS]
        print(f"| {stage} | " + " | ".join(cells) + " |")
    for n in ("base", "mmi"):
        for split, d in manifest["beam"][n].items():
            print(f"beam {n:<4} {split:<18} " +
                  "  ".join(f"{k} {v:.4f}" for k, v in d.items()))
    if manifest["prediction"]:
        print(f"\nprediction ({manifest['prediction']['posted']}): "
              f"{manifest['prediction']['text']}")
    print(f"\nwrote {OUT / 'manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="report",
                    help="comma list of train,mmi,eval,export,fused,report")
    ap.add_argument("--reads", default="val",
                    help=f"comma list of {','.join(READS)} for export/fused")
    ap.add_argument("--post-prediction", default=None,
                    help="record the predicted once-only numbers before "
                         "reading them; required before --reads test,hws")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.post_prediction:
        p = OUT / "prediction.json"
        if p.exists():
            raise SystemExit(f"{p} already posted: "
                             f"{json.loads(p.read_text())}")
        p.write_text(json.dumps({"text": args.post_prediction,
                                 "posted": time.ctime()}, indent=2))
        return log(f"prediction posted: {args.post_prediction}")

    reads = [r.strip() for r in args.reads.split(",")]
    for r in reads:
        if r not in READS:
            raise SystemExit(f"unknown read {r!r}; choose from {list(READS)}")
    for stage in [s.strip() for s in args.stages.split(",")]:
        if stage == "train":
            stage_train()
        elif stage == "mmi":
            stage_mmi()
        elif stage == "eval":
            stage_eval()
        elif stage == "export":
            for r in reads:
                stage_export(r)
        elif stage == "fused":
            for r in reads:
                stage_fused(r)
        elif stage == "report":
            stage_report()
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main()
