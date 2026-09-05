#!/usr/bin/env python3
"""Fetch Glyph's decoder resources from Hugging Face into keyboard/Resources.

    research/.venv/bin/python keyboard/tools/fetch_models.py [--repo edwarddgao/glyph-models]

The Core ML models (AR encoder + step, the distilgpt2 LM, the retired CTC
encoder the tests still check), the trie, the ILM and prior tables, the GPT-2
tokenizer files, the benchmark gestures and the test goldens — ~190 MB that do
not belong in git. `upload_models.py` is the inverse; `export_ar.py`,
`export_lm.py`, `export_priors.py`, `export_ilm.py` regenerate everything from
the research checkpoints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent.parent / "Resources"
HF_REPO = "edwarddgao/glyph-models"
PATTERNS = ["*.mlpackage/**", "lexicon.bin", "ilm.bin", "priors.bin", "gpt2/*", "bench_gestures.json", "goldens.json", "ar_goldens.json", "trace_goldens.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_REPO)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    RESOURCES.mkdir(exist_ok=True)
    path = snapshot_download(a.repo, repo_type="model", local_dir=str(RESOURCES), allow_patterns=PATTERNS)
    got = sorted(p.relative_to(RESOURCES) for p in Path(path).rglob("*") if p.is_file() and ".cache" not in p.parts)
    print(f"{len(got)} files in {RESOURCES}:")
    for p in got[:40]: print("  ", p)


if __name__ == "__main__":
    main()
