#!/usr/bin/env python3
"""Export the browser decoder graphs from an AR checkpoint.

The original export died with the session that ran it, which left the shipped
graphs unreproducible — and K (the beam width) baked into the step graph's
batch dimension, because neither ONNX exporter handles the dynamic shapes this
model needs (the legacy tracer bakes the example batch into an attention
reshape; the dynamo exporter cannot guard TransformerDecoder's mask
detection). This script makes the export a recorded, repeatable step, and
parity-checks every graph against torch before writing anything permanent.

    python scripts/export_web_decoder.py \
        --checkpoint runs/ar_shape10/ar_decoder.pt --k 8 16 --out /tmp/export

Emits ar-encoder.onnx (batch 1) and ar-step-k{K}.onnx per requested K.
The step graph contract matches web/ar-decode.js: memory (K, 64, d) +
tokens int64 (K, 25) -> logp (K, 25, vocab), log-softmaxed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from swipe_typing.model.ar import ARConfig, ARSwipeDecoder  # noqa: E402


class EncoderGraph(torch.nn.Module):
    def __init__(self, model: ARSwipeDecoder):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)


class StepGraph(torch.nn.Module):
    def __init__(self, model: ARSwipeDecoder):
        super().__init__()
        self.model = model

    def forward(self, memory: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(
            self.model.decode_step(memory, tokens), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_shape10/ar_decoder.pt")
    ap.add_argument("--k", type=int, nargs="+", default=[8])
    ap.add_argument("--out", default="/tmp/web-export")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ARSwipeDecoder(ARConfig(**dict(ckpt["cfg"])))
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = model.cfg
    lmax = cfg.max_word_len + 1

    import onnxruntime as rt

    torch.manual_seed(0)
    x = torch.randn(1, 64, cfg.n_input)

    enc_path = out / "ar-encoder.onnx"
    # References computed BEFORE export: the legacy exporter leaves the model
    # in train mode, and a reference taken after runs with dropout live.
    with torch.no_grad():
        want = model.encode(x).numpy()
    torch.onnx.export(EncoderGraph(model), (x,), enc_path,
                      input_names=["x"], output_names=["memory"],
                      dynamo=False)
    model.eval()
    got = rt.InferenceSession(enc_path).run(
        None, {"x": x.numpy()})[0]
    err = float(np.abs(want - got).max())
    assert err < 1e-4, f"encoder parity {err}"
    print(f"{enc_path}  parity {err:.2e}")

    for k in args.k:
        memory = model.encode(x).detach().repeat(k, 1, 1)
        # Realistic prefixes: BOS then letters, zero-padded like the JS beam.
        tokens = torch.zeros(k, lmax, dtype=torch.long)
        tokens[:, 0] = cfg.bos
        tokens[:, 1:5] = torch.randint(0, cfg.n_keys, (k, 4))
        step_path = out / f"ar-step-k{k}.onnx"
        with torch.no_grad():
            want = torch.log_softmax(
                model.decode_step(memory, tokens), -1).numpy()
        torch.onnx.export(StepGraph(model), (memory, tokens), step_path,
                          input_names=["memory", "tokens"],
                          output_names=["logp"], dynamo=False)
        model.eval()
        got = rt.InferenceSession(step_path).run(
            None, {"memory": memory.numpy(),
                   "tokens": tokens.numpy()})[0]
        err = float(np.abs(want - got).max())
        assert err < 1e-4, f"step k={k} parity {err}"
        # A second, different prefix through the SAME graph: proves the
        # example inputs were not baked in.
        tokens2 = torch.zeros(k, lmax, dtype=torch.long)
        tokens2[:, 0] = cfg.bos
        tokens2[:, 1:9] = torch.randint(0, cfg.n_keys, (k, 8))
        with torch.no_grad():
            want2 = torch.log_softmax(
                model.decode_step(memory, tokens2), -1).numpy()
        got2 = rt.InferenceSession(step_path).run(
            None, {"memory": memory.numpy(), "tokens": tokens2.numpy()})[0]
        err2 = float(np.abs(want2 - got2).max())
        assert err2 < 1e-4, f"step k={k} reuse parity {err2}"
        print(f"{step_path}  parity {err:.2e} / reuse {err2:.2e}")


if __name__ == "__main__":
    main()
