"""Experiment #34: neural-LM rescoring of the frozen n-best lists on Modal.

Faithful port of scripts/eval_neural_rerank.py's scoring pass, for LMs too
large to run locally. The first pass (encoder + trie beam) stays local and
frozen — every model rescores the exact same n-best lists, shipped in the
pickle written by scripts/export_rerank_bundle.py. One GPU container per
model, run concurrently; the 9B fits an A10G in bf16.

Usage:
    modal run scripts/modal_rerank.py                    # default ladder
    modal run scripts/modal_rerank.py --models Qwen/Qwen3.5-9B-Base \
        --bundle rerank_bundle.pkl --out ladder_modal.log
"""

import pickle
from pathlib import Path

import modal

app = modal.App("swipe-lm-ladder")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("swipe-hf-cache", create_if_missing=True)

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B-Base",
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def rescore(model_name: str, bundle_bytes: bytes) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bundle = pickle.loads(bundle_bytes)
    nbest = bundle["nbest"]
    refs = bundle["refs"]
    groups = bundle["groups"]
    sentences = bundle["sentences"]
    word_idx = bundle["word_idx"]
    n = len(refs)

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = (AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
             .to(device).eval())
    hf_cache.commit()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    uncond_cache: dict[str, float] = {}

    def ctx_ids(context: str) -> torch.Tensor:
        if not context.strip():
            return torch.tensor([[bos]], device=device)
        ids = tok(context, return_tensors="pt").input_ids.to(device)
        return ids[:, -64:]

    @torch.no_grad()
    def score(context: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        ctx = ctx_ids(context)
        c_len = ctx.shape[1]
        tails = [tok(" " + w, return_tensors="pt").input_ids[0].to(device)
                 for w in candidates]
        max_t = max(len(t) for t in tails)
        m = len(candidates)
        inp = torch.full((m, c_len + max_t), tok.eos_token_id or 0,
                         dtype=torch.long, device=device)
        mask = torch.zeros((m, c_len + max_t), dtype=torch.long, device=device)
        inp[:, :c_len] = ctx
        mask[:, :c_len] = 1
        for i, t in enumerate(tails):
            inp[i, c_len:c_len + len(t)] = t
            mask[i, c_len:c_len + len(t)] = 1
        logits = model(input_ids=inp, attention_mask=mask).logits
        logprobs = F.log_softmax(logits.float(), dim=-1)
        out = []
        for i, t in enumerate(tails):
            step = logprobs[i, c_len - 1:c_len - 1 + len(t)]
            out.append(float(step.gather(1, t[:, None]).sum()))
        return out

    def unconditional(words: list[str]) -> list[float]:
        todo = [w for w in words if w not in uncond_cache]
        if todo:
            for w, s in zip(todo, score("", todo)):
                uncond_cache[w] = s
        return [uncond_cache[w] for w in words]

    base = sum(bool(b) and b[0][0] == refs[i] for i, b in enumerate(nbest)) / n
    ceiling = sum(refs[i] in [w for w, _ in b] for i, b in enumerate(nbest)) / n

    cond_gt: dict[int, list[float]] = {}
    uncond: dict[int, list[float]] = {}
    for idx in groups:
        raw_words = sentences[idx[0]].split() if idx else []
        for pos, i in enumerate(idx):
            cands = [w for w, _ in nbest[i]]
            if not cands:
                continue
            k = word_idx[i] if word_idx[i] >= 0 else pos
            prefix = " ".join(raw_words[:k])
            cond_gt[i] = score(prefix, cands)
            uncond[i] = unconditional(cands)

    def sweep(use_delta: bool, weight: float) -> float:
        hit = 0
        for i, b in enumerate(nbest):
            if not b:
                continue
            lm_s = cond_gt[i]
            if use_delta:
                lm_s = [a - u for a, u in zip(lm_s, uncond[i])]
            j = max(range(len(b)), key=lambda k: b[k][1] + weight * lm_s[k])
            hit += b[j][0] == refs[i]
        return hit / n

    table = {}
    best = (base, 0.0, False)
    for weight in WEIGHTS:
        raw, dlt = sweep(False, weight), sweep(True, weight)
        table[weight] = (raw, dlt)
        for val, is_d in ((raw, False), (dlt, True)):
            if val > best[0]:
                best = (val, weight, is_d)
    _, weight, use_delta = best

    decoded = None
    if weight > 0.0:
        dec = 0
        for idx in groups:
            emitted: list[str] = []
            for i in idx:
                b = nbest[i]
                if not b:
                    continue
                cands = [w for w, _ in b]
                s_ = score(" ".join(emitted), cands)
                if use_delta:
                    s_ = [a - u for a, u in zip(s_, uncond[i])]
                j = max(range(len(b)), key=lambda k: b[k][1] + weight * s_[k])
                dec += b[j][0] == refs[i]
                emitted.append(b[j][0])
        decoded = dec / n

    return {
        "model": model_name,
        "base": base,
        "ceiling": ceiling,
        "table": table,
        "best_oracle": best[0],
        "best_weight": weight,
        "best_delta": use_delta,
        "decoded": decoded,
    }


@app.local_entrypoint()
def main(models: str = "", bundle: str = "rerank_bundle.pkl",
         out: str = "ladder_modal.log"):
    names = [m for m in models.split(",") if m] or DEFAULT_MODELS
    bundle_bytes = Path(bundle).read_bytes()

    out_path = Path(out)
    with open(out_path, "a") as out:
        for r in rescore.map(names, kwargs={"bundle_bytes": bundle_bytes}):
            lines = [f"== futo/validation  n=5000  LM={r['model']} (modal) ==",
                     f"  beam top-1 {r['base']:.4f}   ceiling {r['ceiling']:.4f}",
                     f"  {'weight':>7}{'raw logP':>11}{'delta':>9}"]
            for w, (raw, dlt) in sorted(r["table"].items()):
                lines.append(f"  {w:>7.2f}{raw:>11.4f}{dlt:>9.4f}")
            lines.append(
                f"  best oracle {r['best_oracle']:.4f} at weight "
                f"{r['best_weight']} ({'delta' if r['best_delta'] else 'raw'})")
            if r["decoded"] is not None:
                lines.append(
                    f"  decoded context: {r['decoded']:.4f} "
                    f"({r['decoded'] - r['base']:+.4f}), against "
                    f"{r['ceiling'] - r['base']:+.4f} available")
            block = "\n".join(lines)
            print(block, flush=True)
            out.write(block + "\n\n")
            out.flush()
