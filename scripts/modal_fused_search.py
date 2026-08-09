"""Fused sentence-beam search on Modal: one search instead of three passes.

    modal run scripts/modal_fused_search.py --bundle fused_bundle.pkl

Replaces n-best truncation -> lattice rerank -> deferred-commitment layering
with a single sentence-level beam: states carry (word sequence, score), each
swipe extends every live state through its deep candidate list with the
context LM scoring completions in-search, and commitment is just a lag on
pruning to the best state's prefix:

    lag 0   = streaming (commit each word before the next swipe)
    lag 1   = lookahead-1 (previous word may silently revise once)
    lag inf = joint (any word may revise until sentence end)

Per candidate the acoustic side is  ar + beta*len + prior, where prior is
alpha*unigram (the stack's current form, lam=0) or -lam*ILM (HAT-style
internal-LM subtraction; the external LM then owns the prior exactly once).
One container per (prior, mu, M) config; the three lags share its LM cache.
"""

import pickle
from pathlib import Path

import modal

app = modal.App("swipe-fused-search")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("swipe-hf-cache", create_if_missing=True)

BETA = 1.2
ALPHA = 0.4
BEAM = 8


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def fused_search(cfg: dict, bundle_bytes: bytes) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lam, mode, mu, M = cfg["lam"], cfg["mode"], cfg["mu"], cfg["M"]
    bundle = pickle.loads(bundle_bytes)
    lists = bundle["lists"]
    ilm = bundle["ilm"].get(mode, {})
    refs = bundle["refs"]
    groups = bundle["groups"]
    n = len(refs)

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(cfg["lm"])
    tok.pad_token = tok.eos_token
    lm = (AutoModelForCausalLM.from_pretrained(cfg["lm"], dtype="auto")
          .to(device).eval())
    hf_cache.commit()
    bos = tok.bos_token_id or tok.eos_token_id

    cache: dict[tuple[str, str], float] = {}

    @torch.no_grad()
    def lm_scores(ctx: str, words: list[str]) -> list[float]:
        todo = [w for w in words if (ctx, w) not in cache]
        if todo:
            if ctx.strip():
                ids = tok(ctx, return_tensors="pt").input_ids.to(device)
                ids = ids[:, -64:]
            else:
                ids = torch.tensor([[bos]], device=device)
            c_len = ids.shape[1]
            tails = [tok(" " + w, return_tensors="pt").input_ids[0].to(device)
                     for w in todo]
            max_t = max(len(t) for t in tails)
            m = len(todo)
            inp = torch.full((m, c_len + max_t), tok.eos_token_id,
                             dtype=torch.long, device=device)
            mask = torch.zeros_like(inp)
            inp[:, :c_len] = ids
            mask[:, :c_len] = 1
            for i, t in enumerate(tails):
                inp[i, c_len:c_len + len(t)] = t
                mask[i, c_len:c_len + len(t)] = 1
            logits = lm(input_ids=inp, attention_mask=mask).logits
            lp = F.log_softmax(logits.float(), dim=-1)
            for i, t in enumerate(tails):
                step = lp[i, c_len - 1:c_len - 1 + len(t)]
                cache[(ctx, todo[i])] = float(step.gather(1, t[:, None]).sum())
        return [cache[(ctx, w)] for w in words]

    def acoustic(cl):
        out = []
        for w, ar, uni, ln in cl:
            prior = ALPHA * uni if lam == 0.0 else -lam * ilm.get(w, 0.0)
            out.append((w, ar + BETA * ln + prior))
        return out

    delta = bool(cfg.get("delta"))

    def decode(idx: list[int], lag) -> list[str]:
        states: list[tuple[tuple, float]] = [((), 0.0)]
        n_words = len(idx)
        for t, i in enumerate(idx):
            cl = acoustic(lists[i][:M])
            if cl:
                expansions: dict[tuple, float] = {}
                for words, cum in states:
                    ctx = " ".join(words)
                    ls = lm_scores(ctx, [w for w, _ in cl])
                    if delta:
                        # The stack's #10 form: context evidence only, the
                        # LM's own unigram removed.
                        unc = lm_scores("", [w for w, _ in cl])
                        ls = [a - u for a, u in zip(ls, unc)]
                    for (w, ac), l in zip(cl, ls):
                        wt = words + (w,)
                        sc = cum + ac + mu * l
                        if wt not in expansions or sc > expansions[wt]:
                            expansions[wt] = sc
                states = sorted(expansions.items(), key=lambda kv: -kv[1])[:BEAM]
            else:
                states = [(words + ("",), cum) for words, cum in states]
            if lag is not None and t - lag >= 0:
                j = t - lag
                w_commit = states[0][0][j]
                states = [s for s in states if s[0][j] == w_commit] or states[:1]
        return list(states[0][0]) + [""] * (n_words - len(states[0][0]))

    results = {}
    for lag, name in [(0, "streaming"), (1, "lookahead1"), (None, "joint")]:
        hit = 0
        for idx in groups:
            out = decode(idx, lag)
            hit += sum(w == refs[i] for w, i in zip(out, idx))
        results[name] = hit / n

    ceiling = sum(refs[i] in [w for w, *_ in lists[i][:M]]
                  for i in range(n)) / n
    return {**cfg, "results": results, "ceiling": ceiling,
            "lm_calls": len(cache)}


@app.local_entrypoint()
def main(bundle: str = "fused_bundle.pkl", lm: str = "gpt2",
         out: str = "fused_search_modal.log", lambdas: str = "0.0,0.2,0.4",
         modes: str = "zero,mean", mus: str = "0.4,0.8,1.2",
         ms: str = "8,24", delta: bool = False):
    bundle_bytes = Path(bundle).read_bytes()

    lams = [float(x) for x in lambdas.split(",")]
    mode_list = modes.split(",")
    configs = []
    priors = ([(0.0, "uni")] if 0.0 in lams else []) + [
        (lam, mode) for lam in lams if lam > 0 for mode in mode_list
    ]
    for lam, mode in priors:
        for mu in (float(x) for x in mus.split(",")):
            for M in (int(x) for x in ms.split(",")):
                configs.append({"lam": lam, "mode": mode, "mu": mu, "M": M,
                                "lm": lm, "delta": delta})
    print(f"{len(configs)} configs, one A10G each")

    rows = []
    with open(out, "a") as f:
        for r in fused_search.map(configs,
                                  kwargs={"bundle_bytes": bundle_bytes}):
            line = (f"lam={r['lam']:<4} mode={r['mode']:<5} mu={r['mu']:<4} "
                    f"M={r['M']:<3} ceil={r['ceiling']:.4f}  "
                    f"stream={r['results']['streaming']:.4f}  "
                    f"look1={r['results']['lookahead1']:.4f}  "
                    f"joint={r['results']['joint']:.4f}  "
                    f"({r['lm_calls']:,} lm calls)")
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
            rows.append(r)

    best = max(rows, key=lambda r: r["results"]["joint"])
    print(f"\nbest joint: {best['results']['joint']:.4f}  "
          f"(lam={best['lam']}, mode={best['mode']}, mu={best['mu']}, "
          f"M={best['M']})")
