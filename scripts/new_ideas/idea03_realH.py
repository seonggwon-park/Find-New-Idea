"""Decisive experiment for new idea 3 on real LLM Hessians (scaled-down CPU proxy).

High-rate entropy-coded model (Ordentlich & Polyanskiy 2605.13768):
  rate gap of a GPTQ variant vs WaterSIC = 1/2 log2( AM(D) / GM(lambda) ),
  D = LDL / Cholesky pivots of the (damped) Hessian in the basis GPTQ works in.

Per linear-input Hessian H = X^T X / m (q/k/v share one, o, gate/up share one, down):
  gap_rot_meas : Haar-rotated GPTQ, exact pivots of Q (H + delta I) Q^T (one Haar draw)
  gap_eq33     : Eq. 33 prediction from the spectrum alone (D_k ~= E_k/E_{k-1})
  gap_native   : GPTQ in the original basis (no rotation)
  gap_pca      : GPTQ in the eigenbasis (D = lambda), = 1/2 log2 AM/GM
  eps_eff      : (lambda_min + delta) / (lambda_max + delta)
  bound_2pt    : the idea's asymptotic worst case over spectra in [eps_eff, 1]
Decision rules (from the report): Spearman(gap_eq33, gap_rot_meas) >= 0.7 and |diff| <= 0.05 bit;
practical relevance needs some layers with gap > 0.3 bit at 1e-2 damping.

Usage: python -I idea03_realH.py --model Qwen/Qwen2.5-0.5B --nseq 64 --seqlen 1024 --starved 512 --out out.json
"""
import argparse, json, math, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)


def log_esm(lam):
    n = len(lam)
    L = np.full(n + 1, -np.inf)
    L[0] = 0.0
    ll = np.log(lam)
    for j in range(n):
        L[1:j + 2] = np.logaddexp(L[1:j + 2], ll[j] + L[0:j + 1])
    return L


def eq33_pivots(lam):
    n = len(lam)
    k = np.arange(n + 1)
    from scipy.special import gammaln
    logC = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return np.exp(np.diff(log_esm(lam) - logC))


def gap(D, lam):
    return 0.5 * math.log2(np.mean(D) / math.exp(np.mean(np.log(lam))))


def bound_2pt(eps):
    # idea 3 asymptotic: 1/2 log2(1/eps) - log2 ln(1/eps) + 1/2 log2(2/e^2)
    if eps >= 1e-2:
        return float("nan")
    L = math.log(1 / eps)
    return 0.5 * math.log2(1 / eps) - math.log2(L) + 0.5 * math.log2(2 / math.e ** 2)


def haar(n, rng):
    Z = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    return Q * np.sign(np.diag(R))


def collect(model, wins):
    acc = {}
    hooks = []

    def mk(name):
        def f(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            if name not in acc:
                acc[name] = [torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64), 0]
            acc[name][0] += x.T @ x
            acc[name][1] += x.shape[0]
        return f

    for li, layer in enumerate(model.model.layers):
        for nm, mod in (("qkv", layer.self_attn.q_proj), ("o", layer.self_attn.o_proj),
                        ("gateup", layer.mlp.gate_proj), ("down", layer.mlp.down_proj)):
            hooks.append(mod.register_forward_hook(mk(f"{li}.{nm}")))
    for w in wins:
        model.model(input_ids=w)
    for h in hooks:
        h.remove()
    return {k: (v[0] / v[1]).numpy() for k, v in acc.items()}, sum(w.numel() for w in wins)


def analyze(H, deltas, rng):
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    lam0 = np.clip(np.linalg.eigvalsh(H), 0, None)
    mean = float(np.mean(np.diag(H)))
    Q = haar(n, rng)
    HR = Q @ H @ Q.T
    HR = 0.5 * (HR + HR.T)
    out = []
    for dl in deltas:
        d = dl * mean
        lam = lam0 + d
        I = np.eye(n) * d
        Dr = np.diag(np.linalg.cholesky(HR + I)) ** 2
        Dn = np.diag(np.linalg.cholesky(H + I)) ** 2
        e = float(lam.min() / lam.max())
        out.append(dict(delta=dl, n=n, eps_eff=e, gap_rot_meas=gap(Dr, lam), gap_eq33=gap(eq33_pivots(lam), lam),
                        gap_native=gap(Dn, lam), gap_pca=gap(lam, lam), bound_2pt=bound_2pt(e),
                        zero_eigs=int(np.sum(lam0 < 1e-12 * lam0.max()))))
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--nseq", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--starved", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    from datasets import load_dataset
    text = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    rng = np.random.default_rng(0)
    starts = rng.choice(len(ids) - a.seqlen - 1, size=a.nseq, replace=False)
    full = [ids[s:s + a.seqlen].unsqueeze(0) for s in starts]
    starved = [ids[starts[0]:starts[0] + a.starved].unsqueeze(0)]
    deltas = [1e-4, 1e-2, 1e-1]
    rows = []
    for cond, wins in (("full", full), ("starved", starved)):
        t1 = time.time()
        Hs, ntok = collect(model, wins)
        print(f"[{cond}] collected {len(Hs)} Hessians from {ntok} tokens in {time.time()-t1:.0f}s", flush=True)
        for name, H in Hs.items():
            t2 = time.time()
            for r in analyze(H, deltas, rng):
                r.update(cond=cond, name=name, layer=int(name.split(".")[0]), kind=name.split(".")[1], tokens=ntok)
                rows.append(r)
            print(f"  {cond} {name} n={H.shape[0]} " + " ".join(
                f"d={r['delta']:.0e}:meas={r['gap_rot_meas']:.3f}/eq33={r['gap_eq33']:.3f}/nat={r['gap_native']:.3f}/pca={r['gap_pca']:.3f}/eps={r['eps_eff']:.1e}"
                for r in rows[-3:]) + f" ({time.time()-t2:.0f}s)", flush=True)
    summ = {}
    for cond in ("full", "starved"):
        for dl in deltas:
            R = [r for r in rows if r["cond"] == cond and r["delta"] == dl]
            m = np.array([r["gap_rot_meas"] for r in R]); p = np.array([r["gap_eq33"] for r in R])
            b = np.array([r["bound_2pt"] for r in R])
            summ[f"{cond}_{dl:.0e}"] = dict(
                n_hessians=len(R), spearman_eq33_vs_meas=spearman(p, m), max_abs_diff=float(np.max(np.abs(p - m))),
                mean_abs_diff=float(np.mean(np.abs(p - m))), gap_meas_median=float(np.median(m)), gap_meas_max=float(m.max()),
                frac_gap_gt_0_3=float(np.mean(m > 0.3)), frac_gap_gt_0_1=float(np.mean(m > 0.1)),
                native_median=float(np.median([r["gap_native"] for r in R])), pca_median=float(np.median([r["gap_pca"] for r in R])),
                bound_over_meas_median=float(np.nanmedian(b / m)) if np.any(~np.isnan(b)) else None,
                eps_eff_median=float(np.median([r["eps_eff"] for r in R])))
    print("SUMMARY:", json.dumps(summ, indent=1), flush=True)
    if a.out:
        json.dump(dict(args=vars(a), rows=rows, summary=summ), open(a.out, "w"), indent=1)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
