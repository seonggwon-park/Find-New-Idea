"""Idea 7 (S02) synthetic check: posterior-expected-mass ranking vs point-estimate
ranking for top-k selection from noisy (quantized-proxy) attention logits.

Model: true logits s_i ~ prior; proxy s_hat_i = s_i + sigma_i * eps_i, eps iid N(0,1),
sigma_i heteroscedastic (lognormal dispersion, known to the selector).
Arms:
  naive : top-k by s_hat
  flat  : top-k by s_hat + sigma^2/2        (flat-prior posterior E e^s)
  jensen: top-k by s_hat - sigma^2/2        (published dense-attention Jensen debias, 2605.26266, used as a ranker)
  gEB   : top-k by m + v/2, Gaussian prior fitted per query by moments
  npEB  : top-k by log E[e^s | s_hat] under a per-query NPMLE (grid EM) prior
Metrics: captured true softmax mass of the selected set, recall vs exact top-k,
gap recovery = (mass_arm - mass_naive) / (mass_exact - mass_naive).
Claims tested (sharpened hypothesis):
  C1 crossover: (m+v/2) - s_hat changes sign at s_hat = mu + tau^2/2.
  C2 gain ~0 for sigma^2/tau^2 ~ 1e-3 (FP8-like) and grows with sigma^2/tau^2.
  C3 direction of the swap (demote vs promote noisy tokens) depends on whether the
     budget boundary lies above or below mu + tau^2/2.
  C4 at 1-bit-like noise, posterior ranking recovers >= 30% of the naive-to-exact gap.
  C5 heavy-tailed logits: Gaussian EB can be worse than NPMLE EB (over-shrinkage).
Seeds fixed. CPU runtime well under 2 minutes.
"""
import numpy as np

N = 4096
BUDGETS = [0.01, 0.05, 0.10, 0.20]


def sample_prior(rng, kind, n):
    if kind.startswith("gauss"):
        tau = float(kind.split("_")[1])
        return tau * rng.standard_normal(n), tau
    if kind == "heavy":  # bulk N(0,1) + 2% spikes N(5,1.5^2): heavy-hitter-like upper tail
        s = rng.standard_normal(n)
        m = rng.random(n) < 0.02
        s[m] = 5.0 + 1.5 * rng.standard_normal(m.sum())
        return s, float(np.std(s))
    raise ValueError(kind)


def sample_sigma(rng, n, sig0, disp):
    return sig0 * np.exp(disp * rng.standard_normal(n) - disp ** 2)  # E[sigma^2] = sig0^2


def mass_of(sel, s):
    w = np.exp(s - s.max())
    return w[sel].sum() / w.sum()


def gauss_eb_score(sh, sig):
    mu = sh.mean()
    tau2 = max(sh.var() - np.mean(sig ** 2), 1e-6)
    m = (tau2 * sh + sig ** 2 * mu) / (tau2 + sig ** 2)
    v = tau2 * sig ** 2 / (tau2 + sig ** 2)
    return m + v / 2, mu, tau2


def npmle_score(sh, sig, G=120, iters=80):
    lo, hi = sh.min() - 1.0, sh.max() + 1.0
    grid = np.linspace(lo, hi, G)
    # likelihood matrix L[i,g] = N(sh_i; grid_g, sig_i^2)
    z = (sh[:, None] - grid[None, :]) / sig[:, None]
    L = np.exp(-0.5 * z ** 2) / sig[:, None]
    w = np.full(G, 1.0 / G)
    for _ in range(iters):
        P = L * w[None, :]
        P /= P.sum(1, keepdims=True) + 1e-300
        w = P.mean(0)
    P = L * w[None, :]
    P /= P.sum(1, keepdims=True) + 1e-300
    return np.log(P @ np.exp(grid - hi) + 1e-300)  # log E[e^s|sh] up to a constant


def topk(score, k):
    return np.argpartition(-score, k)[:k]


def run(kind, ratio, disp, Q, rng, use_np=False):
    res = {b: {a: [] for a in ["naive", "flat", "jensen", "gEB", "npEB", "exact"]} for b in BUDGETS}
    rec = {b: {a: [] for a in ["naive", "flat", "jensen", "gEB", "npEB"]} for b in BUDGETS}
    for _ in range(Q):
        s, tau = sample_prior(rng, kind, N)
        sig = sample_sigma(rng, N, np.sqrt(ratio) * tau, disp)
        sh = s + sig * rng.standard_normal(N)
        sc = {"naive": sh, "flat": sh + sig ** 2 / 2, "jensen": sh - sig ** 2 / 2, "gEB": gauss_eb_score(sh, sig)[0]}
        if use_np:
            sc["npEB"] = npmle_score(sh, sig)
        for b in BUDGETS:
            k = int(b * N)
            ex = topk(s, k)
            res[b]["exact"].append(mass_of(ex, s))
            exset = set(ex.tolist())
            for a, v in sc.items():
                sel = topk(v, k)
                res[b][a].append(mass_of(sel, s))
                rec[b][a].append(len(exset.intersection(sel.tolist())) / k)
    return res, rec


def summarize(tag, res, rec, arms):
    for b in BUDGETS:
        mn = np.mean(res[b]["naive"]); me = np.mean(res[b]["exact"])
        line = f"{tag} budget={b:.2f} mass exact={me:.4f} naive={mn:.4f}"
        for a in arms:
            ma = np.mean(res[b][a])
            gap = (ma - mn) / (me - mn) if me - mn > 1e-9 else float("nan")
            line += f" | {a}: mass={ma:.4f} dmass_pp={100*(ma-mn):+.3f} gaprec={gap:+.3f} recall={np.mean(rec[b][a]):.3f}"
        line += f" | naive recall={np.mean(rec[b]['naive']):.3f}"
        print(line)


def main():
    rng = np.random.default_rng(20261011)
    print("=== C1 crossover (algebra, Gaussian prior mu=0,tau=3) ===")
    mu, tau2 = 0.0, 9.0
    for shv in [3.0, 4.0, 4.5, 5.0, 7.0]:
        for s2 in [0.5, 4.0]:
            m = (tau2 * shv + s2 * mu) / (tau2 + s2); v = tau2 * s2 / (tau2 + s2)
            print(f"s_hat={shv:4.1f} sigma2={s2:3.1f} (m+v/2)-s_hat={m+v/2-shv:+.4f}  closed form={s2/(tau2+s2)*(mu-shv+tau2/2):+.4f}")

    print("\n=== C2/C4 Gaussian prior, sweep sigma^2/tau^2 (disp=0.5 lognormal), Q=150 ===")
    for tau in [1.0, 3.0]:
        for ratio in [0.001, 0.01, 0.05, 0.2, 1.0]:
            res, rec = run(f"gauss_{tau}", ratio, 0.5, 150, rng)
            summarize(f"tau={tau} r={ratio}", res, rec, ["flat", "jensen", "gEB"])

    print("\n=== Control: homoscedastic sigma (disp=0), tau=1, r=0.2: gEB must equal naive ===")
    res, rec = run("gauss_1.0", 0.2, 0.0, 100, rng)
    summarize("homo tau=1 r=0.2", res, rec, ["gEB"])

    print("\n=== C3 direction of swaps: mean sigma of selected set, naive minus gEB ===")
    for tau in [1.0, 3.0]:
        for b in BUDGETS:
            k = int(b * N); dsig = []; bnd = []
            for _ in range(100):
                s, _ = sample_prior(rng, f"gauss_{tau}", N)
                sig = sample_sigma(rng, N, np.sqrt(0.2) * tau, 0.5)
                sh = s + sig * rng.standard_normal(N)
                g, muh, t2 = gauss_eb_score(sh, sig)
                dsig.append(sig[topk(sh, k)].mean() - sig[topk(g, k)].mean())
                bnd.append(np.sort(sh)[-k] - (muh + t2 / 2))
            print(f"tau={tau} budget={b:.2f} boundary-(mu+tau2/2)={np.mean(bnd):+.3f}  "
                  f"meansig(naive)-meansig(gEB)={np.mean(dsig):+.4f}  "
                  f"(predicted sign {'+' if np.mean(bnd)>0 else '-'})")

    print("\n=== C5 heavy-tailed logits (bulk N(0,1) + 2% spikes N(5,1.5^2)), NPMLE, Q=30 ===")
    for ratio in [0.01, 0.2, 1.0]:
        res, rec = run("heavy", ratio, 0.5, 30, rng, use_np=True)
        summarize(f"heavy r={ratio}", res, rec, ["jensen", "gEB", "npEB"])
    print("\n=== NPMLE sanity on Gaussian tau=1 r=1.0, Q=30 ===")
    res, rec = run("gauss_1.0", 1.0, 0.5, 30, rng, use_np=True)
    summarize("gauss tau=1 r=1.0", res, rec, ["gEB", "npEB"])


if __name__ == "__main__":
    main()
