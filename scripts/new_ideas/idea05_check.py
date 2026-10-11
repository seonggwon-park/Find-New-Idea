#!/usr/bin/env python3
"""Idea 5 (X06) numeric check: quantized KL anchor = FP anchor + shaping -beta*eta.

eta(y) = sum_t [log pi_ref(y_t|s_t) - log pi_q(y_t|s_t)], y ~ pi (here pi = pi_ref, i.e. step 0).

Synthetic only. KL_tok values for "FP8-like" (1e-3) and "W4-like" (2e-2) are ASSUMPTIONS, not
measurements of real models. Checks:
  A1  Var(eta_t) ~= 2*KL_tok (small-perturbation law used in the kill test)
  A2  E[eta]/T constant (linear mean), sd(eta) growth exponent: independent vs sequence-correlated error
  A3  GRPO group (G=8) with Bernoulli(0.5) reward and variable lengths: size of the centered shaping
      term relative to the reward sd, split into a length-linear part and a residual part
  B   exact optimum of a stationary toy MDP (pi* = a*exp(g/beta) linearly solvable): mean length
      with an FP anchor vs a flattened (quantized) anchor
  C   Monte Carlo: expected k3-as-loss logit gradient equals pi_theta - pi_q (token-level forward KL
      toward the quantized anchor), i.e. no sequence-length factor
Runtime: well under 1 minute on CPU.
"""
import numpy as np

rng = np.random.default_rng(0)
V, NC, M = 2048, 1500, 256  # vocab, number of contexts, eta_t samples per context


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def logsoftmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


# reference contexts: mixture of peaky and flat token distributions
scales = rng.uniform(1.5, 6.0, size=(NC, 1))
L = scales * rng.standard_normal((NC, V))
EPS = rng.standard_normal((NC, V))  # fixed "quantization error" direction per context
lp_ref = logsoftmax(L)
p_ref = np.exp(lp_ref)


def quantize(s):
    # flattening (temperature) + logit noise, both scaled by s
    return logsoftmax(L / (1.0 + 0.5 * s) + s * EPS)


def kl_tok(lp_q):
    return (p_ref * (lp_ref - lp_q)).sum(1)  # KL(pi_ref || pi_q) per context


def calibrate(target):
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if kl_tok(quantize(mid)).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eta_pool(lp_q):
    # per context, M samples of eta_t = log p_ref(y) - log p_q(y), y ~ p_ref
    cum = np.cumsum(p_ref, axis=1)
    u = rng.random((NC, M))
    idx = np.empty((NC, M), dtype=np.int64)
    for c in range(NC):
        idx[c] = np.minimum(np.searchsorted(cum[c], u[c]), V - 1)
    d = lp_ref - lp_q
    return np.take_along_axis(d, idx, axis=1)


print("=== A1: token-level law Var(eta_t) ~= 2 KL_tok ===")
formats = {"FP8-like": 1e-3, "W4-like": 2e-2}
pools, pools_hard, ktoks = {}, {}, {}
for name, tgt in formats.items():
    s = calibrate(tgt)
    lq = quantize(s)
    k = kl_tok(lq)
    pool = eta_pool(lq)
    lq_h = quantize(2.0 * s)  # "outlier-sensitive" contexts: 2x perturbation
    pools[name], pools_hard[name], ktoks[name] = pool, eta_pool(lq_h), k.mean()
    var_t = pool.var()
    print(f"{name:9s} s={s:.4f} KL_tok(mean)={k.mean():.4g}  E[eta_t]={pool.mean():.4g}  "
          f"Var(eta_t)={var_t:.4g}  Var/(2KL)={var_t / (2 * k.mean()):.3f}  sd(eta_t)={np.sqrt(var_t):.4f}")


def sample_eta(name, T, n, correlated):
    pool, pool_h = pools[name], pools_hard[name]
    if correlated:
        w = rng.beta(0.5, 2.0, size=(n, 1))  # per-sequence share of outlier-sensitive contexts
        hard = rng.random((n, T)) < w
    else:
        hard = rng.random((n, T)) < 0.2
    c = rng.integers(0, NC, size=(n, T))
    j = rng.integers(0, M, size=(n, T))
    x = np.where(hard, pool_h[c, j], pool[c, j])
    return x.sum(1)


print("\n=== A2: growth of mean and sd of eta with T (n=400 seqs per T) ===")
Ts = [512, 1024, 2048, 4096, 8192]
for name in formats:
    for corr in (False, True):
        means, sds = [], []
        for T in Ts:
            e = sample_eta(name, T, 400, corr)
            means.append(e.mean())
            sds.append(e.std())
        slope_sd = np.polyfit(np.log(Ts), np.log(sds), 1)[0]
        slope_mean = np.polyfit(np.log(Ts), np.log(means), 1)[0]
        tag = "correlated " if corr else "independent"
        print(f"{name:9s} {tag}: E[eta]/T=" + ",".join(f"{m / T:.4g}" for m, T in zip(means, Ts))
              + f" | sd(eta)=" + ",".join(f"{s_:.3g}" for s_ in sds)
              + f" | slope(log mean)={slope_mean:.3f} slope(log sd)={slope_sd:.3f}"
              + f" | mean/sd @T=8192={means[-1] / sds[-1]:.1f}")

print("\n=== A3: GRPO group G=8, reward Bernoulli(0.5), lengths lognormal(mean~3000, cv~0.5) clipped [200,8192] ===")
print("ratio = sd(centered -beta*eta)/sd(centered reward); len_part = share of shaping variance explained by length")
G, NG = 8, 300
for name in formats:
    for beta in (1e-3, 1e-2, 4e-2):
        sh_all, r_all, lin_all = [], [], []
        for _ in range(NG):
            Tg = np.clip(rng.lognormal(np.log(3000) - 0.11, 0.47, size=G), 200, 8192).astype(int)
            eta = np.array([sample_eta(name, int(T), 1, True)[0] for T in Tg])
            r = (rng.random(G) < 0.5).astype(float)
            sh = -beta * eta
            sh_all.append(sh - sh.mean())
            r_all.append(r - r.mean())
            lin = -beta * ktoks[name] * Tg  # length-linear predictor; its scale is irrelevant (len_part is an R^2)
            lin_all.append(lin - lin.mean())
        sh_all, r_all, lin_all = map(np.concatenate, (sh_all, r_all, lin_all))
        # regress centered shaping on centered length-linear predictor
        b = (sh_all @ lin_all) / (lin_all @ lin_all)
        resid = sh_all - b * lin_all
        len_share = 1 - resid.var() / sh_all.var()
        print(f"{name:9s} beta={beta:<6g} ratio={sh_all.std() / r_all.std():.3f}  "
              f"len_part={len_share:.2f}  resid_ratio={resid.std() / r_all.std():.3f}")

print("\n=== B: exact optimum of stationary toy MDP (per-step reward g, anchor a) ===")
# Policy emits content tokens until EOS. g(top)=0, g(other content)=-c, g(EOS)=0.
# Linearly solvable: pi*(y) = a(y) exp(g(y)/beta) for content, pi*(EOS) = 1 - S,
# S = sum_content a(y) exp(g(y)/beta);  mean length = 1/(1-S).
Vc = 30
lc = 3.0 * rng.standard_normal(Vc)
lc[0] = lc.max() + 4.0  # confident top token
eos_logit = np.log(1 / 400.0)  # EOS prob ~ 1/400 under the reference
full = np.concatenate([lc, [0.0]])


def anchor(s):
    lg = full.copy()
    lg[:Vc] = np.log((1 - 1 / 400.0) * softmax(lc / (1 + 0.5 * s) + s * np.r_[0.0, rng_b.standard_normal(Vc - 1)]))
    lg[Vc] = eos_logit  # EOS probability kept identical to the reference
    return np.exp(lg)


for c_pen in (2.0,):
    for s in (0.0, 0.02, 0.1, 0.3):
        rng_b = np.random.default_rng(1)
        a = anchor(s)
        a_ref = anchor(0.0) if s else a
        kl = (a_ref * np.log(a_ref / a)).sum()
        out = []
        for beta in (1.0, 0.3, 0.1, 0.03):
            g = np.full(Vc, -c_pen)
            g[0] = 0.0
            pc = a[:Vc] * np.exp(g / beta)
            S = pc.sum()
            pstar = np.r_[pc, 1 - S]
            H = -(pstar * np.log(pstar)).sum()
            out.append(f"b={beta}: len={1 / (1 - S):7.1f} H={H:.3f}")
        print(f"anchor s={s:<4} KL(ref||q)/tok={kl:.4f} a_top={a[0]:.4f} anchorLen={1 / a[Vc]:.0f} | " + " | ".join(out))

print("\n=== C: k3-as-loss gradient under a quantized anchor (Monte Carlo, one context) ===")
lt = 2.0 * rng.standard_normal(50)
pt = softmax(lt)
pq = softmax(lt / 1.1 + 0.2 * rng.standard_normal(50))
ys = rng.choice(50, size=400000, p=pt)
rho = pq[ys] / pt[ys]
# d/dlogits of k3 = rho - 1 - log rho, with rho = pq/pt:  (1 - rho) * (onehot(y) - pt)
grad_mc = np.zeros(50)
np.add.at(grad_mc, ys, (1 - rho))
grad_mc = grad_mc / len(ys) - (1 - rho).mean() * pt
grad_fwd = pt - pq  # gradient of KL(pq || pt) wrt logits of pt
print(f"max|MC - (pi_theta - pi_q)| = {np.abs(grad_mc - grad_fwd).max():.2e}  "
      f"(scale max|pi_theta-pi_q| = {np.abs(grad_fwd).max():.3e}); "
      f"cosine = {grad_mc @ grad_fwd / np.linalg.norm(grad_mc) / np.linalg.norm(grad_fwd):.4f}")
