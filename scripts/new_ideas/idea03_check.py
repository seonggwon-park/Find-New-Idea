#!/usr/bin/env python3
"""Idea 3 numeric check: worst-case gap between Haar-rotated GPTQ and WaterSIC.

High-rate, entropy-coded model (Ordentlich & Polyanskiy 2605.13768):
  gap(lambda) = 1/2 log2( AM(D) / GM(lambda) ),  D = LDL pivots of Q diag(lambda) Q^T
Eq. 33 approximation: D_k ~= E_k/E_{k-1},  E_k = e_k(lambda)/C(n,k).

Checks
 A. Eq. 33 vs exact Haar Monte Carlo pivots (two-point spectra).
 B. sup over two-point spectra (Eq. 33, n=2000) vs the sharpened asymptotic
    1/2 log2(1/eps) - log2 ln(1/eps) + 1/2 log2(2/e^2).
 C. Rotation dividend: Specht (PCA-basis) worst gap minus rotated worst gap,
    vs 1/2 log2 ln(1/eps).
 D. mu-free QuIP Lemma-2 functional 1/2 log2(AM(sqrt l)^2/GM(l)):
    (i) sup over [eps,1], (ii) is AM(D_Eq33) <= AM(sqrt l)^2 on random spectra?
 E. Two-point extremality: local search over general spectra (n=40).
 F. Damping: near-singular synthetic covariances (m tokens < n), GPTQ 1% damping.
All numbers are synthetic. Seeds fixed.
"""
import time
import numpy as np
from math import lgamma

_lg = np.vectorize(lgamma)


def gammaln(x):
    return _lg(np.asarray(x, dtype=float))

LOG2E = np.log2(np.e)


def log_esm(lam):
    """log e_k(lam), k=0..n, via log-domain DP."""
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
    logC = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    logE = log_esm(lam) - logC
    return np.exp(np.diff(logE))


def gap_from_pivots(D, lam):
    return 0.5 * np.log2(np.mean(D) / np.exp(np.mean(np.log(lam))))


def gap_eq33(lam):
    return gap_from_pivots(eq33_pivots(lam), lam)


def haar(n, rng):
    Z = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    return Q * np.sign(np.diag(R))


def gap_haar(lam, rng, reps):
    n = len(lam)
    out = []
    for _ in range(reps):
        Q = haar(n, rng)
        H = (Q * lam) @ Q.T
        H = 0.5 * (H + H.T)
        Lc = np.linalg.cholesky(H)
        out.append(gap_from_pivots(np.diag(Lc) ** 2, lam))
    return np.mean(out), np.std(out) / np.sqrt(reps)


def two_point(n, a, eps):
    return np.concatenate([np.full(a, eps), np.ones(n - a)])


def sup_two_point(n, eps, fn=gap_eq33):
    # coarse then fine search over a = number of eps-entries
    grid = np.unique(np.linspace(1, n - 1, 41).astype(int))
    vals = [fn(two_point(n, a, eps)) for a in grid]
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.unique(np.linspace(lo, hi, 25).astype(int))
    vals2 = [fn(two_point(n, a, eps)) for a in fine]
    j = int(np.argmax(vals2))
    return vals2[j], fine[j] / n


def asym_rot(eps):
    L = np.log(1 / eps)
    return 0.5 * np.log2(1 / eps) - np.log2(L) + 0.5 * np.log2(2 / np.e ** 2)


def limit_rot(eps):
    """n->inf two-point functional, max over f: 1/2 log2(((1-f)+f ln f)/eps^f)."""
    f = np.linspace(1e-4, 1 - 1e-6, 400001)
    v = 0.5 * np.log2(((1 - f) + f * np.log(f) + eps * 1e-300) / eps ** f)
    return v.max(), f[np.argmax(v)]


def specht_gap(eps):
    h = 1 / eps
    S = (h - 1) * h ** (1 / (h - 1)) / (np.e * np.log(h))
    return 0.5 * np.log2(S)


def quip_functional(lam):
    return 0.5 * np.log2(np.mean(np.sqrt(lam)) ** 2 / np.exp(np.mean(np.log(lam))))


def quip_sup(eps):
    f = np.linspace(1e-6, 1 - 1e-6, 400001)  # fraction at eps
    v = 0.5 * np.log2((f * np.sqrt(eps) + 1 - f) ** 2 / eps ** f)
    return v.max()


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    print("=== A. Eq.33 vs exact Haar MC (two-point, n=200, 30 rotations) ===")
    n = 200
    for eps, f in [(1e-1, 0.52), (1e-2, 0.60), (1e-3, 0.70), (1e-4, 0.75), (1e-6, 0.84)]:
        lam = two_point(n, int(f * n), eps)
        g33 = gap_eq33(lam)
        m, s = gap_haar(lam, rng, 30)
        print(f"eps={eps:.0e} f={f:.2f}: Eq33={g33:.3f}  HaarMC={m:.3f} +- {s:.3f}  diff={m - g33:+.3f}")

    print(f"[t={time.time()-t0:.0f}s]")
    print("\n=== B. sup over two-point spectra vs sharpened asymptotic ===")
    print("eps    | Eq33 n=1000 sup (f*) | leading-order eps->0 limit sup (f*) | asym formula | limit - (1/2log2(1/eps) - log2 ln(1/eps)) | 1-f* vs 2/ln(1/eps)")
    for eps in [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        s2000, f2000 = sup_two_point(1000, eps)
        sl, fl = limit_rot(eps)
        resid = sl - (0.5 * np.log2(1 / eps) - np.log2(np.log(1 / eps)))
        print(f"{eps:.0e} | {s2000:.3f} ({f2000:.3f}) | {sl:.3f} ({fl:.3f}) | {asym_rot(eps):.3f} | {resid:+.3f} | {1 - fl:.4f} vs {2 / np.log(1 / eps):.4f}")
    print(f"limiting residual constant 1/2 log2(2/e^2) = {0.5 * np.log2(2 / np.e ** 2):.4f}")

    print(f"[t={time.time()-t0:.0f}s]")
    print("\n=== C. rotation dividend: Specht (PCA basis) worst gap - rotated worst gap ===")
    for eps in [1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20]:
        sp = specht_gap(eps)
        sl, _ = limit_rot(eps)
        div = sp - sl
        print(f"eps={eps:.0e}: Specht={sp:.3f}  rot={sl:.3f}  dividend={div:.3f}  1/2log2 ln(1/eps)={0.5 * np.log2(np.log(1 / eps)):.3f}  dividend-that={div - 0.5 * np.log2(np.log(1 / eps)):+.3f}")
    print(f"predicted constant -1/2 log2 e - 1/2 log2(2/e^2) = {-0.5 * LOG2E - 0.5 * np.log2(2 / np.e ** 2):+.4f}")

    print("\n=== D. mu-free QuIP Lemma-2 functional ===")
    for eps in [1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20]:
        qs = quip_sup(eps)
        sl, _ = limit_rot(eps)
        print(f"eps={eps:.0e}: sup QuIP(mu=1)={qs:.3f}  sup rot={sl:.3f}  sup-difference={qs - sl:.3f}")
    print(f"asymptotic sup-difference (1-log2 e) - (1/2-log2 e) = {(1 - LOG2E) - (0.5 - LOG2E):.4f} (both maximized at 1-f = 2/ln(1/eps))")
    # pointwise at the rotated worst spectrum
    for eps in [1e-4, 1e-8]:
        _, fl = limit_rot(eps)
        lam = two_point(2000, int(round(fl * 2000)), eps)
        print(f"eps={eps:.0e} pointwise at rotated-worst two-point (n=2000): QuIP={quip_functional(lam):.3f} Eq33={gap_eq33(lam):.3f} diff={quip_functional(lam) - gap_eq33(lam):.3f}")
    # does AM(D_Eq33) <= AM(sqrt lam)^2 hold?  (mu-free expectation-level Lemma 2)
    rng2 = np.random.default_rng(1)
    worst = -np.inf
    worst_h = -np.inf
    cnt = 0
    for t in range(1500):
        nn = int(rng2.integers(2, 120))
        kind = t % 4
        if kind == 0:
            lam = 10 ** rng2.uniform(-8, 0, nn)
        elif kind == 1:
            lam = rng2.choice([1e-6, 1e-3, 1.0], nn)
        elif kind == 2:
            lam = rng2.lognormal(0, 3, nn)
        else:
            lam = rng2.dirichlet(np.full(nn, 0.1)) + 1e-9
        D = eq33_pivots(lam)
        r = np.log2(np.mean(D) / np.mean(np.sqrt(lam)) ** 2)
        worst = max(worst, r)
        cnt += 1
        if t < 150 and nn <= 60:
            m, _ = gap_haar(lam, rng2, 8)
            worst_h = max(worst_h, (m - quip_functional(lam)))
    print(f"max over {cnt} random spectra of log2(AM(D_Eq33)/AM(sqrt l)^2) = {worst:+.4f} (<=0 means mu-free Lemma 2 holds for Eq.33)")
    print(f"max over small-n subset of HaarMC gap - QuIP(mu=1) functional  = {worst_h:+.4f} bit")

    print(f"[t={time.time()-t0:.0f}s]")
    print("\n=== E. two-point extremality: local search over general spectra, n=40, eps=1e-4 ===")
    n, eps = 40, 1e-4
    best2, f2 = sup_two_point(n, eps)
    rng3 = np.random.default_rng(2)
    best_ls = -np.inf
    for restart in range(4):
        x = rng3.uniform(np.log(eps), 0, n)
        cur = gap_eq33(np.exp(x))
        for it in range(1500):
            y = x.copy()
            idx = rng3.integers(0, n, size=int(rng3.integers(1, 4)))
            y[idx] = np.clip(y[idx] + rng3.normal(0, 2.0, len(idx)), np.log(eps), 0)
            g = gap_eq33(np.exp(y))
            if g >= cur:
                x, cur = y, g
        best_ls = max(best_ls, cur)
        frac_end = np.mean((x < np.log(eps) + 1e-6) | (x > -1e-6))
    print(f"best two-point Eq33 gap = {best2:.4f} (f={f2:.2f}); best local-search gap = {best_ls:.4f}; last run fraction at endpoints = {frac_end:.2f}")

    print(f"[t={time.time()-t0:.0f}s]")
    print("\n=== F. damping on near-singular synthetic covariances (n=1024) ===")
    n = 1024
    rng4 = np.random.default_rng(3)
    base = rng4.lognormal(0, 1.0, n)
    for massive in [True, False]:
        scales = base.copy()
        if massive:
            scales[:4] *= 60.0  # massive channels
        for m in [128, 512, 16384]:
            X = rng4.standard_normal((m, n)) * scales
            lam = np.clip(np.linalg.eigvalsh(X.T @ X / m), 0, None)
            mean = lam.mean()
            for pd in [1e-4, 1e-2, 1e-1]:
                lam_d = lam + pd * mean
                eps_eff = lam_d.min() / lam_d.max()
                g = gap_eq33(lam_d)
                print(f"massive={int(massive)} m={m:5d} damp={pd:.0e}*mean: lmax/mean={lam.max() / mean:6.1f} eps_eff={eps_eff:.1e} rot gap(Eq33)={g:.3f} cap={limit_rot(eps_eff)[0]:.2f} QuIP(mu=1)={quip_functional(lam_d):.3f}")
    scales = base.copy()
    X = rng4.standard_normal((128, n)) * scales
    lam = np.clip(np.linalg.eigvalsh(X.T @ X / 128), 0, None)
    lam = lam + 0.01 * lam.mean()
    m_, s_ = gap_haar(lam, rng4, 3)
    print(f"exact Haar MC, no massive, m=128, damp=1e-2*mean: {m_:.3f} +- {s_:.3f}  vs Eq33 {gap_eq33(lam):.3f}")
    print(f"[t={time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
