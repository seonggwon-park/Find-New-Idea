"""Idea 1 numeric check: token-axis first-order error feedback (noise shaping) of the V cache.

Claim tested (sharpened hypothesis):
  attention-output error of shaped V  ~= h * sigma^2 * sum_t (p_t - p_{t+1})^2
  attention-output error of RTN V     ~= sigma^2 * sum_t p_t^2
  => measured ratio MSE_SD / MSE_RTN ~= h * R,  R = sum(dp)^2 / sum p^2,
  h = range-headroom inflation of the per-element error (~16/9 at 2-bit per-token min-max).
Also: (i) closed-form crossover for i.i.d. logit jitter  p ~ exp(beta z): R = 2(1-exp(-beta^2)),
      so shaping wins iff beta < beta* = sqrt(-ln(1 - 1/(2h)));
      (ii) shaped 2-bit vs unshaped 3-bit; (iii) K-side shaping gives no gain.
Synthetic data only (Gaussian V/K with heterogeneous channels); nothing here is an LLM measurement.
"""
import numpy as np

N, D = 4096, 128
SEEDS = range(6)


def make_V(rng):
    mu = rng.normal(0, 0.5, D)
    sc = np.exp(rng.normal(0, 0.5, D))
    return mu + sc * rng.normal(size=(N, D))


def qtok(w, bits, xi=None):
    """per-token asymmetric min-max (KIVI-V style); optional subtractive dither xi in [-1/2,1/2)."""
    L = 2 ** bits - 1
    lo, hi = w.min(-1, keepdims=True), w.max(-1, keepdims=True)
    s = (hi - lo) / L
    s[s == 0] = 1.0
    x = (w - lo) / s
    if xi is None:
        return np.clip(np.round(x), 0, L) * s + lo
    return (np.clip(np.round(x + xi), 0, L) - xi) * s + lo


def errors(V, bits, rng, dither):
    """returns RTN error matrix E_r and shaped error matrix E_s (token 0 = sink kept FP in both)."""
    xi = rng.uniform(-0.5, 0.5, size=V.shape) if dither else None
    Vr = qtok(V, bits, xi)
    Er = Vr - V
    Er[0] = 0.0
    Es = np.zeros_like(V)
    u = np.zeros(D)                       # write-time carry only, never stored
    car = np.zeros_like(V)
    for t in range(1, N):
        w = V[t] - u
        q = qtok(w[None], bits, None if xi is None else xi[t][None])[0]
        u = q - w                          # error of this write
        car[t] = u
        Es[t] = q - V[t]                   # = u_t - u_{t-1}
    return Er, Es, car


def out_mse(E, p):
    return float(np.sum((p @ E) ** 2))


def Rstat(p):
    p = p.copy(); p[0] = 0.0               # sink excluded (FP)
    dp = p - np.r_[p[1:], 0.0]
    return float(np.sum(dp[1:] ** 2) / np.sum(p[1:] ** 2))


def families(rng):
    t = np.arange(N)
    F = {}
    def norm(x):
        x = x.copy(); return x / x.sum()
    F['uniform'] = norm(np.ones(N))
    F['local256'] = norm((t >= N - 256).astype(float))
    F['recency_tau512'] = norm(np.exp(-(N - 1 - t) / 512.0))
    sm = np.zeros(N); a = 0.0
    z = rng.normal(size=N)
    for i in range(N):
        a = 0.995 * a + np.sqrt(1 - 0.995 ** 2) * z[i]; sm[i] = a
    F['smooth_logits_std3'] = norm(np.exp(3.0 * sm))
    for beta in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        F[f'iid_jitter_beta{beta}'] = norm(np.exp(beta * rng.normal(size=N)))
    F['sink0.3+jitter0.3'] = None  # built below
    p = norm(np.exp(0.3 * rng.normal(size=N))) * 0.7; p[0] += 0.3
    F['sink0.3+jitter0.3'] = p
    sp = np.full(N, 0.2 / N); sp[rng.choice(np.arange(1, N), 4, replace=False)] += 0.2
    F['retrieval_4spikes'] = sp
    F['last_token_0.5+local'] = norm((t >= N - 256).astype(float)) * 0.5; F['last_token_0.5+local'][-1] += 0.5
    return F


def main():
    rng0 = np.random.default_rng(1234)
    F = families(rng0)
    names = list(F)
    print('== Part A: V shaping, per-token min-max, sink token FP; averages over %d seeds, N=%d, d=%d' % (len(SEEDS), N, D))
    res = {}
    for bits in (2, 3, 4):
        for dither in (False, True):
            acc_r = {k: 0.0 for k in names}; acc_s = {k: 0.0 for k in names}; hs = []
            for sd in SEEDS:
                rng = np.random.default_rng(100 + sd)
                V = make_V(rng)
                Er, Es, car = errors(V, bits, rng, dither)
                hs.append(np.mean(car[1:] ** 2) / np.mean(Er[1:] ** 2))
                for k in names:
                    acc_r[k] += out_mse(Er, F[k]); acc_s[k] += out_mse(Es, F[k])
            h = float(np.mean(hs))
            res[(bits, dither)] = (h, acc_r, acc_s)
            print(f'\n-- bits={bits} dither={dither}: measured headroom h = carry var / RTN err var = {h:.3f}'
                  f'  (16/9={16/9:.3f}, (8/7)^2={(8/7)**2:.3f}, (16/15)^2={(16/15)**2:.3f})')
            print('%-24s %8s %9s %9s %10s %10s' % ('family', 'R', 'pred h*R', 'meas', 'MSE_RTN', 'MSE_SD'))
            lp, lm = [], []
            for k in names:
                R = Rstat(F[k]); m = acc_s[k] / acc_r[k]
                lp.append(np.log(h * R)); lm.append(np.log(m))
                print('%-24s %8.2e %9.2e %9.2e %10.3e %10.3e' % (k, R, h * R, m, acc_r[k] / len(SEEDS), acc_s[k] / len(SEEDS)))
            rho = np.corrcoef(lp, lm)[0, 1]
            agree = sum((a < 0) == (b < 0) for a, b in zip(lp, lm))
            print(f'   Pearson corr(log pred, log meas) = {rho:.4f}; sign agreement (win/lose) = {agree}/{len(names)}')
    # crossover
    print('\n== Part B: i.i.d. logit jitter crossover, closed form R = 2(1-exp(-beta^2))')
    for bits in (2, 4):
        h = res[(bits, False)][0]
        bstar = np.sqrt(-np.log(1 - 1 / (2 * h)))
        print(f'bits={bits}: h={h:.3f} -> predicted beta* = {bstar:.3f}')
        for beta in (0.25, 0.5, 0.75, 1.0):
            k = f'iid_jitter_beta{beta}'
            print(f'   beta={beta}: R_closed={2*(1-np.exp(-beta**2)):.3f}  R_emp={Rstat(F[k]):.3f}  '
                  f'meas ratio={res[(bits, False)][2][k]/res[(bits, False)][1][k]:.3f}')
    print('\n== Part C: shaped 2-bit vs unshaped 3-bit (no dither): MSE_SD2 / MSE_RTN3 (<1 means 2-bit shaped beats 3-bit RTN)')
    for k in names:
        print('%-24s %9.3e' % (k, res[(2, False)][2][k] / res[(3, False)][1][k]))
    # K side
    print('\n== Part D: K-side shaping (4-bit and 2-bit), V in FP; smooth K so that p is smooth')
    for bits in (4, 2):
        rr = []
        for sd in range(3):
            rng = np.random.default_rng(500 + sd)
            Ks = np.zeros((N, D)); a = np.zeros(D)
            z = rng.normal(size=(N, D))
            for i in range(N):
                a = 0.99 * a + np.sqrt(1 - 0.99 ** 2) * z[i]; Ks[i] = a
            K = Ks + 0.3 * rng.normal(size=(N, D))
            V = make_V(rng)
            q = rng.normal(size=(8, D)) * 1.5          # 8 queries
            Er, Es, _ = errors(K, bits, rng, False)
            def attn(Kx):
                s = q @ Kx.T / np.sqrt(D); s -= s.max(1, keepdims=True); P = np.exp(s); P /= P.sum(1, keepdims=True)
                return P @ V, P
            o, P = attn(K)
            mr = np.sum((attn(K + Er)[0] - o) ** 2); ms = np.sum((attn(K + Es)[0] - o) ** 2)
            Rk = np.mean([Rstat(P[j]) for j in range(8)])
            rr.append((ms / mr, Rk))
        print(f'bits={bits}: mean R(p) of these heads = {np.mean([x[1] for x in rr]):.3f}; '
              f'output MSE ratio SD/RTN (K shaped) = {np.mean([x[0] for x in rr]):.3f}  per-seed {[round(x[0],3) for x in rr]}')


if __name__ == '__main__':
    main()
