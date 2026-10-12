"""Idea 10 (XS-EVT) synthetic check.

Tests the sharpened hypothesis of idea 10 under fully controlled synthetic
conditions (numpy only, fixed seeds, < 1 min CPU):

  A. Is the leading-order Gaussian-EVT law  ln N* = m^2 / (2 (sd^2 + sq^2))
     a usable *predictor* of the 50%-accuracy onset N50, or only the finite-N
     exact Gaussian formula?  Does the ratio law ln N*_q / ln N*_FP = 1/(1+r)
     hold for the exact N50?
  B. Fit-on-short / predict-long: estimate (m, sd, sq) from 4K-token contexts
     and predict N50 when distractors are heavy-tailed (Student-t, df=4).
     Compare a Gaussian-moment predictor with a pooled-empirical-tail predictor.
  C. What r = sq^2/sd^2 does well-scaled KV quantization actually produce
     (per-token INT2/3/4/8 and FP8-e4m3 on Gaussian keys, plus an outlier-channel
     variant)?  What N50 does the law then predict for a head whose FP N50 = 1M?
  D. Per-head allocator: at a fixed mean bit budget, does a length-targeted
     allocator beat a length-agnostic logit-MSE allocator, and does the gain grow
     with target length?

Model of one retrieval head: needle logit m + sq*z0, N distractor logits
sd*X_j + sq*e_j (X_j iid standardized, e_j, z0 iid N(0,1)); success iff the
needle logit beats every distractor (argmax readout).
"""
import math
import numpy as np

LN2 = math.log(2.0)
_erfc = np.frompyfunc(math.erfc, 1, 1)


def gauss_Q(x):
    return 0.5 * _erfc(np.asarray(x, dtype=float) / math.sqrt(2.0)).astype(float)


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)
GH_W = GH_W / GH_W.sum()


def p_success(N, m, sq, tail_Q):
    """E_z0[(1 - Q_D(m + sq z0))^N], Q_D = upper tail of one noisy distractor."""
    q = np.clip(tail_Q(m + sq * GH_X), 0.0, 1.0 - 1e-16)
    return float(np.sum(GH_W * np.exp(N * np.log1p(-q))))


def n50(m, sq, tail_Q, lo=0.0, hi=45.0):
    """Context length at which retrieval accuracy = 0.5 (bisection in ln N)."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if p_success(math.exp(mid), m, sq, tail_Q) > 0.5:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def gauss_tail(sd, sq):
    s = math.sqrt(sd * sd + sq * sq)
    return lambda x: gauss_Q(np.asarray(x) / s)


def law_lnN(m, sd, sq):
    return m * m / (2.0 * (sd * sd + sq * sq))


print("=" * 78)
print("A. Leading-order law vs exact finite-N Gaussian onset (sd = 1)")
print("=" * 78)
print(f"{'m':>4} {'r':>6} {'N50 exact':>12} {'N* law':>10} {'exact/law':>10}"
      f" {'lnN50q/lnN50FP':>15} {'1/(1+r)':>8}")
for m in (4.0, 5.0, 6.0):
    n_fp = n50(m, 0.0, gauss_tail(1.0, 0.0))
    for r in (0.0, 0.06, 0.33, 1.0):
        sq = math.sqrt(r)
        ne = n50(m, sq, gauss_tail(1.0, sq))
        nl = math.exp(law_lnN(m, 1.0, sq))
        print(f"{m:4.1f} {r:6.2f} {ne:12.4g} {nl:10.4g} {ne / nl:10.2f}"
              f" {math.log(ne) / math.log(n_fp):15.3f} {1 / (1 + r):8.3f}")

print()
print("=" * 78)
print("B. Fit on 4K contexts, predict N50 under heavy-tailed (t, df=4) distractors")
print("=" * 78)
rng = np.random.default_rng(0)
DF = 4.0
TSCALE = math.sqrt((DF - 2) / DF)  # unit-variance Student-t


def t_tail_grid(sq):
    """Exact (numerical) tail of TSCALE*T_df + sq*N(0,1) on a fine grid."""
    dx = 0.002
    x = np.arange(-60.0, 60.0 + dx, dx)
    c = math.gamma((DF + 1) / 2) / (math.sqrt(DF * math.pi) * math.gamma(DF / 2))
    u = x / TSCALE
    pdf = c * (1 + u * u / DF) ** (-(DF + 1) / 2) / TSCALE
    if sq > 0:
        kx = np.arange(-8 * sq, 8 * sq + dx, dx)
        ker = np.exp(-0.5 * (kx / sq) ** 2)
        ker /= ker.sum()
        pdf = np.convolve(pdf, ker, mode="same")
    tail = np.cumsum((pdf * dx)[::-1])[::-1]
    return lambda z: np.interp(np.asarray(z, float), x, tail)


N_SHORT, N_PROMPTS = 4096, 1000  # 4.1M pooled distractor logits from 4K contexts
print(f"short-context calibration: {N_PROMPTS} prompts x {N_SHORT} distractors")
print(f"{'m':>4} {'r':>5} {'N50 true(t)':>12} {'Gauss-pred':>11} {'ratio':>6}"
      f" {'empir-pred':>11} {'ratio':>6} {'law-pred':>9} {'ratio':>6}")
for m in (5.0, 6.0):
    for r in (0.0, 0.33):
        sq = math.sqrt(r)
        true_tail = t_tail_grid(sq)
        n_true = n50(m, sq, true_tail)
        # calibration sample at 4K (quantized logits observed directly)
        X = TSCALE * rng.standard_t(DF, size=N_PROMPTS * N_SHORT)
        D = X + sq * rng.standard_normal(X.size)
        s_hat = float(D.std())
        sq_hat = sq  # assume sq measured exactly (noise is directly observable)
        sd_hat = math.sqrt(max(s_hat ** 2 - sq_hat ** 2, 1e-12))
        n_g = n50(m, sq_hat, gauss_tail(sd_hat, sq_hat))
        Ds = np.sort(D)

        def emp_tail(z, Ds=Ds):
            z = np.asarray(z, float)
            k = Ds.size - np.searchsorted(Ds, z, side="right")
            return np.maximum(k, 0.5) / Ds.size  # floor: half a count

        n_e = n50(m, sq_hat, emp_tail)
        n_l = math.exp(law_lnN(m, sd_hat, sq_hat))
        print(f"{m:4.1f} {r:5.2f} {n_true:12.4g} {n_g:11.4g} {n_g / n_true:6.3f}"
              f" {n_e:11.4g} {n_e / n_true:6.3f} {n_l:9.4g} {n_l / n_true:6.3g}")
del X, D, Ds

print()
print("=" * 78)
print("C. r = sq^2/sd^2 produced by well-scaled KV quantization (Gaussian keys, d=128)")
print("=" * 78)
d, T = 128, 20000
K = rng.standard_normal((T, d))
K_out = K.copy()
K_out[:, :4] *= 10.0  # 4 outlier channels, typical of pre-RoPE keys
q = rng.standard_normal(d)


def q_token(Kx, b):  # asymmetric per-token min-max (KIVI-style for V, generic)
    lo, hi = Kx.min(1, keepdims=True), Kx.max(1, keepdims=True)
    step = (hi - lo) / (2 ** b - 1)
    return np.round((Kx - lo) / step) * step + lo


def q_channel(Kx, b):  # asymmetric per-channel min-max (KIVI-style for K)
    lo, hi = Kx.min(0, keepdims=True), Kx.max(0, keepdims=True)
    step = (hi - lo) / (2 ** b - 1)
    return np.round((Kx - lo) / step) * step + lo


def q_fp8(Kx):  # e4m3 mantissa rounding, range assumed well-scaled
    e = np.floor(np.log2(np.abs(Kx) + 1e-30))
    step = 2.0 ** (e - 3)
    return np.round(Kx / step) * step


def r_of(Kx, Kq):
    s = Kx @ q
    return float(((Kq - Kx) @ q).var() / s.var())


rows = []
for name, fn in [("INT2", lambda X: q_token(X, 2)), ("INT3", lambda X: q_token(X, 3)),
                 ("INT4", lambda X: q_token(X, 4)), ("INT8", lambda X: q_token(X, 8)),
                 ("FP8e4m3", q_fp8)]:
    rows.append((name, r_of(K, fn(K)), r_of(K_out, fn(K_out)),
                 r_of(K_out, q_channel(K_out, int(name[3])) if name.startswith("INT") else fn(K_out))))
m_1M = None
lo, hi = 3.0, 8.0  # margin whose FP N50 = 1M under Gaussian distractors
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if n50(mid, 0.0, gauss_tail(1.0, 0.0)) < 1e6:
        lo = mid
    else:
        hi = mid
m_1M = 0.5 * (lo + hi)
print(f"margin m giving FP N50 = 1M (Gaussian, sd=1): m = {m_1M:.3f}")
print(f"{'format':>8} {'r tok(gauss)':>13} {'r tok(outl)':>12} {'r chan(outl)':>13}"
      f" {'N50 tok(g)':>11} {'N50 tok(o)':>11} {'N50 chan(o)':>12}")
for name, rg, ro, rc in rows:
    ns = [n50(m_1M, math.sqrt(x), gauss_tail(1.0, math.sqrt(x))) for x in (rg, ro, rc)]
    print(f"{name:>8} {rg:13.3e} {ro:12.3e} {rc:13.3e} {ns[0]:11.4g} {ns[1]:11.4g} {ns[2]:12.4g}")
R0 = {2: rows[0][1], 3: rows[1][1], 4: rows[2][1], 8: rows[3][1]}

print()
print("=" * 78)
print("D. Per-head allocation at fixed mean bits (16 heads, bits in {2,3,4,8})")
print("=" * 78)
rng = np.random.default_rng(1)
H, BITS, B_MEAN = 16, (2, 3, 4, 8), 3.0
BUDGET = int(round(B_MEAN * H))
lnN_fp = rng.uniform(math.log(2 ** 14), math.log(2 ** 24), H)  # FP N50 16K..16M
k_h = np.exp(rng.uniform(math.log(0.5), math.log(4.0), H))      # noise sensitivity
# margin per head giving its FP N50
m_h = []
for target in lnN_fp:
    lo, hi = 2.0, 9.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if math.log(n50(mid, 0.0, gauss_tail(1.0, 0.0))) < target:
            lo = mid
        else:
            hi = mid
    m_h.append(0.5 * (lo + hi))
m_h = np.array(m_h)


def acc(h, b, N):
    sq = math.sqrt(k_h[h] * R0[b])
    return p_success(N, m_h[h], sq, gauss_tail(1.0, sq))


def dp_alloc(score):  # maximize sum_h score[h][b] s.t. sum bits == BUDGET
    best = {0: (0.0, [])}
    for h in range(H):
        nb = {}
        for used, (val, path) in best.items():
            for b in BITS:
                u = used + b
                if u > BUDGET:
                    continue
                v = val + score[h][b]
                if u not in nb or v > nb[u][0]:
                    nb[u] = (v, path + [b])
        best = nb
    return best[BUDGET][1]


mse_alloc = dp_alloc([{b: -k_h[h] * R0[b] for b in BITS} for h in range(H)])
print("length-agnostic logit-MSE allocation:", mse_alloc)
print(f"{'N target':>9} {'uniform3':>9} {'MSE':>7} {'targeted':>9} {'gain vs MSE':>12}"
      f" {'gain vs unif':>13} {'FP16':>7}")
for N in (2 ** 13, 2 ** 15, 2 ** 17, 2 ** 19, 2 ** 21):
    tbl = [{b: acc(h, b, N) for b in BITS} for h in range(H)]
    tgt = dp_alloc(tbl)
    a_u = np.mean([tbl[h][3] for h in range(H)])
    a_m = np.mean([tbl[h][mse_alloc[h]] for h in range(H)])
    a_t = np.mean([tbl[h][tgt[h]] for h in range(H)])
    a_fp = np.mean([p_success(N, m_h[h], 0.0, gauss_tail(1.0, 0.0)) for h in range(H)])
    print(f"{N:9d} {a_u:9.4f} {a_m:7.4f} {a_t:9.4f} {a_t - a_m:12.4f} {a_t - a_u:13.4f} {a_fp:7.4f}")
