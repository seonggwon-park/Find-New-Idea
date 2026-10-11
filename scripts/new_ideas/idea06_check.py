"""Idea 6 (P04) numeric check: pivot-WUSH vs marginal-WUSH under block error feedback.

Synthetic, CPU-only, fixed seeds. Nothing here is an LLM measurement.

Setting (weight-only unless stated):
  W in R^{din x dout}, loss tr(dW^T Sigma dW), din = G*d, group = block = d.
  Block SIC (GPTQ inter-block feedback): for g = 1..G, quantize group g with a
  per-group transform, error delta_g (original coords), then
  W_{>g} <- W_{>g} - Sigma_{>g,>g}^{-1} Sigma_{>g,g} delta_g.
  Lemma (a): loss == sum_g tr(delta_g^T D_g delta_g), D_g = Schur pivot.
Arms: I (identity), Had (Hadamard), WUSH-marg (M_X = Sigma_gg, as WUSH Alg. 2),
      WUSH-piv (M_X = D_g, the proposal).
Variants: 'blk' = no intra-block feedback (the setting of the lemma),
          'gptq' = WUSH Alg. 2 style intra-block GPTQ with H(i) = T D_g T^T.
Quantizers: INT4 AbsMax (levels +-7) and FP4 E2M1 AbsMax (max -> 6), exact
            (non-power-of-two) scales, group size d = 32 per output row.
Sweep rho = share of a cross-group low-rank factor in Sigma.
Secondary: W4A4 deployed output error with activations RTN-quantized (no feedback).
"""
import numpy as np

d, G, dout = 32, 16, 256
din = d * G
SEEDS = [0, 1, 2]
RHOS = [0.0, 0.3, 0.6, 0.9]
LAM = 0.01

def hadamard(n):
    Hm = np.array([[1.0]])
    while Hm.shape[0] < n:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]])
    return Hm / np.sqrt(n)

HAD = hadamard(d)
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6.0])
FP4G = np.concatenate([-FP4[::-1], FP4[1:]])

def make_sigma(rng, rho, r=8):
    # within-group anisotropic blocks + shared cross-group low-rank factor
    B = np.zeros((din, din))
    for g in range(G):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        ev = np.exp(rng.normal(0, 1.0, d))
        Bg = (Q * ev) @ Q.T
        B[g*d:(g+1)*d, g*d:(g+1)*d] = Bg / np.trace(Bg) * d
    F = rng.standard_normal((din, r)) * np.exp(rng.normal(0, 0.5, (din, 1)))
    Fs = F @ F.T
    Fs = Fs / np.trace(Fs) * din
    S = (1 - rho) * B + rho * Fs + 1e-3 * np.eye(din)
    return S

def make_w(rng):
    W = rng.standard_t(4, (din, dout)) * np.exp(rng.normal(0, 0.5, (din, 1)))
    return W

def damp(M):
    return M + LAM * np.trace(M) / M.shape[0] * np.eye(M.shape[0])

def wush(MW, MX):
    Wp = np.linalg.cholesky(damp(MW)); Xp = np.linalg.cholesky(damp(MX))
    U, s, Vt = np.linalg.svd(Wp.T @ Xp)
    return HAD @ np.diag(s ** -0.5) @ U.T @ Wp.T  # T_wush (activation side)

def q_absmax(V, fmt, scale=None):
    # V: d x n, each column is one group (one output row / one token)
    if scale is None:
        scale = np.abs(V).max(0, keepdims=True) + 1e-12
    if fmt == 'int4':
        s = scale / 7.0
        return np.clip(np.round(V / s), -7, 7) * s, scale
    s = scale / 6.0
    Z = np.clip(V / s, -6, 6)
    idx = np.abs(Z[..., None] - FP4G).argmin(-1)
    return FP4G[idx] * s, scale

def intra_gptq(V, Hb, fmt):
    # standard GPTQ over the d coordinates of transformed block, group scale fixed at start
    _, scale = q_absmax(V, fmt)
    Hinv = np.linalg.inv(damp(Hb))
    Lc = np.linalg.cholesky(Hinv).T  # upper factor as in GPTQ
    V = V.copy(); Q = np.zeros_like(V)
    for j in range(d):
        qj, _ = q_absmax(V[j:j+1], fmt, scale)
        Q[j] = qj[0]
        e = (V[j] - Q[j]) / Lc[j, j]
        V[j+1:] -= np.outer(Lc[j, j+1:], e)
    return Q

def run_layer(S, W0, arm, fmt, variant):
    W = W0.copy(); dW = np.zeros_like(W); piv_loss = 0.0
    for g in range(G):
        a, b = g*d, (g+1)*d
        Sgg = S[a:b, a:b]
        if b < din:
            Sgr = S[a:b, b:]; Srr = S[b:, b:]
            K = np.linalg.solve(Srr, Sgr.T)          # Srr^{-1} Srg
            Dg = Sgg - Sgr @ K
        else:
            K = None; Dg = Sgg
        Wg = W[a:b]
        if arm == 'I':
            T = np.eye(d)
        elif arm == 'Had':
            T = HAD
        else:
            MW = Wg @ Wg.T / dout
            T = wush(MW, Sgg if arm == 'WUSH-marg' else Dg)
        Tx = np.linalg.inv(T).T                   # weight-side transform
        V = Tx @ Wg
        if variant == 'blk':
            Q, _ = q_absmax(V, fmt)
        else:
            Q = intra_gptq(V, T @ Dg @ T.T, fmt)
        delta = T.T @ Q - Wg                       # error in original coords
        piv_loss += np.sum(delta * (Dg @ delta))
        W[a:b] = Wg + delta
        if K is not None:
            W[b:] -= K @ delta
    dW = W - W0
    return np.sum(dW * (S @ dW)), piv_loss

def w4a4_err(S, W0, arm, fmt, rng, n=1024):
    # weights: block-SIC 'blk'; activations: per-token per-group RTN, no feedback
    W = W0.copy(); C = np.linalg.cholesky(S)
    X = C @ rng.standard_normal((din, n))
    Y = W0.T @ X; Yq = np.zeros_like(Y)
    for g in range(G):
        a, b = g*d, (g+1)*d
        Sgg = S[a:b, a:b]
        if b < din:
            K = np.linalg.solve(S[b:, b:], S[a:b, b:].T); Dg = Sgg - S[a:b, b:] @ K
        else:
            K = None; Dg = Sgg
        Wg = W[a:b]
        MW = Wg @ Wg.T / dout
        T = wush(MW, Sgg if arm == 'WUSH-marg' else Dg) if arm != 'Had' else HAD
        Tx = np.linalg.inv(T).T
        Qw, _ = q_absmax(Tx @ Wg, fmt)
        Qx, _ = q_absmax(T @ X[a:b], fmt)
        Yq += Qw.T @ Qx
        delta = T.T @ Qw - Wg
        W[a:b] = Wg + delta
        if K is not None:
            W[b:] -= K @ delta
    return np.mean((Yq - Y) ** 2) / np.mean(Y ** 2)

def main():
    print(f"setup: din={din} (G={G} groups of d={d}), dout={dout}, seeds={SEEDS}, damping={LAM}")
    # Lemma (a) identity check
    rng = np.random.default_rng(123)
    S = make_sigma(rng, 0.6); W0 = make_w(rng)
    for variant in ['blk', 'gptq']:
        L, P = run_layer(S, W0, 'WUSH-piv', 'int4', variant)
        print(f"[lemma a] variant={variant}: full loss={L:.6e}  sum_g pivot loss={P:.6e}  rel.diff={abs(L-P)/L:.2e}")
    arms = ['I', 'Had', 'WUSH-marg', 'WUSH-piv']
    for fmt in ['int4', 'fp4']:
        for variant in ['blk', 'gptq']:
            print(f"\n== weight-only, fmt={fmt}, variant={variant} (loss normalized by tr(W^T S W)) ==")
            print("rho   gap(D vs Sigma_gg)  " + "  ".join(f"{a:>10s}" for a in arms) + "   piv-vs-marg gain%")
            for rho in RHOS:
                res = {a: [] for a in arms}; gaps = []
                for sd in SEEDS:
                    r = np.random.default_rng(1000 * sd + int(rho * 10))
                    S = make_sigma(r, rho); W0 = make_w(r)
                    ref = np.sum(W0 * (S @ W0))
                    if variant == 'blk' and fmt == 'int4':
                        gp = []
                        for g in range(G - 1):
                            a, b = g*d, (g+1)*d
                            Dg = S[a:b, a:b] - S[a:b, b:] @ np.linalg.solve(S[b:, b:], S[b:, a:b])
                            gp.append(np.linalg.norm(S[a:b, a:b] - Dg) / np.linalg.norm(S[a:b, a:b]))
                        gaps.append(np.mean(gp))
                    for a_ in arms:
                        res[a_].append(run_layer(S, W0, a_, fmt, variant)[0] / ref)
                m = {a_: np.mean(v) for a_, v in res.items()}
                gain = 100 * np.mean([(mm - pp) / mm for mm, pp in zip(res['WUSH-marg'], res['WUSH-piv'])])
                gs = f"{np.mean(gaps):.3f}" if gaps else "  -  "
                print(f"{rho:.1f}   {gs:>18s}  " + "  ".join(f"{m[a_]:10.3e}" for a_ in arms) + f"   {gain:+.2f}")
    print("\n== W4A4 (weights blk-SIC, activations RTN, same transform pair), rel. output MSE ==")
    print("fmt   rho        Had   WUSH-marg    WUSH-piv   piv-vs-marg gain%")
    for fmt in ['int4', 'fp4']:
        for rho in [0.0, 0.6, 0.9]:
            out = {a: [] for a in ['Had', 'WUSH-marg', 'WUSH-piv']}
            for sd in SEEDS:
                r = np.random.default_rng(5000 + 1000 * sd + int(rho * 10))
                S = make_sigma(r, rho); W0 = make_w(r)
                for a_ in out:
                    out[a_].append(w4a4_err(S, W0, a_, fmt, np.random.default_rng(sd)))
            gain = 100 * np.mean([(mm - pp) / mm for mm, pp in zip(out['WUSH-marg'], out['WUSH-piv'])])
            print(f"{fmt:5s} {rho:.1f}  " + "  ".join(f"{np.mean(v):10.3e}" for v in out.values()) + f"   {gain:+.2f}")

if __name__ == '__main__':
    main()
