"""Idea 9 numeric check: band-resolved PTQ counteraction kappa_i in a synthetic
pre-norm (RMSNorm) residual block trained as a denoiser.

Claims under test (sharpened hypothesis, synthetic analogue only):
 C1 textbook: least-squares linear residual block on h=x+n is a Wiener filter,
    per-eigendirection kappa_i = tau^2/(lambda_i+tau^2).
 C2 pre-norm nonlinear block (random-feature ReLU MLP on RMSNorm(h), readout fit by
    ridge) trained with input noise tau^2>0: after projecting out the radial
    direction, kappa is large in low-variance bands and ~0 in the top-PC band;
    with tau^2=0 training kappa is ~0 in all bands. Per-band kappa matches the
    symmetric-Jacobian prediction.
 C3 confound: error along the RMSNorm radial direction (massive channel on massive
    tokens) gets kappa~0 for ANY pre-norm block, trained or random, so "no
    cancellation along massive/top direction" is not diagnostic without control.
 C4 depth fragility of the global-Wiener model: composing L Wiener maps destroys
    legitimate low-variance signal.
 C5 error-shaping headroom: same error energy placed in bottom bands vs isotropic.
All numbers are synthetic; nothing here is evidence about real LLMs.
"""
import numpy as np

rng = np.random.default_rng(0)
d, k_tan, M_massive, p_massive = 48, 6, 30.0, 0.05
TAU2 = 0.02

# ---------------- data model ----------------
Q = np.linalg.qr(rng.standard_normal((d - 1, d - 1)))[0]          # basis of coords 1..d-1
lam = 2.0 * np.arange(1, d) ** -1.5                                # power-law spectrum
Ufull = np.zeros((d, d - 1)); Ufull[1:, :] = Q
mu = np.zeros(d); mu[1:] = Q @ (rng.standard_normal(d - 1) * 0.25)
U_tan = Ufull[:, :k_tan]
B_tan = rng.standard_normal((k_tan, k_tan)) * 0.4


def sample_x(n, massive_frac=p_massive, r=rng):
    z = r.standard_normal((n, d - 1)) * np.sqrt(lam)
    x = mu + z @ Ufull.T
    is_m = r.random(n) < massive_frac
    x[:, 0] = np.where(is_m, M_massive, 0.3 * r.standard_normal(n))
    return x, is_m


def target(x):  # clean computation inside the tangent subspace
    c = (x - mu) @ U_tan
    return x + np.tanh(c) @ B_tan.T @ U_tan.T


def rms(h):
    return np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True))


m_feat = 640
W1 = rng.standard_normal((m_feat, d)) / np.sqrt(d) * 2.0
b1 = rng.uniform(-1, 1, m_feat)


def feats(h):
    nh = h / rms(h)
    return np.concatenate([nh, np.maximum(nh @ W1.T + b1, 0.0)], axis=1)


def train_block(tau2, n=24000, ridge=1e-3, seed=1):
    r = np.random.default_rng(seed)
    x, _ = sample_x(n, r=r)
    h = x + np.sqrt(tau2) * r.standard_normal(x.shape)
    F = feats(h)
    T = target(x) - h
    A = F.T @ F + ridge * n * np.eye(F.shape[1]) / F.shape[1]
    return np.linalg.solve(A, F.T @ T)


def block(W2):
    return lambda h: feats(h) @ W2


# eigenbands of the hidden-state covariance (incl. massive channel)
xc, _ = sample_x(60000, r=np.random.default_rng(7))
evals, evecs = np.linalg.eigh(np.cov(xc.T))
order = np.argsort(evals)[::-1]; evals, evecs = evals[order], evecs[:, order]
bands = {"top(1-4)": range(0, 4), "mid(5-16)": range(4, 16), "bot(17-48)": range(16, d)}
P = {b: evecs[:, list(ix)] @ evecs[:, list(ix)].T for b, ix in bands.items()}


def kappa_bands(f, h, eps=0.01, radial_out=True, n_rep=4, r=None):
    r = r or np.random.default_rng(11)
    num = {b: 0.0 for b in bands}; den = {b: 0.0 for b in bands}
    fh = f(h)
    for _ in range(n_rep):
        dh = r.standard_normal(h.shape)
        if radial_out:
            hh = h / np.linalg.norm(h, axis=1, keepdims=True)
            dh -= np.sum(dh * hh, axis=1, keepdims=True) * hh
        dh *= eps * np.linalg.norm(h, axis=1, keepdims=True) / np.linalg.norm(dh, axis=1, keepdims=True)
        du = f(h + dh) - fh
        for b in bands:
            a, c = dh @ P[b], du @ P[b]
            num[b] += -np.sum(a * c); den[b] += np.sum(a * a)
    return {b: num[b] / den[b] for b in bands}


def jac_pred(f, h, n_tok=80, eps=1e-4):
    """kappa predicted by the symmetric Jacobian restricted to each band, radial removed."""
    out = {b: [] for b in bands}
    for t in range(n_tok):
        h0 = h[t]; f0 = f(h0[None])[0]
        J = np.stack([(f((h0 + eps * np.eye(d)[j])[None])[0] - f0) / eps for j in range(d)], 1)
        Js = 0.5 * (J + J.T)
        hh = h0 / np.linalg.norm(h0); R = np.eye(d) - np.outer(hh, hh)
        for b in bands:
            Pb = R @ P[b] @ R
            out[b].append(-np.trace(Pb @ Js @ Pb) / np.trace(Pb @ Pb))
    return {b: float(np.mean(v)) for b, v in out.items()}


print("=== C1 textbook linear-Gaussian Wiener check (no norm, linear LS block) ===")
r = np.random.default_rng(3)
z = r.standard_normal((200000, 8)) * np.sqrt(np.array([2, 1, .5, .2, .1, .05, .02, .005]))
hn = z + np.sqrt(TAU2) * r.standard_normal(z.shape)
Wls = np.linalg.lstsq(hn, z - hn, rcond=None)[0]       # f(h)=h@Wls
kap_lin = -np.diag(Wls)
kap_th = TAU2 / (np.array([2, 1, .5, .2, .1, .05, .02, .005]) + TAU2)
print("kappa fitted :", np.round(kap_lin, 3))
print("kappa theory :", np.round(kap_th, 3))
print("max abs err  : %.4f" % np.max(np.abs(kap_lin - kap_th)))

print("\n=== C2 pre-norm random-feature block, band-resolved kappa (radial removed) ===")
print("band eigenvalues: top %.3g..%.3g | mid %.3g..%.3g | bot %.3g..%.3g" % (
    evals[0], evals[3], evals[4], evals[15], evals[16], evals[-1]))
xt, ism = sample_x(4000, r=np.random.default_rng(21))
xt_n = xt[~ism]
res = {}
W2_store = {}
for tau2 in [0.0, TAU2, 4 * TAU2]:
    for seed in [1, 2]:
        W2_store[(tau2, seed)] = train_block(tau2, seed=seed)
        res[(tau2, seed)] = kappa_bands(block(W2_store[(tau2, seed)]), xt_n)
for tau2 in [0.0, TAU2, 4 * TAU2]:
    k1, k2 = res[(tau2, 1)], res[(tau2, 2)]
    s = "  ".join("%s %.3f/%.3f" % (b, k1[b], k2[b]) for b in bands)
    ratio = k1["bot(17-48)"] / max(abs(k1["top(1-4)"]), 1e-3)
    print("tau2=%.3f  kappa(seed1/seed2): %s  | bot/|top| = %.1f" % (tau2, s, ratio))
wien = {b: float(np.mean(TAU2 / (np.clip(evals[list(ix)], 0, None) + TAU2))) for b, ix in bands.items()}
print("global-Wiener band means at tau2=%.3f: %s" % (TAU2, {b: round(v, 3) for b, v in wien.items()}))
f_noise = block(W2_store[(TAU2, 1)])
jp = jac_pred(f_noise, xt_n)
print("sym-Jacobian predicted kappa (tau2=%.3f): %s" % (TAU2, {b: round(v, 3) for b, v in jp.items()}))
print("measured kappa            (tau2=%.3f): %s" % (TAU2, {b: round(float(v), 3) for b, v in res[(TAU2, 1)].items()}))
print("note: top band is NOT ~0; the tangent computation g contributes its own Jacobian (tau2=0 arm: top %.3f)" % res[(0.0, 1)]["top(1-4)"])

print("\n=== C3 RMSNorm radial confound ===")
W2_clean = W2_store[(0.0, 1)]
f_clean = block(W2_clean)
W2_rand = np.random.default_rng(5).standard_normal(W2_clean.shape) * 0.02
f_rand = block(W2_rand)
xm = xt[ism]
cos_e0 = np.mean(np.abs(xm[:, 0]) / np.linalg.norm(xm, axis=1))
print("massive tokens: n=%d, mean |cos(h, e0)| = %.3f" % (len(xm), cos_e0))


def kappa_dir(f, h, v_fn, eps=0.01):
    v = v_fn(h); v /= np.linalg.norm(v, axis=1, keepdims=True)
    dh = eps * np.linalg.norm(h, axis=1, keepdims=True) * v
    du = f(h + dh) - f(h)
    return -np.sum(dh * du) / np.sum(dh * dh)


e0 = lambda h: np.tile(np.eye(d)[0], (len(h), 1))
radial = lambda h: h.copy()
for name, f in [("noise-trained", f_noise), ("clean-trained", f_clean), ("random W2", f_rand)]:
    print("%-14s kappa along e0 (massive tok) = %+.4f | along radial h (normal tok) = %+.4f | "
          "isotropic-tangential bot band = %+.3f" % (
              name, kappa_dir(f, xm, e0), kappa_dir(f, xt_n, radial),
              kappa_bands(f, xt_n)["bot(17-48)"]))
lin_wien_e0 = TAU2 / (evals[0] + TAU2)
print("no-norm global Wiener would give along top PC (e0, lambda=%.1f): %.4f" % (evals[0], lin_wien_e0))
mu_hat = np.mean(xt_n, 0); mu_hat /= np.linalg.norm(mu_hat)
wb = np.sum((mu_hat @ evecs[:, 16:]) ** 2)
print("normal tokens: fraction of mean-direction energy in bottom band = %.3f "
      "(no-norm Wiener would cancel it; pre-norm block gives ~0 along h)" % wb)

print("\n=== C4 depth fragility of global-Wiener composition ===")
for L in [1, 8, 24]:
    keep = (evals[16:] / (evals[16:] + TAU2)) ** (2 * L)
    print("L=%2d retained signal energy in bottom band (mean over dirs) = %.2e" % (L, np.mean(keep)))

print("\n=== C5 error-shaping headroom (one noise-trained block, equal error energy) ===")
r = np.random.default_rng(31)
h = xt_n
for name, Pm in [("isotropic", np.eye(d)), ("top band only", P["top(1-4)"]),
                 ("bottom band only", P["bot(17-48)"])]:
    dh = r.standard_normal(h.shape) @ Pm
    dh *= 0.01 * np.linalg.norm(h, axis=1, keepdims=True) / np.linalg.norm(dh, axis=1, keepdims=True)
    du = f_noise(h + dh) - f_noise(h)
    print("%-17s ||dh+du||^2/||dh||^2 = %.3f" % (name, np.sum((dh + du) ** 2) / np.sum(dh ** 2)))
