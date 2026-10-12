#!/usr/bin/env python3
"""Idea 02 numeric check: shared-seed vs independent vs rank-stratified stochastic
rounding (SR) across P data-parallel replicas.

Part A: scalar SR with iid U(0,1) residues. Exact laws to test:
    independent : Var(mean e) = 1/(6P)
    shared u    : Var(mean e) = 1/12 + 1/(12P)        (plateau, <= 2x reduction)
    stratified  : Var(mean e) = 1/(12P) + 1/(12P^2)   (NOT O(1/P^2))
Part B: per-element cross-rank error correlation rho under a SHARED uniform for
  (i) floor-based SR on a uniform grid (residue f = x - floor(x)), and
  (ii) sign-magnitude SR (what FP formats / PTX cvt.rs do: round |x| toward or away
       from zero), on an NVFP4-like quantizer (E2M1 grid, 16-elem blocks, scale=amax/6),
  as a function of the gradient mean/std ratio m_g (sign bias across ranks).
  Uniform-residue prediction for (ii): rho ~ (1/2) * E[sign]^2 = (1/2)(2 Phi(m_g) - 1)^2.
Part C: Wgrad transfer. Rank r computes E_r = (Q(dY_r) - dY_r)^T X_r (sum over T tokens),
  E = mean_r E_r. Prediction per (gradient feature i, input channel k), x_k = mu_k + z:
    Var(E)/(T v) = (mu^2 + 1)/P + (1 - 1/P) * rho(m_g) * mu^2      (shared u)
  We report the plateau ratio Var(P=256)/Var(P=16) (pure 1/P scaling = 0.0625).
All seeds fixed. Runtime well under 2 minutes on CPU.
"""
import numpy as np

rng = np.random.default_rng(12345)


def draw_u(mode, shape, P, rng):
    """Uniforms of shape (P,)+shape for the three randomness modes."""
    if mode == "independent":
        return rng.random((P,) + shape)
    if mode == "shared":
        return np.broadcast_to(rng.random(shape), (P,) + shape)
    if mode == "stratified":  # Suresh et al. 2022: U_r = pi_r/P + gamma_r
        perm = np.argsort(rng.random((P,) + shape), axis=0)
        return (perm + rng.random((P,) + shape)) / P
    raise ValueError(mode)


# ---------------- Part A ----------------
print("=== Part A: scalar SR, iid U(0,1) residues; P*Var(mean e) ===")
print(f"{'P':>5} {'mode':>12} {'P*Var(MC)':>11} {'P*Var(theory)':>14}")
theory = {
    "independent": lambda P: 1 / 6,
    "shared": lambda P: P / 12 + 1 / 12,
    "stratified": lambda P: 1 / 12 + 1 / (12 * P),
}
M = 200_000
for P in (8, 64, 512):
    m = M if P <= 64 else 40_000
    for mode in ("independent", "shared", "stratified"):
        f = rng.random((P, m))
        u = draw_u(mode, (m,), P, rng)
        e = (u < f).astype(np.float64) - f
        v = e.mean(axis=0).var()
        print(f"{P:>5} {mode:>12} {P*v:>11.4f} {theory[mode](P):>14.4f}")

# ---------------- NVFP4-like SR quantizer ----------------
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4_sr(a, u, axis_tokens=1, block=16):
    """a: (P, T, N) float; blocks of 16 along token axis (Wgrad reduction axis).
    u: uniforms same shape. Scale per block = amax/6 (kept in fp32; E4M3 scale
    rounding ignored). Returns dequantized SR value."""
    P, T, N = a.shape
    ab = a.reshape(P, T // block, block, N)
    amax = np.abs(ab).max(axis=2, keepdims=True)
    s = np.where(amax > 0, amax / 6.0, 1.0)
    y = np.abs(ab) / s
    idx = np.clip(np.searchsorted(GRID, y, side="right") - 1, 0, len(GRID) - 2)
    lo, hi = GRID[idx], GRID[idx + 1]
    frac = (y - lo) / (hi - lo)
    q = np.where(u.reshape(ab.shape) < frac, hi, lo)
    return (np.sign(ab) * q * s).reshape(P, T, N)


def hadamard16():
    H = np.array([[1.0]])
    for _ in range(4):
        H = np.block([[H, H], [H, -H]])
    return H / 4.0


H16 = hadamard16()
SIGNS = np.where(np.random.default_rng(7).random(16) < 0.5, -1.0, 1.0)  # fixed, shared


def rht_tokens(a):
    """Blockwise 16-pt RHT along token axis (axis 1) with fixed shared signs."""
    P, T, C = a.shape
    ab = a.reshape(P, T // 16, 16, C) * SIGNS[None, None, :, None]
    return np.einsum("ij,pbjc->pbic", H16, ab).reshape(P, T, C)


def grad_sample(dist, shape, rng):
    if dist == "gauss":
        return rng.standard_normal(shape)
    return rng.standard_t(3, shape)


# ---------------- Part B ----------------
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
print("\n=== Part B: cross-rank per-element error correlation, P=2 ===")
T, N = 4096, 256
g = rng.standard_normal((2, T, N))
for mode in ("shared", "independent", "stratified"):
    u = draw_u(mode, (T, N), 2, rng)
    step = 0.05
    f = g / step - np.floor(g / step)
    e = (u < f) - f
    print(f"floor-SR uniform grid, m_g=0   {mode:>12}: rho = {np.corrcoef(e[0].ravel(), e[1].ravel())[0,1]:+.4f}  (theory shared +0.5, strat ~ -1)")
print(f"{'m_g':>5} {'rho_shared(NVFP4 sign-mag)':>28} {'uniform-residue pred':>22} {'rho_strat':>10}")
RHO = {}
for m_g in (0.0, 0.25, 0.5, 1.0, 2.0, 1e9):
    gm = rng.standard_normal((2, T, N)) + (m_g if m_g < 1e8 else 0.0)
    if m_g >= 1e8:
        gm = np.abs(gm)  # all-positive limit
    out = []
    for mode in ("shared", "stratified"):
        u = draw_u(mode, (T, N), 2, rng)
        e = nvfp4_sr(gm, u) - gm
        out.append(np.corrcoef(e[0].ravel(), e[1].ravel())[0, 1])
    pred = 0.5 * (2 * Phi(m_g) - 1) ** 2 if m_g < 1e8 else 0.5
    RHO[m_g] = out[0]
    lab = "all+" if m_g >= 1e8 else f"{m_g:.2f}"
    print(f"{lab:>5} {out[0]:>28.4f} {pred:>22.4f} {out[1]:>10.4f}")

# ---------------- Part C ----------------
print("\n=== Part C: Wgrad SR-noise of rank-averaged NVFP4 gradient ===")
T, N = 512, 24
MG = np.array([0.0, 0.5, 2.0])        # gradient-feature mean/std (sign bias)
MUS = np.array([0.0, 3.0])            # activation-channel mean/std
mg_vec = np.repeat(MG, N // len(MG))
mu_vec = np.repeat(MUS, 8)
K = mu_vec.size
TRIALS = 8


def run_C(P, mode, rht=False):
    acc = np.zeros((N, K))
    v_acc = np.zeros(N)
    for _ in range(TRIALS):
        dY = mg_vec[None, None, :] + rng.standard_normal((P, T, N))
        X = mu_vec[None, None, :] + rng.standard_normal((P, T, K))
        if rht:
            dY, X = rht_tokens(dY), rht_tokens(X)
        u = draw_u(mode, (T, N), P, rng)
        e = nvfp4_sr(dY, u) - dY
        v_acc += (e ** 2).mean(axis=(0, 1))
        E = np.einsum("ptn,ptk->nk", e, X) / P
        acc += E ** 2
    v = v_acc / TRIALS
    var = acc / TRIALS / (T * v[:, None])
    return np.array([[var[np.ix_(mg_vec == a, mu_vec == b)].mean() for b in MUS] for a in MG])


res = {}
for P in (16, 256):
    for mode in ("independent", "shared", "stratified"):
        res[(P, mode)] = run_C(P, mode)
res[(16, "shared+RHT")] = run_C(16, "shared", rht=True)
res[(256, "shared+RHT")] = run_C(256, "shared", rht=True)
print("Var(E)/(T v)  [MC | shared-u prediction (mu^2+1)/P + (1-1/P) rho(m_g) mu^2]")
for P in (16, 256):
    for mode in ("independent", "shared", "stratified", "shared+RHT"):
        cells = []
        for a_i, a in enumerate(MG):
            for b_i, b in enumerate(MUS):
                pred = (b**2 + 1) / P + ((1 - 1 / P) * RHO[a] * b**2 if mode.startswith("shared") else 0.0)
                cells.append(f"mg={a},mu={b}: {res[(P, mode)][a_i, b_i]:.4f}|{pred:.4f}")
        print(f"P={P:>3} {mode:>12}  " + "  ".join(cells))
print("\nPlateau ratio Var(P=256)/Var(P=16)  (pure 1/P = 0.0625; full plateau -> ~1)")
for mode in ("independent", "shared", "stratified", "shared+RHT"):
    r = res[(256, mode)] / res[(16, mode)]
    print(f"{mode:>12}  " + "  ".join([f"mg={a},mu={b}: {r[i, j]:.4f}" for i, a in enumerate(MG) for j, b in enumerate(MUS)]))
r = res[(256, "independent")] / res[(256, "stratified")]
print(f"\nGain independent/stratified at P=256: " + "  ".join([f"mg={a},mu={b}: {r[i, j]:.3f}" for i, a in enumerate(MG) for j, b in enumerate(MUS)]))
r = res[(256, "shared")] / res[(256, "independent")]
print(f"Penalty shared/independent at P=256:  " + "  ".join([f"mg={a},mu={b}: {r[i, j]:.3f}" for i, a in enumerate(MG) for j, b in enumerate(MUS)]))
