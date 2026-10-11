"""Idea 4 numeric check: token-axis sorting vs RHT for block-scaled (NVFP4-like) Wgrad.

Wgrad = dY^T X, reduction over tokens t. Both operands are quantized column-wise
(along the token axis) in contiguous blocks of 16 tokens with an E2M1 element grid
and an E4M3-rounded (ceil) block scale times an FP32 tensor scale.

Synthetic model: dY[t,i] = a_t * c_i * g, X[t,k] = b_t * g'.
  a_t: token gradient scale (Pareto tail index alpha, or 'flat', or 'mix' forking model)
  b_t: activation token scale (1, or massive-activation tokens every 256 positions)
Arms (same transform on both operands, so the exact GEMM is unchanged):
  id        natural (random) token order
  rht16     random-sign 16-point Hadamard inside each 16-token block
  sort_dy   descending ||dY_t||
  sort_lev  descending leverage ||dY_t||*||X_t||
  sortlev+rht   sort_lev then rht16
Metrics: relative Frobenius MSE of Wgrad vs exact (SR and RTN), and the RTN zero
fraction of dY entries belonging to the bottom-50% tokens by ||dY_t||.
Claim tested (sharpened hypothesis (a),(b),(d)): sorting beats RHT on Wgrad MSE when
alpha < ~2; sort+RHT dominates both; sorting reduces zeroing of small-gradient tokens.
"""
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
B = 16
T, M, K = 2048, 32, 32
TRIALS = 8


def e4m3_ceil(v):
    v = np.maximum(v, 2.0 ** -9)
    e = np.floor(np.log2(v))
    e = np.maximum(e, -6)
    step = 2.0 ** (e - 3)
    return np.minimum(np.ceil(v / step - 1e-12) * step, 448.0)


def fp4_round(q, rng, mode):
    s = np.sign(q)
    a = np.minimum(np.abs(q), 6.0)
    idx = np.clip(np.searchsorted(E2M1, a, side="right") - 1, 0, len(E2M1) - 2)
    lo, hi = E2M1[idx], E2M1[idx + 1]
    p = (a - lo) / (hi - lo)
    if mode == "sr":
        up = rng.random(a.shape) < p
    else:
        up = p >= 0.5
    return s * np.where(up, hi, lo)


def quant_tokenaxis(Z, rng, mode):
    """Z: (T, C). Quantize each column in contiguous 16-token blocks."""
    Tn, C = Z.shape
    Zb = Z.reshape(Tn // B, B, C)
    amax = np.abs(Zb).max(axis=1, keepdims=True)
    gscale = np.abs(Z).max() / (6.0 * 448.0) + 1e-30
    sc = e4m3_ceil(amax / 6.0 / gscale) * gscale
    q = fp4_round(Zb / sc, rng, mode)
    return (q * sc).reshape(Tn, C)


H16 = np.array([[1.0]])
for _ in range(4):
    H16 = np.block([[H16, H16], [H16, -H16]])
H16 /= 4.0


def rht_blocks(Z, signs):
    Tn, C = Z.shape
    Zb = (Z * signs[:, None]).reshape(Tn // B, B, C)
    return np.einsum("ij,njc->nic", H16, Zb).reshape(Tn, C)


def token_scales(kind, rng):
    if kind == "flat":
        return np.ones(T)
    if kind == "mix":  # 5% forking tokens 30x larger
        a = np.ones(T)
        a[rng.random(T) < 0.05] = 30.0
        return a
    alpha = float(kind)
    return (1.0 - rng.random(T)) ** (-1.0 / alpha)  # Pareto(alpha), x_min=1


def run(akind, xkind, seed):
    rng = np.random.default_rng(seed)
    a = token_scales(akind, rng)
    c = np.exp(0.5 * rng.standard_normal(M))
    dY = a[:, None] * c[None, :] * rng.standard_normal((T, M))
    b = np.ones(T)
    if xkind == "massive":
        b[::256] = 50.0
    X = b[:, None] * rng.standard_normal((T, K))
    W = dY.T @ X
    nW = np.sum(W ** 2)
    ndy = np.linalg.norm(dY, axis=1)
    nx = np.linalg.norm(X, axis=1)
    small = ndy <= np.median(ndy)
    perms = {
        "id": np.arange(T),
        "sort_dy": np.argsort(-ndy),
        "sort_lev": np.argsort(-(ndy * nx)),
    }
    # 'lead': each of the T/16 largest-||dY_t|| tokens leads its own block (it sets the
    # scale and sits on the grid max), followers are the remaining tokens in descending
    # order, so large leaders get the largest followers (contiguous after the leader).
    srt = perms["sort_dy"]
    nb = T // B
    lead, fol = srt[:nb], srt[nb:].reshape(nb, B - 1)
    perms["lead"] = np.concatenate([lead[:, None], fol], axis=1).reshape(-1)
    signs = rng.choice([-1.0, 1.0], size=T)
    out = {}
    for mode in ("sr", "rtn"):
        for arm in ("id", "rht16", "sort_dy", "sort_lev", "sortlev+rht", "lead"):
            p = perms["sort_lev"] if arm == "sortlev+rht" else perms.get(arm, perms["id"])
            D, Xp = dY[p], X[p]
            if "rht" in arm:
                D, Xp = rht_blocks(D, signs), rht_blocks(Xp, signs)
            Dq = quant_tokenaxis(D, rng, mode)
            Xq = quant_tokenaxis(Xp, rng, mode)
            err = np.sum((Dq.T @ Xq - W) ** 2) / nW
            out[(mode, arm)] = err
            if mode == "rtn" and "rht" not in arm:
                sm = small[p]
                out[("zero", arm)] = np.mean(Dq[sm] == 0.0)
    return out


def main():
    arms = ("id", "rht16", "sort_dy", "sort_lev", "sortlev+rht", "lead")
    akinds = ["0.75", "1.0", "1.5", "2.0", "3.0", "5.0", "flat", "mix"]
    for xkind in ("plain", "massive"):
        cache = {ak: [run(ak, xkind, 1000 * s + 7) for s in range(TRIALS)] for ak in akinds}
        print(f"=== X token scales: {xkind} (T={T}, M=K={M}, block={B}, trials={TRIALS}) ===")
        print("relative Wgrad MSE ||What-W||^2/||W||^2 (mean over trials); ratio = arm/rht16;"
              " se% = std.err of the per-trial log-ratio sort_lev/rht16, in %")
        for mode in ("sr", "rtn"):
            print(f"-- {mode.upper()} --")
            print("a_t tail    " + "  ".join(f"{a:>12s}" for a in arms)
                  + "  sort_dy/rht  sort_lev/rht  sl+rht/rht  lead/rht   se%")
            for ak in akinds:
                res = cache[ak]
                m = {a: np.mean([r[(mode, a)] for r in res]) for a in arms}
                lr = np.log([r[(mode, "sort_lev")] / r[(mode, "rht16")] for r in res])
                se = 100 * lr.std(ddof=1) / np.sqrt(len(lr))
                print(f"{ak:>9s}   " + "  ".join(f"{m[a]:12.3e}" for a in arms)
                      + f"  {m['sort_dy']/m['rht16']:11.3f}  {m['sort_lev']/m['rht16']:12.3f}"
                      + f"  {m['sortlev+rht']/m['rht16']:10.3f}  {m['lead']/m['rht16']:8.3f}  {se:5.1f}")
        print("-- RTN zero fraction of dY entries in bottom-50% tokens (by ||dY_t||) --")
        print("a_t tail    " + "  ".join(f"{a:>10s}" for a in ("id", "sort_dy", "sort_lev", "lead")))
        for ak in akinds:
            res = cache[ak]
            print(f"{ak:>9s}   " + "  ".join(
                f"{np.mean([r[('zero', a)] for r in res]):10.4f}" for a in ("id", "sort_dy", "sort_lev", "lead")))
        print()


if __name__ == "__main__":
    main()
