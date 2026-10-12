"""Idea 8 (T09, narrowed) synthetic check.

Claim under test (sharpened hypothesis):
  A perturbation shared by all G rollouts of a group (QeRL/AQN as coded: one
  draw per optimizer step) yields more zero-advantage (degenerate) groups than
  redrawing the SAME perturbation distribution per rollout chunk, and the drop
  is predicted by the convexity bound of 2605.07689 Prop. 2 applied to the
  across-draw variance Var(p') (cited, not claimed).

What is checked on synthetic data (logit-shift model of a perturbation):
  1. exact degeneracy for chunk counts N in {1,2,4,8} (N=1 shared, N=G iid);
     monotone decrease in N (power-mean argument, see write-up);
  2. second-order (Taylor) prediction vs exact gap; strong-convexity lower bound;
  3. how much of the total degeneracy is removable by per-rollout redraw
     (within-prompt perturbation part) vs not removable (prompt heterogeneity);
  4. estimating Var(p') from K draws x M rollouts with a binomial-noise correction;
  5. statistical power: groups needed to detect the removable gap;
  6. premise check: a DETERMINISTIC error (Var=0) can lower degeneracy.
Seeds fixed; runtime a few seconds.
"""
import numpy as np

rng = np.random.default_rng(20261011)
G = 8
NP = 4000          # prompts
K = 4000           # perturbation draws per prompt for "exact" expectations


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def D(p, g=G):
    return p ** g + (1.0 - p) ** g


# Prompt difficulty: bimodal logits, like RLVR math sets (many p~0 or p~1).
base = np.concatenate([rng.normal(-3.0, 1.5, NP // 3),
                       rng.normal(0.0, 1.0, NP // 3),
                       rng.normal(3.0, 1.5, NP - 2 * (NP // 3))])

print("=== 1-3. exact degeneracy, G=%d, %d prompts, %d draws/prompt ===" % (G, NP, K))
print("tau = std of the perturbation-induced logit shift per prompt per draw")
for tau in [0.1, 0.3, 0.6, 1.0]:
    z = rng.standard_normal((NP, K))
    p = sig(base[:, None] + tau * z)            # p'(x, draw)
    pbar = p.mean(1)
    res = {}
    for N in [1, 2, 4, 8]:
        m = G // N
        res[N] = ((p ** m).mean(1) ** N + ((1 - p) ** m).mean(1) ** N).mean()
    d_iid = D(pbar).mean()                       # = res[8] up to MC error
    var = p.var(1)
    taylor = (0.5 * G * (G - 1) * var * (pbar ** (G - 2) + (1 - pbar) ** (G - 2))).mean()
    lower = (0.5 * G * (G - 1) * var * 2.0 ** (3 - G)).mean()
    gap = res[1] - d_iid
    d_global = D(pbar.mean())
    mono = all(res[a] >= res[b] - 1e-12 for a, b in [(1, 2), (2, 4), (4, 8)])
    print(f"tau={tau:.1f}  D_shared(N=1)={res[1]:.4f}  N=2={res[2]:.4f}  N=4={res[4]:.4f}"
          f"  N=8(iid)={res[8]:.4f}  D(pbar_x)={d_iid:.4f}  monotone={mono}")
    print(f"         removable gap={gap*100:.3f} pp  Taylor pred={taylor*100:.3f} pp"
          f" (ratio {taylor/gap if gap>0 else float('nan'):.2f})  strong-convexity LB={lower*100:.4f} pp"
          f"  LB<=gap: {lower <= gap + 1e-12}")
    print(f"         share of N=2 in removable gap={(res[1]-res[2])/gap:.2f};"
          f"  D_iid(global pbar={pbar.mean():.3f})={d_global:.4f};"
          f"  removable/total-excess={gap/(res[1]-d_global):.3f}")

print()
print("=== 4. estimating Var(p') from K draws x M rollouts (tau=0.3) ===")
tau = 0.3
sub = base[rng.choice(NP, 512, replace=False)]
ztrue = rng.standard_normal((512, 20000))
ptrue = sig(sub[:, None] + tau * ztrue)
true_gap = (D(ptrue).mean(1) - D(ptrue.mean(1))).mean()
for Kd, M in [(8, 16), (16, 32), (32, 64)]:
    z = rng.standard_normal((512, Kd))
    pk = sig(sub[:, None] + tau * z)
    phat = rng.binomial(M, pk) / M
    s2 = phat.var(1, ddof=1)
    corr = (phat * (1 - phat)).mean(1) / (M - 1)
    vhat_naive = s2
    vhat = np.maximum(s2 - corr, 0.0)
    pb = phat.mean(1)
    w = 0.5 * G * (G - 1) * (pb ** (G - 2) + (1 - pb) ** (G - 2))
    print(f"K={Kd:2d} M={M:2d} ({Kd*M} rollouts/prompt): true gap={true_gap*100:.3f} pp"
          f"  pred naive={np.mean(w*vhat_naive)*100:.3f} pp  pred corrected={np.mean(w*vhat)*100:.3f} pp")

print()
print("=== 5. power: groups needed to see the removable gap (two-proportion z, 2 sigma) ===")
for tau in [0.1, 0.3, 0.6]:
    z = rng.standard_normal((NP, K))
    p = sig(base[:, None] + tau * z)
    d1 = ((p ** G).mean(1) + ((1 - p) ** G).mean(1)).mean()
    d8 = D(p.mean(1)).mean()
    gap = d1 - d8
    n = (2.0 * np.sqrt(d1 * (1 - d1) + d8 * (1 - d8)) / gap) ** 2
    print(f"tau={tau:.1f}: D_shared={d1:.4f} D_iid={d8:.4f} gap={gap*100:.3f} pp"
          f"  groups per arm for 2-sigma ~ {n:,.0f}"
          f"  (= {n/128:,.0f} steps at 128 prompts/step)")

print()
print("=== 6. premise check: deterministic error, Var(p')=0 ===")
for p0 in [0.95, 0.80, 0.50, 0.05]:
    for d in [-0.5, +0.5]:
        q = sig(np.log(p0 / (1 - p0)) + d)
        print(f"p={p0:.2f} logit shift {d:+.1f} -> p_Q={q:.3f}: D(p)={D(p0):.4f} D(p_Q)={D(q):.4f}"
              f"  {'lower' if D(q) < D(p0) else 'higher'}")
