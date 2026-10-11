# Idea 3 — Extreme-value-statistics theory of block-scaled micro-formats: citation-neighbourhood sweep (Angle C)

Date: 2026-10-10. Scope: prior art / contradicting evidence for "closed-form expected distortion of NVFP4 / MXFP4 / UE5M3 / INT4+scale as a function of (block size B, element grid, scale format, rounding direction) under Gaussian blocks, Gumbel block-max, Lloyd–Max grid; shrinkage bias of E2M1 RTN".

## Bottom line

The idea's stated literature gap ("format comparisons are all empirical; no quantitative theory") is **false as written**. Four papers already do substantial parts of the proposed theory:

1. **Fasoli et al., ICLR 2026 (2601.19026)** derive the expected MSE of FP4-E2M1 microscaling under i.i.d. Gaussian blocks *exactly*: half-normal order-statistic density of the block max (their eq. 5), conditional per-Voronoi-bin MSE of the E2M1 grid (eq. 2–3), integrated over x_max (eq. 4), then extended to **UE4M3-quantized scales** with three error terms (xi≠xmax, xi=xmax, all-zero block; eq. 6–10), INT4 elements (App. G) and FP6 scales (App. H). Agreement with simulation χ²≈4e-8; crossovers across block sizes 8/16 reproduced (App. F). They also *proposed UE5M3 scales* from this analysis. This is the (B, grid, FP-scale-format) part of Idea 3, done with exact order statistics rather than the Gumbel approximation.
2. **Soloveychik 2022 (2210.05470)** uses Fisher–Tippett–Gnedenko / Gumbel (Lemma 2, App.) on Gaussian blocks to bound inner-product error of BFP (integer mantissa + power-of-two scale with ceiling) and SBFP (exact scale) as functions of n and p (Props. 1–4), with explicit √(2 ln n) block-max growth and bound jumps "at those block sizes at which the expected values of the scales pass over integer powers of two"; derives optimal block size (n=64 at 4 bits, Sec. VI-B/VII-D). This is the extreme-value + optimal-B machinery.
3. **2605.20402 (MXFP4 error decomposition for RL)** uses the *identical* UE8M0 ceiling model γ_b = 2^U, U~Uniform[0,1) (Benford argument, App. A Step 1), E[γ]=1/ln2≈1.44, RMSE(γ−1)≈0.57 (eq. 9), proves exact orthogonality of deadzone error (Lemma 3.2), shows grid-noise floor invariant to scale precision, and argues scale bias compounds exponentially through depth in the backward pass. The −0.66 scale×grid anti-correlation is left as "an empirical constant that we do not derive in closed form" (Remark 3.3) — an explicit opening.
4. **MXAttention 2607.24377, Theorem 1**: for E2M1 + E8M0 under the ceiling-based scale family, the global MSE-optimal boundary is Q*max = 29/4 = 7.25 for *any* absolutely continuous law of log2 M — i.e. the optimal power-of-two scale rule is **independent of the block-maximum distribution** (periodicity/Fourier argument, App. D). It tabulates OCP-floor (efloor(8)) vs TetraJet TFS (eceil(6)) vs OAS (q=7) vs UOS. This partly contradicts the premise that extreme-value statistics of M_B are what decides the rounding rule; they decide only the MSE level.

Plus **UFP4 (2606.20381)** already formalizes shrinkage bias analytically: b_G(P)=E[ρ_G(t)−t], per-bin conditional bias (ℓ_i−r_i)/2 under locally uniform density (eq. 2), −0.125 for the (1.75,2.5) bin, and multiplicative accumulation ∏(1−δ_k)≈exp(−Σδ_k) (eq. 5, App. C). The idea text says "no quantitative theory" — wrong; what is missing is the Gaussian-integrated magnitude (idea's −0.66%, 0.9968 gain) and the bias-vs-variance depth criterion.

What is **not** found anywhere (surviving novelty, narrowed):
- A single closed-form (or 1-D-integral) comparison of NVFP4 (B=16, E4M3) vs MXFP4 (B=32, UE8M0 ceil/floor) vs UE5M3 vs INT4+scale on one footing, with the rounding-direction factor E[4^U]=3/(2 ln 2)≈2.16 and the floor-direction clipping term 0.098·2ln(2B)σ²/B stated explicitly. (2605.20402 has the first-moment version; nobody states the MSE factor or the floor-clipping term.)
- Any theory connecting UE8M0 floor-rounding clipping bias to the Mishra et al. divergence (2506.08027 App.: "rounding down, as in OCP v1 suggestion, will..." is empirical; NVIDIA CUDA `__nv_fp8_e8m0` conversion uses round-toward-+inf).
- Shrinkage bias under decreasing (Gaussian) density rather than locally-uniform density, and the (L·bias)² vs variance decision rule for SR vs RTN.
- The "real-data ratio 1.2 vs NVIDIA 1.67 gap" question. (2509.23202 Fig. 3 does measure E8M0 vs E4M3 MSE_rel across G on Llama-3.1-8B but gives no theory for the gap.)

Recommendation: reframe Idea 3 as an *extension* of Fasoli 2601.19026 + Soloveychik 2210.05470 to power-of-two scale rounding direction and rounding bias, explicitly citing 2605.20402 (γ=2^U model) and 2607.24377 (Theorem 1 distribution-independence). Drop "all comparisons are empirical". Also fix: UE5M3 originates with Hu et al. 2509.17791 and Fasoli 2601.19026, not 2609.02846 (which says so itself, lines 82/327).

## Candidates

| # | id | title (short) | published | threat | prio | evidence |
|---|----|---------------|-----------|--------|------|----------|
| 1 | 2601.19026 | Fasoli et al., Is Finer Better? Limits of Microscaling Formats (ICLR 2026) | 2026-01-26 | preempts | 1 | Exact Gaussian framework: f_xmax(θ)=2N/σ·(2Φ(θ/σ)−1)^{N−1}φ(θ/σ) (eq.5); E2M1 per-bin conditional MSE (eq.2–3); UE4M3-scale extension with 3 terms (eq.6–10); INT4 (App.G), FP6 scales (App.H); χ²≈2e-9 (BF16 scales) / 4e-8 (UE4M3); block-size crossovers (App.F); proposes UE5M3. Does NOT treat E8M0/PoT scales (only cites Rouhani that PoT "significantly degrades accuracy", l.225), rounding direction, or bias sign. |
| 2 | 2210.05470 | Soloveychik, Block Format Error Bounds and Optimal Block Size Selection | 2022-10-11 | partially_preempts | 1 | Gumbel via Fisher–Tippett (Lemma 2), sub-Gaussian tail bounds; Prop.2 BFP variance bound with ⌈log2⌉ ceiling of scale ⇒ jumps at powers of two; Prop.1/3/4 SBFP & high-dim; optimal n=64 at p=4 (Sec.VII-D, Fig.4); GPT2-XL validation. Integer mantissas only; no E2M1 grid, no FP8 scale formats, ceiling direction only. |
| 3 | 2605.20402 | Decomposing MXFP4 quantization error for LLM RL: reducible bias, recoverable deadzone, irreducible floor | 2026-05-19 | partially_preempts | 1 | Identical UE8M0 ceiling model γ_b=2^U, U~U[0,1) by Benford (App.A step 1); E[γ]=1/ln2≈1.44, RMSE(γ−1)≈0.57 (eq.9), empirically 0.57 on Qwen3-30B-A3B; Lemma 3.2 exact orthogonality DZ⊥scale, DZ⊥grid; MSE identity eq.2 with one cross term; cos(e_scale,e_grid)≈−0.66 "empirical constant we do not derive in closed form" (Remark 3.3); ‖e_scale‖²/‖e‖²≈1.72; scale bias compounds exponentially in backward pass (CLT over L layers). Idea's E[4^U]=2.16 is the second-moment version of their model. |
| 4 | 2607.24377 | MXAttention: Data-Free Optimal Scaling (UOS) for MXFP4 Attention | 2026-07-27 | partially_preempts / contradicts premise | 1 | Theorem 1: under ceiling-based scale family, for any absolutely continuous law of log2 M, global minimizer Q*max=29/4=7.25; Lemma 1 boundary-density ratio; "Neither a Gaussian nor a log-uniform prior is required" (l.629). Eq.11–14 formalize OCP floor efloor(8) → M/σ∈[4,8), TFS = eceil(6), OAS = q=7 (App.G). Table 1 compares rules at B=32. Empirical optimum 7.25 for 90–95% of Wan2.2 blocks. Implication: optimal PoT rounding rule is distribution-independent; extreme-value stats only set the MSE level. |
| 5 | 2606.20381 | Rethinking Shrinkage Bias in LLM FP4 Pretraining (UFP4) | 2026-06-18 | partially_preempts | 1 | Defines b_G(P)=E[ρ_G(t)−t]; eq.(2): E[ρ(t)−t | t∈B_i]=(ℓ_i−r_i)/2=(2q_i−q_{i−1}−q_{i+1})/4 under locally uniform density; −0.125 for q_i=2 bin (1.75,2.5), same at q=4 doubled; Key Finding 2 + eq.(5): ∏η_k=∏(1−δ_k)≈exp(−Σδ_k) multiplicative accumulation (App.C); RHT shifts regime to "local-resolution-limited". Idea's refinement (decreasing density, −0.66%, gain 0.9968, SR-vs-RTN depth criterion) is incremental. |
| 6 | 2605.08565 | Schaefer et al., Finer is Better (with the Right Scaling) | 2026-05-08 | partially_preempts | 2 | Gaussian simulation: 50,000×B samples, B∈{4,8,16,32}, σ over 151 points in [0.0005,0.05]; abs-max vs prevent-zero vs 4-over-6 vs brute-force optimal scale with E4M3 scale; three regimes A/B/C; inversion explained by block max of small blocks clustering near elements + coarse E4M3 step; Fig.7 clipping-only MSE. Covers the idea's Experiment 1 for E4M3 scales (numerically, no closed form). |
| 7 | 2509.23202 | Egiazarian et al., Bridging the Gap ... Microscaling FP4 (ICLR 2026) | 2025-09-27 | partially_preempts | 2 | Sec.3.1 "Analytical MSE bounds": Gaussian/Laplace MFP model with abs-max scale, Lemma 1 top-element MSE under Hadamard = per-element MSE; Lemma 2 dead-zone rates R_N(G)=Θ(√log G·G^{−δ²}), R_L(G)=Θ((log G)²G^{−δ}) ⇒ crossover in G; Fig.3 numerically compares S=E8M0 vs S=E4M3 over G=8..128 on Laplace and Llama-3.1-8B. Scale left unquantized in theory. Idea text's "all empirical" claim about this paper is wrong. |
| 8 | 2609.02846 | Hu, Luschi, Balanca, UE5M3 FP4 Block Scaling for Stable LM Pretraining | 2026-09-02 | supports | 3 | Mostly empirical; Prop.1 (unsaturated delayed-scale range); credits UE5M3 to Hu et al. [18]=2509.17791 and Fasoli [13]=2601.19026 (l.82, 327). App. analyses RTZ magnitude bias in ppm for BF16 K=4096 dots on Gaussian/Student-t/contaminated inputs (Table 18). Idea misattributes UE5M3 origin. |
| 9 | 2509.17791 | Hu et al., Elucidating the Design Space of FP4 training | 2025-09-22 | supports | 3 | Introduces UE5M3 as scale format; quantization-gradient framework (Prop.1, Prop.2 derivative of scale); notes "E8M0 has very large intervals ... leading to potential bias" (l.276); evaluates RTN vs SR for E4M3 and E8M0 scales. Empirical format sweep, no distortion theory. |
| 10 | 2506.08027 | Mishra et al., Recipes for Pre-training LLMs with MXFP8 | 2025-05-30 | supports | 3 | App.: expXint←ceil(expXfloat) (l.651); "rounding up X ensures ... no V/X value exceeds the maximum"; "rounding down, as in OCP v1 suggestion, will ..." (l.667–668); hardware log2 rounding caveat (l.1028). Purely empirical premise the idea seeks to explain. |
| 11 | 2512.02010 | Cook et al., Four Over Six | 2025-12-01 | supports | 3 | No theorem/lemma/Gaussian analysis found in full text; MSE vs MAE scale selection (Fig.5), biased 4/6 selection under SR preferred (Fig.7). Empirical heuristic the idea aims to derive. |
| 12 | 2608.25188 | Jokar et al., Transforms for LLM Quantization: The Great Inversion and Format Co-Design | 2026-08-25 | tangential | 3 | Survey/theory of shared-scale (AbsMax) quantization; Lemma 2.14/Prop.2.15: expected crest factor of a random rotation ≤ √(2 ln 2G) (same extreme-value quantity), "extreme-value, non-surrogate objective"; majorization theorems about within-group flattening. Relevant background for the "post-RHT Gaussian block" assumption, not a distortion formula. |
| 13 | 2605.12464 | Gupta et al., Search Your Block Floating Point Scales! | 2026-05-12 | tangential | 4 | ScaleSearch: fine-grained search over mantissa bits of MX scales to minimize error; shows abs-max scale is suboptimal; empirical (cited by MXAttention as [22]). |
| 14 | 2502.20566 | Ozkara, Yu, Park, Stochastic Rounding for LLM Training: Theory and Practice (AISTATS 2025) | 2025-02-27 | tangential | 4 | Thm.1–3: implicit regularization and convergence with SR vs nearest rounding; bias terms of nearest rounding in convergence bound. Relevant to the idea's (L·bias)² vs variance rule but not to block formats. |
| 15 | 2603.10444 | Cao et al., The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training | 2026-03-11 | tangential | 5 | "Mean bias" = rank-one columnwise activation mean inducing outliers (Thm.1 tail amplification), not rounding bias. Same words, different object; do not cite as shrinkage-bias prior art. |

Also seen, not listed: 2403.20137 (Trukhanov & Soloveychik 2024, K-sort block quantization, builds on #2, empirical); 2502.20853 TetraJet (TFS ceil(6) rule); 2510.27527 TetraJet-v2; 2605.12245 SOAR, 2606.07618 ScaleSweep, 2603.22370 FAAR (empirical NVFP4 scale/rounding optimization); 2602.00165 (Benford prior for PTQ); 2605.07662 (direction-preserving representations, geometric coverage theory); 2607.24953 (2D block FP4, transposition-invariant scaling, HiFloat4 group); 2602.02047 (NVFP4 outlier dynamics, block-level spikes).

## Queries run

arXiv API (arxiv_search.py, --sort relevance):
- abs:microscaling AND abs:quantization AND (abs:analysis OR abs:theory OR abs:optimal)
- abs:"block floating point" AND abs:quantization AND abs:analysis
- abs:"extreme value" AND abs:quantization
- abs:"shared exponent" AND abs:quantization AND abs:error
- (abs:MXFP4 OR abs:NVFP4) AND (abs:theory OR abs:theoretical OR abs:"closed-form" OR abs:analytical)
- abs:"scale factor" AND abs:rounding AND abs:FP8 AND abs:training
- abs:"shrinkage bias" OR (abs:"mean bias" AND abs:FP4)
- abs:"block size" AND abs:quantization AND abs:optimal AND abs:gaussian
- abs:"block floating point" AND (abs:error OR abs:analysis OR abs:theory)
- abs:"block-wise" AND abs:quantization AND abs:"mean squared error" AND (abs:gaussian OR abs:"closed-form")
- abs:"Lloyd-Max" AND (abs:FP4 OR abs:"floating point" OR abs:"floating-point")
- abs:Gumbel AND abs:quantization AND (abs:block OR abs:maximum OR abs:"max")
- au:Soloveychik AND abs:block
- abs:"shared scale" AND abs:quantization AND (abs:analysis OR abs:optimal OR abs:theory OR abs:theoretical)
- abs:E2M1 AND (abs:rounding OR abs:bias OR abs:grid)
- (abs:UE8M0 OR abs:E8M0) AND abs:quantization
- abs:"power-of-two" AND abs:scale AND abs:quantization AND (abs:bias OR abs:rounding) AND (abs:block OR abs:group OR abs:microscaling)
- abs:"optimal clipping" AND abs:quantization AND (abs:gaussian OR abs:laplace OR abs:analytical)
- abs:"group size" AND abs:quantization AND (abs:theory OR abs:theoretical OR abs:"closed-form" OR abs:analytical)
- abs:"stochastic rounding" AND (abs:FP4 OR abs:"4-bit") AND abs:bias AND abs:training
- abs:"block quantization" AND abs:gaussian AND abs:error AND (abs:theoretical OR abs:analytical OR abs:bound)
- au:Ozkara AND abs:"stochastic rounding"
- abs:"scaling factor" AND abs:quantization AND abs:"block maximum"
- abs:"maximum" AND abs:"block" AND abs:"FP4" AND abs:"distribution" AND abs:"error"
- abs:"crest factor" AND abs:quantization AND (abs:block OR abs:group OR abs:scale)
- abs:Benford AND abs:quantization AND (abs:scale OR abs:exponent)
- abs:"expected maximum" AND abs:quantization
- abs:"optimal scaling" AND abs:MXFP4
- abs:"Truncation-Free" OR (abs:TetraJet)
- abs:microscaling AND abs:"error bound"

WebSearch:
- "extreme value" theory block scaling quantization NVFP4 MXFP4 closed-form error block size scale format
- Gumbel block maximum microscaling quantization error analysis E2M1 E4M3 UE8M0 2026
- UE8M0 scale exponent round up round down MXFP8 divergence theoretical analysis power-of-two scale rounding direction
- "shrinkage bias" FP4 E2M1 round-to-nearest non-uniform grid expected rounding error theory
- NVFP4 vs MXFP4 quantization error theory "block size" 16 32 "E4M3 scale" analytical comparison Gaussian blog 2026
- Soloveychik "Block Format Error Bounds and Optimal Block Size Selection" extreme value Gumbel citing microscaling MXFP4 NVFP4 2025 2026
(No 2026 blog post or preprint applying Gumbel/extreme-value theory to NVFP4/MXFP4 was found; CUDA docs confirm `__nv_fp8_e8m0` converts with round-toward-+inf.)

Full texts fetched (arxiv_fetch.py): 2509.23202, 2603.08713, 2509.25149, 2506.08027, 2609.02846, 2512.02010, 2604.08826, 2606.20381, 2501.02423, 2606.07116, 2210.05470, 2603.10444, 2605.20402, 2605.08565, 2607.24377, 2509.17791, 2603.22370, 2605.07662, 2608.25188, 2607.24953, 2605.12464, 2601.09555, 2602.02047, 2601.19026, 2502.20853, 2403.20137, 2609.32429, 2502.20566. Reference-section keyword scans done on the first 13 plus 2601.19026/2605.08565.
