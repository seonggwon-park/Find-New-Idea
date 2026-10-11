# New PTQ idea candidates — lens: PTQ algorithms, formats, theory (generated 2026-10-11)

Scope: GPTQ/LDLQ/lattice/VQ, rotations/transforms, block-scaled formats, calibration, mixed precision, rate–distortion and proxy-to-loss bridges, scaling laws.
Excluded (already preempted, see `reports/LLM 양자화 연구 아이디어 10선.md` §0, §0.4): softmax −σ²/2 bias, recurrent-state error feedback, EVT/Gaussian block-format distortion, FP4 LR-decay floors, OPTQ calibration sample complexity (2609.31560), entropy-gated IS for quantized RL, quantization-cell theorem, entropy/Fisher-weighted calibration, pre/post-RoPE, MLA latent transform coding.

Method. Gaps were mined from `research_notes/arxiv_sweep/theorems/*.md` (limitations sections), `recent/*.md`, and full text of 2608.25188 (transform survey), 2605.13768 (High-Rate Quantized MatMul II), 2603.04956 (WaterSIC), 2201.11113 (GPFQ finite alphabet). Every candidate was screened with ≥4 `arxiv_search.py` queries (≥2 with `--sort date --from 20250601`, ≥1 adjacent-field vocabulary); queries are listed per section. Three toy numerical checks were run (scripts in the session scratchpad `sim/loopgain.py`, `sim/coherent.py`, `sim/schur.py`; synthetic Hessians, n=512 inputs, 256 rows, INT absmax, group 128). Toy numbers are labelled as such; nothing below is an LLM measurement.

Two candidates were generated and **dropped** during screening (recorded at the end): survivor/K-best search for GPTQ (preempted by OJBKQ 2602.08376 and CurveTQ 2610.09212) and Schur-complement column selection for mixed precision (no gain over OWQ in my toy check).

| # | Title | Sub-area | Provable core | Cost | Novelty risk |
|---|---|---|---|---|---|
| 1 | Loop gain of error-feedback rounding: low-rate overload of GPTQ/LDLQ under fixed-rate block scales | GPTQ theory, formats | yes (under independent-uniform error model / SR) | ~1 GPU-week | medium (GPFQ stability theory is the nearest) |
| 2 | Damping is a loop-gain stabilizer: the optimal GPTQ damping depends on bit-width, not only on sample size | calibration | partial | 3–5 GPU-days | low–medium |
| 3 | Worst-case gap between rotated GPTQ and WaterSIC grows like ½·log₂(1/ε); damping caps it | rate–distortion theory | yes (via Newton inequalities on O&P's pivot law) | CPU + 2 GPU-days | medium (stated open problem; small practical gap measured on Llama-3-8B) |
| 4 | SIC-aware WUSH: the AbsMax-optimal block transform under error feedback uses Schur-complement (pivot) blocks, not marginal blocks | rotations/transforms | yes (closed form under WUSH's own model) | ~1 GPU-week | medium (WUSH full text not read) |
| 5 | Closed-loop codebooks: LDLQ codebooks must be fit to the feedback-inflated quantizer input, not to the weights | VQ/lattice | yes (Gaussian case) | ~1 GPU-week | low–medium |
| 6 | Second-moment linearity theorem: rotation-seed variance is first order in the error (∝ 2^{−b}), mean is second order (∝ 4^{−b}) | proxy-to-loss, rotations | yes | 3–5 GPU-days | medium |
| 7 | Counteraction-aware calibration targets: asymmetric (GPTAQ) calibration cancels the network's own error counteraction | calibration, proxy-to-loss | yes (linearized block) | ~1 GPU-week | medium |
| 8 | Sequence-KL chain rule as the exact layer-sensitivity metric for reasoning models; the "perplexity illusion" is a domain effect | mixed-precision allocation | yes (identity) + empirical | ~1–2 GPU-weeks | medium–high |
| 9 | Overtraining raises the loop gain: GPTQ-vs-RTN degradation ratio grows with tokens/parameter | scaling laws | no (empirical, mechanism-predicted) | ~1 GPU-week | medium |
| 10 | Precision scaling laws in SQNR units: granularity confound behind Kumar et al.'s γ_w ≫ γ_kv | scaling laws | no (empirical; linearity-theorem motivated) | 1–2 GPU-weeks | medium–high |

---

## 1. Loop gain of error-feedback rounding: low-rate overload of GPTQ/LDLQ under fixed-rate block scales

**Stage:** hypothesis + toy evidence.

**Hypothesis.** In GPTQ/LDLQ, the value fed to the scalar quantizer at column j is z_j = w_j + Σ_{k<j} e_k U_{kj}/U_{kk} (U = upper Cholesky factor of (H+λI)^{-1}). Its variance is inflated to σ_w² + (Δ²/12)·g_j² with loop gain g_j² = Σ_{k<j}(U_{kj}/U_{kk})². When the block scale is computed from the *original* weights (standard for INT-g128, NVFP4, MXFP4), the probability that z_j leaves the representable range is controlled by Δ²g_j²/σ_w², not by the weight distribution. At 2–3 bits on coherent (un-rotated) Hessians this overload is large, and the clipped error re-enters the feedback, so error-feedback rounding loses its advantage. Incoherence processing works at 2 bits mainly because it shrinks max_j g_j², not because it Gaussianizes the weights.

**Gap evidence (citable).**
- 2507.18553 (GPTQ = Babai, ICLR 2026) §6 future work: "extending the analysis to clipped grids … are the immediate next steps", and asserts MXFP4/NVFP4 are "essentially no-clipping" because groups are small; §5: clipping "introduc[es] large errors that violate the error bound in Theorem 5". The no-clipping claim is about the weights, but the quantizer sees z, not w.
- 2605.13768 (O&P, HRQMM II) l.717–727: "In high-resolution quantization, we may assume Ŵ* ≈ W, and the overload probability is well approximated by Pr(W ∉ S)". That is exactly the assumption that fails at low rate. The same paper (l.1630) lists low-rate effects as open.
- 2508.04853 (Provable PTQ): finite alphabets are handled only heuristically (Remarks 3.7, 4.8); the deterministic case "cannot be made fully rigorous".
- 2608.25188 §7.2 quoting QuIP: at 2 bits the LDLQ rounder alone gives Llama-2-70B PPL 123.9 against 6.33 with incoherence processing, and the random permutation alone is "worth 74 perplexity" on OPT-125m. Both are unexplained by the high-rate theory, in which rotation and rounding are near-substitutes.
- 2406.11235 (QTIP) v1–v3: "Hessian-based adaptive rounding (GPTQ/LDLQ/BlockLDLQ) performs worse than expected on Llama 3", with no explanation.

**Toy evidence (synthetic, not LLM).** Coherent Hessian (power-law spectrum α=1.5 in a random basis, 16 coordinate-aligned channels ×1000, plus rank-1 shared component), INT absmax g128, scales from the original W:
- 2-bit, damping 1e-4·mean diag: 15.7% of quantizer inputs overload, input-variance inflation 14.3×, max g_j² ≈ 75, proxy (relative H-weighted error) 0.152. Damping 0.01: overload 0.5%, inflation 1.71, proxy 0.031. RTN: 0.548.
- Same spectrum in a random (incoherent) basis: overload ≤0.1%, inflation ≤1.36, and there is no overload pathology.
- Counter-example to an over-simple version: α=1.0 with 8 outliers ×1000 shows 5.6% overload at 2 bits, yet the smallest damping is still best (0.240 vs 0.264). Overload is therefore necessary but not sufficient, and the theorem must be quantitative.

**Math core.**
- *Proposition (exact under subtractive dither / SR, approximate for RTN).* If the rounding errors e_k are independent, zero-mean, with variance Δ_k²/12 and independent of w (exact for subtractive dither; this is the model Provable PTQ uses for ℓ∞ bounds), then E[z_j] = w_j and Var(z_j − w_j) = Σ_{k<j}(Δ_k²/12)(U_{kj}/U_{kk})². With a block range q·s fixed from W, Pr(overload at j) ≤ 2exp(−(q s − |w_j|)²/(2·Var)) (sub-Gaussian tail of a weighted sum of bounded variables, Hoeffding).
- *Theorem target (clipped Babai bound).* Under the same model, E‖(W−Ŵ)H^{1/2}‖² ≤ (1/12)Σ_j D_jj Δ_j² + Σ_j D_jj·E[(|z_j| − q s)_+²], and the second term is controlled by g_j² via the tail bound. This yields a finite-alphabet version of Babai's bound (2507.18553 Thm 5) with an explicit overload term.
- *Corollary (headroom).* The block scale that minimizes the bound is s* = max(|w|)/(q − κ·√(Σ_k U_kj²/U_kk²)/√12) for a constant κ set by the target overload probability. It is computable before quantization from the Cholesky factor at no extra pass.
- *Not provable:* the RTN (deterministic) case. The deterministic errors correlate with w, so the result is stated as an empirical law for RTN.

**Decisive experiment (≈1 GPU-week).** Llama-3-8B, Llama-2-7B and Qwen3-8B; GPTQ at INT2/INT3 g128 and NVFP4/MXFP4, without and with RHT. Log z_j for every column, together with g_j², overload rate and the clipped error mass. Compare (i) default scales, (ii) headroom scales s*, and (iii) post-SIC scales (rescale each block after its in-block feedback and re-quantize, a fixed point).
- *Prediction:* per-layer PPL damage correlates with Σ_j D_jj·overload_j (Spearman ≥0.6), and headroom scales recover ≥30% of the GPTQ-vs-RHT gap at 2 bits without rotation.
- *Failure criterion:* overload rate <1% in all layers at INT2 without rotation, or headroom scales change W2 PPL by <1%. Either result means the overload mechanism is irrelevant on real models.

**Nearest papers.**
- GPFQ 2010.15979 / 2201.11113 prove stability of a *different* greedy feedback quantizer under Gaussian data with ‖w‖∞ ≤ q_max; nothing on GPTQ's Cholesky loop gain or block scales.
- 2507.18553 and 2508.01077 assume no clipping. 2605.13768 assumes Ŵ*≈W for overload and uses post-SIC rectangular shaping only in the entropy-coded setting.
- JARQ 2609.38599 and SchurQuant 2608.15567 refit scales after rounding but give no overload theory.
- ADMM-Q 2605.11222 is an alternative solver.

**Self prior-art queries.**
1. `abs:GPTQ AND (abs:overload OR abs:clipping OR abs:saturation) AND (abs:low-bit OR abs:2-bit)` --sort date --from 20250601 → E-PMQ, ADMM-Q, coverage calibration (none on point)
2. `abs:"error feedback" AND abs:quantization AND (abs:stability OR abs:unstable) AND (abs:LLM OR abs:"language model")` --sort date --from 20250601 → none
3. `abs:"sigma-delta" AND abs:"neural network" AND abs:quantization AND abs:weights` (adjacent) → 2404.08131, 2405.01952 (not GPTQ)
4. `abs:"noise shaping" AND (abs:LLM OR abs:"language model" OR abs:"neural network") AND abs:quantization` (adjacent) → 2112.09181
5. `abs:OPTQ AND (abs:"finite alphabet" OR abs:"bounded alphabet")` → none
6. `abs:"greedy path following" AND abs:quantization` → GPFQ family (nearest, different algorithm)

---

## 2. Damping is a loop-gain stabilizer: the optimal GPTQ damping depends on bit-width

**Stage:** hypothesis + toy evidence; a contradiction between two theories.

**Hypothesis.** GPTQ's damping λ (default 1% of mean diag H) has two roles. The *generalization* role (2609.31560 Thm 3.1, Eq. 35: λ = β‖X‖_F²/(m^{1/3}min{m,N}^{2/3})) depends only on the sample size m. The *stability* role bounds the loop gain g_j² of idea 1, and its optimum grows as Δ grows, i.e. as bits drop. Prediction: at fixed m, λ*(2 bit) ≫ λ*(4 bit). The 2609.31560 prescription is then mis-specified at 2–3 bits, and a two-term rule λ* = max(λ_gen(m), λ_stab(b, H)) beats both.

**Gap evidence.**
- 2609.31560 (recent/20260801 note, item 13) derives λ from matrix Bernstein, independent of the bit-width, and validates it on synthetic data only.
- The GPTQ default damping is a constant across bit-widths.
- 2507.18553 §4.5: min-pivot ordering reduces tr(D) but "downstream accuracy gains are modest". The stability view predicts this, because tr(D) ignores the loop gain.

**Toy evidence.** The coherent setting of idea 1 gives a proxy-optimal damping of ≈0.01 at 2 bits (0.031 vs 0.152 at 1e-4) and ≤1e-4 at 3 and 4 bits (0.0022 vs 0.0037; 0.0003 vs 0.0007). In the incoherent setting, damping is monotonically harmful at every bit-width. The shift is therefore specific to coherent Hessians at low rate.

**Math core.**
- Since U = chol((H+λI)^{-1}), each g_j² is non-increasing in λ for typical (diagonally dominated) H. This is not true for all H and must be proved or restricted.
- The deployed proxy is then J(λ) = (1/12)Σ D_jj(λ)Δ² [increasing in λ: suboptimal compensation] + overload(λ, Δ) [decreasing in λ]. Under the bound of idea 1, ∂λ*/∂Δ > 0, which is a provable monotone-comparative-statics statement.
- The interaction with the sample-size term is empirical.

**Decisive experiment (3–5 GPU-days).** Full factorial on Llama-3-8B and Qwen3-8B, without rotation: λ ∈ {1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1} × bits {2, 3, 4} (INT g128) × calibration size m ∈ {128, 512, 2048} sequences. Report held-out C4/Wiki PPL.
- *Prediction:* argmin λ moves up by ≥10× from 4 to 2 bits at fixed m (stability view). The 2609.31560 view predicts no bit dependence.
- *Failure criterion:* λ* is the same within a factor of 3 across bit-widths at every m.

**Nearest papers.** 2609.31560 (λ for generalization; excluded idea, but the bit-dependence is not in it); GPTQ default; Provable PTQ 2508.04853 (sets λ = 0 for Qronos).

**Self prior-art queries.**
1. `(abs:GPTQ OR abs:OPTQ) AND (abs:dampening OR abs:damping)` --sort date --from 20250601 → none
2. `abs:"Hessian" AND abs:"regularization" AND abs:"post-training quantization" AND abs:"calibration" AND abs:"generalization"` --sort date --from 20250601 → none
3. `(abs:GPTQ OR abs:OPTQ) AND (abs:damping OR abs:dampening OR abs:"Tikhonov" OR abs:"ridge") AND (abs:"bit-width" OR abs:"2-bit" OR abs:"low-bit")` → none
4. `abs:"Levenberg" AND abs:quantization AND abs:Hessian AND abs:weights` (adjacent: LM-damping vocabulary) → none
5. `(abs:"WaterSIC" OR abs:"Cholesky pivots") AND abs:damping` → none

---

## 3. Worst-case gap between rotated GPTQ and WaterSIC grows like ½·log₂(1/ε); damping caps it

**Stage:** theorem target + numerical evidence.

**Hypothesis.** Over input spectra λ ∈ [ε, 1], the high-rate gap between GPTQ after a Haar rotation and WaterSIC is unbounded as ε→0. It grows as ½log₂(1/ε) − ½log₂log(1/ε) + O(1), attained by two-point spectra, and a damping λ_d caps it at the value for ε = λ_d/mean(λ). The ≈0.1-bit gap measured on Llama-3-8B (2605.13768) is therefore a property of those spectra, not of rotation. Layers with clustered near-null directions (few calibration tokens, massive-activation channels) should show gaps of tenths of a bit or more.

**Gap evidence.** 2605.13768 l.1594–1596: "It is an interesting open problem to estimate worst possible gap (over possible spectra λ_j ≥ ϵ) between GPTQ with rotation and WaterSIC". 2608.25188 §7.2 repeats: "the authors leave the worst-case gap between rotated GPTQ and WaterSIC open".

**Numerics (using O&P's pivot approximation Eq. 33: U_kk² ≈ k·e_k/((n−k+1)·e_{k−1}), m=200).**

| ε | worst two-point gap (best fraction of ε-mass) | log-uniform spectrum gap |
|---|---|---|
| 1e-1 | 0.045 bit (0.52) | 0.006 bit |
| 1e-2 | 0.47 bit (0.60) | 0.074 bit |
| 1e-3 | 1.23 bit (0.54) | 0.28 bit |

**Math core.** Write E_k = e_k(λ)/C(n,k) (normalized elementary symmetric means). Eq. 33 gives pivots r_k = E_k/E_{k−1}, so GM(r) = E_n^{1/n} = GM(λ), and the rotated-GPTQ/WaterSIC gap is ½log₂(AM(r)/GM(λ)).
- Newton's inequalities (E_k² ≥ E_{k−1}E_{k+1}) make r_k non-increasing, from r_1 = AM(λ) to r_n = HM(λ). This gives the upper bound gap ≤ ½log₂(AM(λ)/GM(λ)), the PCA-basis gap.
- For the two-point family {ε^{(f n)}, 1^{((1−f) n)}}, E_k has a closed form. As n→∞, AM(r) → (1−f)·c(f) + O(ε), so gap = ½ f log₂(1/ε) + ½log₂((1−f)c(f)) + o(1). Optimizing f gives the stated rate.
- Two steps remain to prove: (a) that two-point spectra are extremal, via a Schur-concavity argument on log E_k; (b) that Eq. 33 is a valid approximation, either as an expectation identity for volume-sampled orders or via a concentration statement for Haar rotation.

**Decisive experiment (CPU + ≈2 GPU-days).**
1. Compute the exact (not Eq. 33) pivots for 100 Haar rotations of synthetic two-point spectra to confirm the rate.
2. Collect all 224 linear-layer Hessians of Llama-3-8B and Qwen3-8B at m ∈ {32, 512} sequences; compute the predicted gap per layer and compare with measured GPTQ+RHT vs WaterSIC entropy-coded rate at equal distortion.
- *Prediction:* the per-layer gap rank-correlates with the predicted gap (ρ ≥ 0.7), and at m = 32 some layers exceed 0.3 bit.
- *Failure criterion:* the exact-pivot simulation shows the gap bounded by <0.3 bit for all ε ≥ 1e-4. In that case the theorem is false and the open problem has a benign answer.

**Honest assessment.** This is a clean answer to a stated open problem, but the practical margin is entropy-coded rate, which fixed-rate kernels do not inherit (2608.25188's own caveat). Venue: theory track or ISIT, unless it is combined with idea 2 (damping as the cap).

**Nearest papers.** 2605.13768 (open problem, Eq. 33 approximation); 2603.04956 WaterSIC (basis-free optimum, 0.25-bit space-filling gap); 2609.14706 WaterKron (Kronecker Hessians, high-rate).

**Self prior-art queries.**
1. `(abs:WaterSIC OR abs:waterfilling OR abs:"water-filling") AND abs:GPTQ AND (abs:rotation OR abs:"worst-case")` → only 2605.13768
2. `(abs:rotation OR abs:Hadamard) AND (abs:GPTQ OR abs:"error feedback" OR abs:"successive") AND (abs:"rate-distortion" OR abs:waterfilling OR abs:"water-filling" OR abs:"information-theoretic")` --sort date --from 20250601 → 2605.13768, 2603.11021 (Leech lattice; not on point)
3. `abs:"successive cancellation" AND abs:quantization AND abs:"random rotation" AND abs:gap` --sort date --from 20250601 → none
4. `abs:"Newton inequalities" OR (abs:"elementary symmetric" AND abs:Cholesky AND abs:"random rotation")` (adjacent: symmetric-function vocabulary) → only math papers
5. `au:Ordentlich AND au:Polyanskiy` --sort date → HRQMM I/II, WaterSIC, NestQuant (no worst-case result)

---

## 4. SIC-aware WUSH: under error feedback, the AbsMax-optimal block transform uses Schur-complement blocks

**Stage:** derivation target; WUSH full text NOT read (only the survey's statement of Thm 4.9). This is the first thing to check.

**Hypothesis.** WUSH (2512.00956) derives a closed-form data-aware block transform that is optimal for the AbsMax per-group RTN model with unbiased noise and "matrix columns treated as i.i.d. samples". It then applies the transform to GPTQ and "reports that the advantage transfers empirically" (2608.25188 Thm 4.9 text). Under SIC, the error of group g is weighted not by the marginal covariance block Σ_gg but by the block-LDL pivot D_g = Σ_gg − Σ_{g,>g}Σ_{>g,>g}^{-1}Σ_{>g,g}, the conditional covariance given the later groups. Hypothesis: replacing Σ_gg by D_g in WUSH's formula gives the transform that is optimal for (AbsMax + error feedback), and it beats WUSH+GPTQ by a margin that grows with the inter-group correlation.

**Gap evidence.**
- 2608.25188 §13 open problem 1: "Every optimality result in the survey holds one stage fixed. The KLT and WUSH optima … assume a given quantizer, and LDLQ's optimality is among rounders for a fixed basis … What is missing is a theory of the jointly optimal (T, Q) pair".
- CurveTQ 2610.09212 shows that the D-weights are "compute[d] but never read" inside blocks, but it works in the native basis without a transform.

**Math core.** Under block SIC with block size equal to the group size and the high-rate independent-error model, the deployed distortion is Σ_g E‖e_g‖²_{D_g} with e_g the group-g rounding error in transformed coordinates. A block-diagonal transform T_g acts on group g only. Because SIC within the transformed problem uses T_gᵀD_gT_g (block LDL is equivariant under block-diagonal transforms), the per-group objective is exactly WUSH's objective with Σ_gg → D_g. The WUSH optimum then transfers verbatim. This is a lemma, provable in a page once WUSH's Theorem is read. The non-trivial part is that D_g depends on the group order, so the ordering and the transform must be chosen jointly; a greedy order by det(D_g) is a natural heuristic.

**Decisive experiment (≈1 GPU-week).** Llama-3-8B and Qwen3-8B, MXFP4 and NVFP4 weight-only plus INT4 g32. Compare GPTQ with: none, block Hadamard (MR-GPTQ style), WUSH, and SIC-WUSH (pivot blocks). Report W2/C4 PPL and the layer proxy.
- *Prediction:* SIC-WUSH reduces the GPTQ proxy by ≥5% vs WUSH on layers with strong inter-group correlation (q/k/v, gate/up) and recovers ≥20% of the WUSH→BF16 PPL gap at MXFP4.
- *Failure criterion:* <2% proxy improvement or no PPL change. If WUSH's paper already uses conditional covariances, drop the idea.

**Nearest papers.** WUSH 2512.00956 (marginal blocks, RTN model); WUSH-KV 2609.38121; CurveTQ 2610.09212 (curvature in the Viterbi metric, no transform); MR-GPTQ (block rotations for FP4); 2605.13768 (rotation lowers AM of pivots).

**Self prior-art queries.**
1. `abs:rotation AND (abs:GPTQ OR abs:OPTQ) AND (abs:Cholesky OR abs:"LDL")` → none
2. `(abs:"learned rotation" OR abs:"rotation learning" OR abs:"optimized rotation") AND (abs:GPTQ OR abs:"error compensation" OR abs:"error feedback")` --sort date --from 20250601 → 2507.17417 (survey)
3. `abs:rotation AND abs:quantization AND (abs:"GPTQ-aware" OR abs:"rounding-aware" OR abs:"compensation-aware")` --sort date --from 20250601 → none
4. `abs:"basis reduction" AND abs:quantization AND (abs:"neural network" OR abs:LLM OR abs:weights)` (adjacent: lattice vocabulary) → Birnick 2508.01077 only
5. `ti:WUSH OR abs:WUSH` → 2512.00956, 2609.38121

---

## 5. Closed-loop codebooks: LDLQ codebooks should be fit to the feedback-inflated input law

**Stage:** hypothesis + toy evidence; explains a stated anomaly.

**Hypothesis.** In BlockLDLQ with a VQ codebook (QuIP#, AQLM-style initialisation, k-means), the codebook quantizes z = w + feedback, not w. The feedback inflates the input variance (1.06–1.36× at 2 bits even after an incoherent rotation in my toy; up to 14× without rotation) and makes it column-dependent through g_j². A codebook fit to the weight law (k-means on W) under-covers the tails. This is why E8P, which spends 29 "padding" codewords on norm² 12 shells, beats a lower-MSE k-means codebook end-to-end. Fitting the codebook to the closed-loop input law — Lloyd iterations on z recorded during an LDLQ pass, as in analysis-by-synthesis codebook design — and optionally scaling it per column by √(1 + (Δ²/12)g_j²/σ²) should beat both.

**Gap evidence.**
- 2402.04396 (QuIP#) App. C.3: "we observe that empirically, E8P actually outperforms K-means, which is somewhat interesting and suggests that allocating more information to the edge of the distribution, even after incoherence processing, is useful". Table 7 (Llama-2-70B, no FT, ctx 4096): E8P 2-bit 4.156/6.535 vs K-means 2-bit 4.452/6.925 (W2/C4). The paper offers no explanation.
- 2605.13768 assumes Ŵ*≈W for the shaping region (l.720).

**Math core.**
- *Proposition (Gaussian, independent-error model).* If w ~ N(0,σ²) and the feedback is a sum of independent errors, z_j ~ N(0, σ²(1+ρ_j)) with ρ_j = (D/12σ²)g_j² for per-dimension distortion D. A codebook C optimal for N(0,σ²) incurs, on z_j, the mismatch excess D_C(σ²(1+ρ_j)) − D*(σ²(1+ρ_j)). For a scaled copy of the optimal codebook, the excess is strictly positive whenever ρ_j > 0 and grows with the overload-region mass.
- Because LDLQ's distortion and ρ_j depend on each other, the closed-loop codebook is a fixed point. Convergence of alternating Lloyd/LDLQ is not guaranteed (as in closed-loop DPCM design); state it as an algorithm with a monotone-proxy safeguard.

**Decisive experiment (≈1 GPU-week).** Llama-2-7B and Llama-3-8B at 2 bits, dim-8 codebooks (k-means on W; k-means on recorded z; E8P; E8P rescaled per column), all with BlockLDLQ + RHT and no fine-tuning.
- *Prediction:* the order is closed-loop k-means ≤ E8P < open-loop k-means in W2 PPL, and the per-column rescaling alone recovers ≥50% of the E8P–k-means gap.
- *Failure criterion:* closed-loop k-means is no better than open-loop k-means (within 1% PPL). Then the C.3 anomaly has another cause, e.g. lattice structure or codeword-index effects.

**Nearest papers.** QuIP# 2402.04396 (anomaly); AQLM (codebooks learned on the output objective, effectively closed-loop but with beam search, not LDLQ); QTIP 2406.11235 (fixed Gaussian codes); Leech-lattice VQ 2603.11021.

**Self prior-art queries.**
1. `abs:codebook AND (abs:LDLQ OR abs:"error feedback" OR abs:"Hessian-aware rounding") AND abs:"vector quantization"` --sort date --from 20250601 → none
2. `abs:codebook AND abs:quantization AND abs:"language model" AND (abs:"k-means" OR abs:Lloyd) AND (abs:lattice OR abs:E8)` --sort date --from 20250601 → none
3. `abs:"lattice" AND abs:codebook AND abs:"language model" AND abs:"error feedback"` --sort date --from 20250601 → none
4. `abs:"closed-loop" AND abs:codebook AND (abs:design OR abs:training) AND (abs:predictive OR abs:DPCM)` (adjacent: predictive coding) → unrelated
5. `abs:"vector quantization" AND abs:LLM AND abs:codebook AND (abs:overload OR abs:"tail" OR abs:"edge of the distribution")` → unrelated

---

## 6. Second-moment linearity theorem: seed variance is first order, mean degradation is second order

**Stage:** theorem + cheap decisive test.

**Hypothesis.** The HIGGS linearity theorem predicts the *mean* PPL increase Σα_l t_l², and it removes the first-order term either through local optimality (A1) or through unbiasedness. Neither holds for the realized quantized model. For a single draw of the rotation or dither, ΔL = gᵀδ + ½δᵀ∇²Lδ, where g = ∇L at the deployed weights on the *evaluation* distribution. g is non-zero for post-trained (SFT/RL/chat) models evaluated on generic text, and for base models on downstream tasks. Hence Std_seed[ΔL] ≈ (Σ_l ‖g_l‖² ‖W_l‖² t_l²/d_l)^{1/2} ∝ 2^{−b}, while E[ΔL] ∝ 4^{−b}. Two predictions follow.
- (i) There is a crossover bit-width b× above which seed-to-seed fluctuation exceeds the mean degradation. This explains why quantized models "beat" BF16 on some tasks at 6–8 bits.
- (ii) SpinQuant's "up to 13 points" spread across 100 random rotations (W4A4 LLaMA-2-7B, Sec. 2.2) is first-order in weight-only settings. Seed selection on calibration data generalizes only to the extent that g_cal and g_eval are aligned, which is measurable as cos(g_cal, g_eval).

**Gap evidence.**
- HIGGS 2411.17525 Assumption 1 ("w⋆ local minimizer") and the remark that it is "not needed if the compression mechanism is unbiased". That remark is a statement about expectation only; no variance statement exists.
- SpinQuant 2405.16406 Sec. 2.2 / Fig. 4: the spread is reported but not explained (theorem note: "random-rotation sampling procedure unspecified", one model, one setting).
- 2609.21450 explains sign-pattern selection through activation-outlier interference (W4A4), not through a loss-gradient term.

**Math core.**
- *Theorem.* Let Ŵ_l = W_l + δ_l with δ_l = R_lᵀ ε_l R_l-type rotated errors (Haar or randomized Hadamard with random signs), independent across layers, E δ_l = 0 (subtractive dither, or rotation-symmetrized RTN), ‖δ_l‖_F² = t_l²‖W_l‖_F². Under A2–A3 of HIGGS, Var[ΔL] = Σ_l (t_l²‖W_l‖_F²/d_l)‖g_l‖² + O(t⁴) for Haar R; for Hadamard the constant changes by the sign-randomization factor.
- *Corollary.* The best-of-K seed improves the mean by ≈ σ_seed·√(2 ln K)·cos(g_cal, g_eval), so selection overfits when the cosine is small.
- Provable with standard Haar moment calculations; the Hadamard case needs the sub-Gaussian bound of 2608.25188 Prop. 2.15.

**Decisive experiment (3–5 GPU-days).** Llama-3.1-8B base vs instruct; RTN with 64 random-Hadamard seeds at weight-only INT {3, 4, 5, 6} g128; evaluate C4 NLL, plus GSM8K for instruct. Separately measure ‖g_l‖ by one backward pass.
- *Prediction:* the log₂ Std_seed vs b slope ≈ −1 and the log₂ mean vs b slope ≈ −2. Std is ≥3× larger for instruct on C4. Predicted Std from ‖g_l‖ matches measured within a factor of 2.
- *Failure criterion:* both slopes ≈ −2, or the predicted Std is off by more than 5×.

**Nearest papers.** HIGGS 2411.17525 (mean only); FOEM 2507.11017 (first-order term from accumulated deviation of latent weights during GPTQ, a different source); 2609.21450 (sign sampling, W4A4 outlier interference); 2601.11663 (activation sensitivity framework); 2609.07664 (divergence-based evaluation instability, empirical).

**Self prior-art queries.**
1. `abs:rotation AND abs:quantization AND (abs:seed OR abs:seeds) AND (abs:variance OR abs:lottery)` --sort date --from 20250601 → AXELRAM (unrelated)
2. `abs:quantization AND abs:"first-order" AND abs:gradient AND abs:Taylor AND (abs:LLM OR abs:"language model")` --sort date --from 20250601 → FOEM, SubRot, 2601.11663 (different contribution)
3. `abs:quantization AND abs:"random Hadamard" AND (abs:variance OR abs:"different seeds" OR abs:randomness)` --sort date --from 20250601 → 2609.21450 (nearest), CurveTQ, 2605.13810
4. `abs:quantization AND (abs:"run-to-run" OR abs:"seed variance" OR abs:"different random seeds") AND (abs:rotation OR abs:Hadamard)` --sort date --from 20250601 → none
5. `abs:"linearity theorem" AND abs:quantization` → only HIGGS
6. `abs:"post-training quantization" AND abs:"gradient" AND abs:"non-stationary"` (adjacent: optimization vocabulary) → none

---

## 7. Counteraction-aware calibration targets: asymmetric calibration cancels the network's own error counteraction

**Stage:** hypothesis with a linearized theorem.

**Hypothesis.** 2609.11716 shows that, in pretrained LLMs, the error a block newly introduces is anti-aligned with the inherited hidden-state error (negative inner product in Prop. 1), so errors partially cancel. The paper attributes PTQ robustness to this mechanism, which develops during pretraining. Asymmetric calibration (GPTAQ, ICML 2025) forces Ŵx̃ ≈ Wx: it removes the layer's linear response to the inherited error, and that response is part of the counteraction. In layers with strong counteraction, GPTAQ therefore over-corrects. The optimal target interpolates, Ŵx̃ ≈ Wx + α(Wx̃ − Wx), with α* = −κ/ρ determined by the measured counteraction cosine κ, not fixed at 0 (GPTAQ) or 1 (GPTQ).

**Gap evidence.**
- 2609.11716 App. (l.5439–5450): the counteraction cosine "changes little across calibration-free RTN and the calibration-based GPTQ and AWQ" (correlation >0.998). GPTAQ is not tested.
- GPTAQ 2504.02692 abstract: "we always match the quantized layer's output to the exact output in the full-precision model".
- No paper relates the two.

**Math core.** Linearize the residual block around the FP trajectory: the block's response to an inherited error Δh is BΔh, of which the quantized layer can cancel the part in its reachable span (projector P). Let α ∈ [0,1] be the fraction of that response the calibration target keeps (α=1: GPTQ's symmetric target Ŵx̃≈Wx̃; α=0: GPTAQ's Ŵx̃≈Wx). The propagated error is Δh⁺ = Δh + (I−P)BΔh + αPBΔh + e. If e is independent of α, minimizing ‖Δh⁺‖² gives α* = −⟨Δh + (I−P)BΔh, PBΔh⟩/‖PBΔh‖². In the simple case P = I this is α* = −κ/ρ, with κ the counteraction cosine between Δh and BΔh (the quantity 2609.11716 measures) and ρ = ‖BΔh‖/‖Δh‖. Therefore: (a) GPTAQ (α=0) is optimal only when κ ≥ 0 (no counteraction); (b) with counteraction (κ<0), some of the network's response should be kept; (c) α* > 1, i.e. a target that also cancels part of the inherited residual error, is optimal when −κ > ρ. The closed form is exact for the linearized model. That e does not depend on α is an assumption (GPTAQ's rounding error differs slightly from GPTQ's) and is checked empirically.

**Decisive experiment (≈1 GPU-week).** Qwen3-4B/8B and Llama-3-8B; GPTQ vs GPTAQ vs α-interpolated targets (α ∈ {0, 0.25, 0.5, 0.75, 1}) at INT3/INT2 g128 and NVFP4. Per layer, measure the counteraction cosine (2609.11716 protocol) and the predicted α*.
- *Prediction:* the best global α is strictly between 0 and 1. A per-layer α* from the formula beats both GPTQ and GPTAQ by ≥3% PPL at 3 bits. The layers where GPTAQ helps least are those with the most negative cosine.
- *Failure criterion:* GPTAQ (α=0) is best or tied at all bit-widths and no correlation with the cosine exists (|ρ| < 0.3).

**Nearest papers.** 2609.11716 (mechanism, RTN/GPTQ/AWQ only); GPTAQ 2504.02692; 2607.14630 (cross-layer compensation, exact for finite-difference A_l; abstract-level only); Model-Preserving Adaptive Rounding (Tseng et al. 2026, end-to-end target; not read).

**Self prior-art queries.**
1. `(abs:GPTQ OR abs:"post-training quantization") AND (abs:"asymmetric calibration" OR abs:"quantized inputs") AND (abs:"full-precision inputs" OR abs:"accumulated error")` → none
2. `abs:quantization AND (abs:"error cancellation" OR abs:"cancellation") AND abs:"inherited" AND abs:layer` --sort date --from 20250601 → only 2609.11716
3. `(abs:"sequential quantization" OR abs:"error propagation") AND abs:"post-training quantization" AND (abs:overcompensation OR abs:"over-compensation" OR abs:"partial compensation")` --sort date --from 20250601 → none
4. `abs:"Jacobian" AND abs:"quantization error" AND abs:propagation AND abs:transformer AND abs:contraction` --sort date --from 20250601 (adjacent: dynamical-systems vocabulary) → none

---

## 8. Sequence-KL chain rule as the exact layer-sensitivity metric for reasoning models

**Stage:** identity + empirical claim; highest preemption risk in this set.

**Hypothesis.** By the chain rule, KL(P_FP(y_{1:T}|x) ‖ P_Q(y_{1:T}|x)) = Σ_t E_{y_{<t}~P_FP}[KL_t]. Teacher-forced token KL on the FP model's *own* generations is therefore an unbiased estimator of the sequence-level divergence. The "perplexity illusion" (TASA 2607.00908: Kendall τ ≈ 0 between PPL-sensitive and reasoning-sensitive layers) should then vanish when per-layer sensitivity is measured as KL on self-generated reasoning traces instead of PPL on WikiText/C4. The illusion is a calibration-domain effect, not a failure of token-level metrics. TASA's "alignment–diversity trade-off" (target-task calibration alone can hurt) is predicted to disappear for sensitivity estimation, though not necessarily for Hessian estimation.

**Gap evidence.**
- 2607.00908 abstract: τ≈0 between PPL-ranked and reasoning-ranked layers; its remedy is a calibration-data search.
- 2606.00206 (FAIR) shows quantized reasoning models diverge mainly at uncertain positions; it measures KL along FP traces but not for bit allocation.

**Math core.**
- The chain-rule identity is exact.
- *Proposition:* if per-layer perturbations are small and independent, the sequence KL is additive across layers to second order (HIGGS-style), with weights α_l^{seq} = E over self-generated traces. The rank correlation between α^{seq} and the reasoning-accuracy sensitivity is then bounded below by a function of the accuracy-vs-KL slope. That part is empirical.
- Not provable: the link from KL to accuracy (task metrics are non-smooth).

**Decisive experiment (1–2 GPU-weeks).** DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B (thinking). Per-layer 3-bit sensitivity measured four ways: (a) WikiText PPL, (b) GSM8K/MATH accuracy (ground truth, expensive), (c) token KL on human-written math solutions, (d) token KL on the model's own traces. Then run allocations under a 3.5-bit budget from each.
- *Prediction:* τ(d, b) ≥ 0.5 while τ(a, b) ≈ 0 (reproducing TASA), and the allocation from (d) matches (b) within 1 point of accuracy.
- *Failure criterion:* τ(d, b) < 0.3. In that case token-level divergence is not the right proxy even on-distribution.

**Nearest papers.** TASA 2607.00908; 2609.26708 (on-policy distillation for low-bit reasoning, a QAT-side counterpart); 2608.01078 and 2405.20835 (self-generated calibration for Hessians, not for sensitivity ranking); 2609.07664 (divergence-based evaluation).

**Self prior-art queries.**
1. `(abs:"mixed-precision" OR abs:"bit allocation") AND abs:reasoning AND abs:quantization AND (abs:sensitivity OR abs:"layer-wise")` --sort date --from 20250601 → TASA, 2607.12266, others (none use self-generated KL)
2. `abs:quantization AND (abs:"self-generated" OR abs:"model-generated") AND (abs:"KL divergence" OR abs:sensitivity) AND abs:layer` --sort date --from 20250601 → unrelated
3. `abs:quantization AND abs:"exposure bias"` (adjacent: sequence-generation vocabulary) → 2609.26708 (QAT distillation) and vision papers
4. `abs:"perplexity" AND abs:quantization AND abs:"chain rule" AND abs:KL` → none
Residual risk: KL-on-generated-text evaluation is folklore in practice. The contribution has to be the allocation result and the dissolution of the τ≈0 finding.

---

## 9. Overtraining raises the loop gain: the GPTQ-vs-RTN degradation ratio grows with tokens/parameter

**Stage:** empirical hypothesis derived from idea 1's mechanism.

**Hypothesis.** The input covariance of linear layers becomes more redundant over training (marginal variance H_jj large relative to the conditional variance D_jj), so the loop gain g_j² grows with training tokens. Prediction: the benefit of error feedback (RTN damage / GPTQ damage) first rises with tokens/parameter at 4 bits and then *falls* at 2–3 bits as overload sets in. Part of the "overtrained models quantize worse" effect (Kumar 2411.04330 Eq. 2, γ_D = 0.51; Ouyang 2411.17691) is then specific to Hessian-based rounding and is not a property of the weights.

**Gap evidence.**
- Kumar App. H: the mechanism is "speculative".
- Ouyang's D axis is confounded with the LR-schedule phase (theorem note). Catalan-Tatjer 2510.06213 attributes the effect to LR decay using GPTQ only (theorem note: "weight-only PTQ (GPTQ, with AWQ/int8 checks)").
- No study splits the effect into RTN and GPTQ, or tracks Cholesky pivots over checkpoints.

**Math core.** None beyond idea 1's proposition, which links g_j² to overload. The scaling claim is empirical. One identity helps interpretation: Σ_j log D_jj = log det H is order-invariant, so a growing AM/GM of the pivots means growing redundancy.

**Decisive experiment (≈1 GPU-week).** OLMo-2-1B and 7B intermediate checkpoints (WSD/cosine phases labelled) plus Pythia-1.4B (143 checkpoints; control for the schedule by comparing equal-LR-phase pairs). At 10 checkpoints each, measure per-layer max_j g_j², AM/GM(D) and overload rate, and quantize with RTN, GPTQ, and GPTQ+RHT at INT2/3/4.
- *Prediction:* g_j² rises monotonically with tokens at fixed LR phase. At INT2 without rotation, the GPTQ/RTN damage ratio increases with tokens; with RHT, the trend flattens.
- *Failure criterion:* g_j² flat (±20%) over training, or RTN and GPTQ show the same token dependence (ratio flat within ±10%).

**Nearest papers.** 2510.06213 (LR dynamics, GPTQ); 2411.04330; 2411.17691; 2609.32060 (depth laws for the precision floor); 2605.23901 (Shannon scaling law with quantization perturbation; fitted, no rounding mechanism).

**Self prior-art queries.**
1. `(abs:overtrained OR abs:"training tokens") AND (abs:GPTQ OR abs:"post-training quantization") AND (abs:Hessian OR abs:covariance OR abs:conditioning)` --sort date --from 20250601 → none
2. `(abs:"post-training quantization" OR abs:PTQ) AND (abs:"training duration" OR abs:overtraining OR abs:"token budget") AND abs:degradation` --sort date --from 20250601 → QATFactory (unrelated)
3. `abs:"activation covariance" AND (abs:"condition number" OR abs:"effective rank" OR abs:anisotropy) AND abs:quantization` --sort date --from 20250601 → none
4. `abs:anisotropy AND abs:representations AND abs:"training dynamics" AND abs:"language model" AND abs:"effective rank"` --sort date --from 20250601 (adjacent: representation-geometry vocabulary) → none
5. `abs:"quantization" AND abs:"Pythia" AND abs:checkpoints AND (abs:GPTQ OR abs:RTN)` → none
6. `(abs:"conditional variance" OR abs:collinearity OR abs:redundancy) AND abs:features AND abs:"training tokens" AND abs:quantization` → none

---

## 10. Precision scaling laws in SQNR units: the granularity confound behind γ_w ≫ γ_kv

**Stage:** empirical reconciliation; medium–high risk.

**Hypothesis.** Kumar et al. fit N_eff = N∏(1 − e^{−P_x/γ_x}) with γ_w = 2.67, γ_a = 2.21, γ_kv = 0.96 (App. K table). The authors themselves flag that activations are quantized per-tensor and weights per-channel (App. L: "this could be because we quantize activations per-tensor and weights-per channel"). Hypothesis: once bit-width is replaced by measured SQNR (dB) of each tensor class under its granularity, the three curves collapse onto one N_eff(SQNR) with a single γ. Granularity, block size and format then enter the law only through SQNR. The linearity theorem motivates this, since loss depends on the format only through t² = 10^{−SQNR/10}. Deviations from the collapse would isolate genuinely non-MSE effects (outlier-specific or sink-specific errors).

**Gap evidence.**
- Kumar App. L (above) and App. D.2: no floating-point treatment.
- 2505.14302 and 2508.18609 fit granularity or group size as separate factors, so parameters multiply.
- KBBQ 2609.08135 shows SNR = 10log₁₀κ + C_fmt per format (matmul level), but makes no loss-scaling claim.

**Math core.** Not provable in general. Under HIGGS-style linearity with layer weights α_l, ΔL depends on the format only via t_l², which motivates the collapse to first order. The training-time version (QAT/precision-aware pretraining) has no theorem and is stated as empirical.

**Decisive experiment (1–2 GPU-weeks).** About 36 pretraining runs at 30M/85M parameters, 20 tokens per parameter (Kumar's setting scaled down): QAT forward with weights/activations/KV at {3, 4, 5, 6} bits × granularity {per-tensor, per-channel, block-32 E8M0, block-16 E4M3}, one class at a time. Measure the SQNR of each tensor class during training and fit (a) Kumar's form per class and (b) a single-γ SQNR form.
- *Prediction:* the single-γ SQNR form has held-out R² within 0.02 of the per-class form with 2–4 fewer parameters, and γ_kv/γ_w → ≈1 after conversion.
- *Failure criterion:* the per-class γ ratio stays above 1.8 after conversion. Then the KV sensitivity is intrinsic (e.g. attention-softmax amplification), which would itself be a finding.

**Nearest papers.** 2411.04330; 2505.14302 (QAT law with granularity); 2508.18609 (PTQ knowledge law with group size); 2501.02423 (FP training scaling law); 2605.23901 (Shannon/SNR scaling law, about training signal-to-noise, not tensor SQNR).

**Self prior-art queries.**
1. `abs:"scaling law" AND abs:quantization AND (abs:SNR OR abs:"signal-to-noise" OR abs:SQNR)` --sort date --from 20250601 → 2605.23901 (different notion of SNR)
2. `abs:"scaling law" AND abs:quantization AND (abs:granularity OR abs:"block size" OR abs:"group size")` --sort date --from 20250601 → 2508.18609 (factor-wise fit)
3. `abs:"scaling law" AND (abs:"per-tensor" OR abs:"per-channel" OR abs:"microscaling") AND abs:"training" AND abs:precision` --sort date --from 20250601 → none
4. `abs:"scaling law" AND abs:precision AND (abs:"noise injection" OR abs:"additive noise")` (adjacent: noise-injection vocabulary) → 2605.07546 (unrelated)
5. `(abs:"precision" OR abs:quantization) AND abs:"scaling law" AND (abs:"effective bits" OR abs:"quantization noise variance" OR abs:"noise variance")` --sort date --from 20250601 → none
6. `abs:"scaling law" AND abs:quantization AND abs:"mean squared error" AND abs:pretraining AND abs:precision` --sort date --from 20250601 → none

---

## Dropped during screening

- **Survivor / K-best / trellis noise-shaped search for GPTQ.** Preempted. OJBKQ 2602.08376 (Babai–Klein with K-best sampling on box-constrained ILS); CurveTQ 2610.09212 (LDL-diagonal weights placed in the Viterbi branch metric, "rotation-free trellis via curvature-weighted search"); JARQ 2609.38599 (bounded Babai proposals); CoreQ 2602.05902.
- **Schur-complement column selection for mixed precision** (keep the columns that minimize the pivots of the conditional covariance, instead of OWQ's H_jj‖w_j‖²). Toy check at 2 bits (keep 4 or 16 of 512 columns; 4 Hessian families): Schur 0.190/0.098, OWQ 0.190/0.099, OBS 0.225/0.100 on the first family, and nearly identical elsewhere. OBS was better in one family (0.0008 vs Schur 0.0015). The selected sets overlapped 50–80% with OWQ. There is no evidence of a gain, so the idea was dropped. SchurQuant 2608.15567 also already uses Schur-complement curvature for group decisions.
- **Pivot-flattening view of rotations** (rotation lowers AM(D) toward the rotation-invariant det^{1/n}). Preempted by O&P 2605.13768 §III and the 2608.25188 §7.2 summary. Idea 3 keeps only the stated open problem (the worst case).
- **Layer-0 / attention-sink Hessian weighting** (as an explanation of the QTIP Llama-3-70B layer-0 v collapse, which v4 fixes with a "\n" prefix). Likely covered by IntactKV (pivot-token KV) and attention-aware Hessians (BoA, APTQ). Not screened further.
