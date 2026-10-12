# Idea 5 — PTQ calibration theory (sample complexity / MP, LW damping, covariate shift, VQ/rotation overfitting): citation-neighbourhood sweep (Angle C)

Date: 2026-10-10. Sources: arXiv API + full-text grep of fetched papers; WebSearch for 2026 material. OpenReview is behind a challenge page from this container (pfw3saHzGU could not be read; treated as unverified).

## Verdict in one paragraph

No paper applies Marchenko–Pastur / Vershynin covariance-estimation bounds or a Ledoit–Wolf-optimal damping to the GPTQ Hessian, and nobody has fitted a calibration–held-out gap against parameter count P and n_eff. But every *individual* building block of the idea now has a 2025–2026 paper sitting on it: (i) the decomposition "held-out proxy = calibration proxy + (w−q)ᵀ(E[zzᵀ] − XᵀX/m)(w−q)" is written out verbatim as Remark 3.8 of Zhang–Zhang–Colbert–Saab (2508.04853); (ii) the covariate-shift bound through a second-moment perturbation Δ is Theorem 3.1/3.2 of DRQ (2610.11226, 8 Oct 2026), which also proves (Thm B.2) that more calibration data cannot fix proxy/target mismatch; (iii) finite-sample noise in Ĥ and *linear shrinkage* of the Hessian citing Ledoit–Wolf is the whole motivation section of DASH-Q (2604.13806), with a direct measurement of ‖Ĥ_n − Ĥ_4096‖₁ for n = 8…2048; (iv) a finite-class uniform-convergence bound O(R²M²√(log|Q|/n)) on the calibration-vs-population reconstruction risk is Theorem 3.1 of SARQC (2605.05693); (v) GlowQ (2603.25385, ICLR 2026) actually swept LW-style shrinkage α ∈ {0, .02, .05} × N ∈ {32…256} and found ±0.01–0.02 ppl, i.e. no effect (though on a low-rank-correction covariance, not GPTQ's H); (vi) learned-rotation calibration overfitting is measured cross-domain in DartQuant Table 1 (PTB-calibrated vs Wiki-calibrated rotations: 38.24 vs 45.13 PTB ppl on LLaMA-2-7B) and codebook memorisation of calibration statistics is measured in 2604.08118 (1.04× in-domain → 3.49× out-of-domain degradation). Two results cut against the premise: Paglieri et al. (2405.20835) argue Williams–Aletras' variations are within 2σ and modern models are insensitive; SpinQuant's own ablations (Tables 11/13) claim 128 vs 800 samples and C4 vs Wiki make no significant difference. The remaining novelty is the *quantitative* unification (n_eff via token correlation, LW-optimal λ for H specifically, gap ∝ √(P/n_eff) across GPTQ/AWQ/SpinQuant/AQLM/QTIP, and ‖H_T − H_C‖_op as the single predictor of domain degradation) — and the risk flagged in the idea ("GPTQ is fine with 128 samples") now has three papers' worth of evidence behind it.

## Key papers named in the idea — what they actually say

| id | check |
|---|---|
| 2311.09755 Williams & Aletras (ACL 2024) | Confirmed: 1,800 compressed models (ten calibration sets × …, l.206); 128 × 2048 tokens (l.327); Fig. 3: PPL/zero-shot plateau after a few examples, SparseGPT keeps improving beyond 128 (+0.4% with 4×). The "GPTQ tends to overfit" sentence (l.171) is *attributed to Wu et al. 2023 = ZeroQuant(4+2), arXiv 2312.08583*, not their own finding. |
| 2608.01078 (ScaleQ-1.58 / AYOT) | Does **not** itself show "newer models are less sensitive". That sentence (l.1288–1290) is its related-work summary of **[69] Paglieri et al., 2405.20835** (ICML WS 2024). Cite the primary. Its own finding is the opposite direction: CoT-aware self-generated calibration (4M tokens) massively changes 1.58-bit reasoning accuracy. |
| 2601.18306 Chimoto et al. | Confirmed: multilingual mixes reduce PPL by up to 3.52 on Llama-3.1 GPTQ (l.25, l.128, l.980). |
| 2606.25519 Lian et al. | Confirmed: "raises accuracy from 28% to 65%" for Qwen3-4B INT3 GPTQ when calibrating on Math-CoT (l.1688, Fig. 5). |
| OpenReview pfw3saHzGU | Not reachable (api2.openreview.net returns ChallengeRequiredError; forum page is a JS shell). Unverified. |
| QuIP 2307.13304 | Confirmed: "a vast majority of H matrices have fewer than a quarter of eigenvalues > 1% of the max eigenvalue" (l.286). Theory (Thm 1, LDLQ optimality) takes H as *given*; no finite-sample statement. |

## Candidates (ranked)

### P1 — must deep-read

**2508.04853 — Provable Post-Training Quantization: Theoretical Analysis of OPTQ and Qronos** (H. Zhang, S. Zhang, I. Colbert, R. Saab; 2025-08). threat: partially_preempts.
- Remark 3.8 (l.994–1020) writes E_z|zᵀ(w−q_X)|² = (1/m)‖X(w−q_X)‖² + (w−q_X)ᵀ(E[zzᵀ] − XᵀX/m)(w−q_X) and says generalization "depends on the quality of the empirical estimate of the second-moment matrix". This is exactly the idea's eq. (1)/(4) decomposition. It stops there: no concentration bound, no n_eff, no MP.
- Thm 3.3 / Cor 3.5: OPTQ with damping λ ≡ OPTQ on X̂ = [X; √λ I]; ℓ₂ error bound in terms of σ_min of column sub-matrices and λ. Damping is analysed as a regulariser of the *deterministic* problem, not as covariance shrinkage.
- Conclusion (l.2202): "understanding when error bounds on calibration data also control error on unseen samples" is explicitly left as future work → the idea's part (1) is an open problem *as stated by them*.

**2610.11226 — When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization (DRQ)** (Y. Zhao et al.; 2026-10-08, 2 days old). threat: partially_preempts (part 4) + contradicts (framing of part 3).
- Thm 3.1: for second-moment H+Δ, Δ ⪰ 0, tr(CΔ) ≤ α, Ŵ₁ stays better than Ŵ₂ iff α‖(Ŵ₁−W)u‖² − ‖(Ŵ₂−W)u‖² ≤ L(Ŵ₂)−L(Ŵ₁) for all u with uᵀCu = 1. Thm 3.2: worst-case loss = calibration loss + α·max_u(...). This is the idea's covariate-shift term tr(ΔW(H_T−H_C)ΔWᵀ) with a budget instead of ‖H_T−H_C‖_op.
- Thm B.2 (l.1693): with i.i.d. calibration samples and a finite grid, the exact minimiser of the empirical reconstruction loss converges (SLLN) to the population minimiser, which can have strictly worse evaluation loss → "more calibration data need not improve the selected weights". Trivial proof but it directly argues the overfit gap is *not* a sample-size phenomenon.
- Empirics (l.160–166): of 560 per-method weight edits that lower calibration reconstruction loss, 73.8% (AWQ) and 47.1% (GPTQ) raise held-out WT2 PPL; Sec. 4.5/Fig. 4: calibration size 16–256 seqs × 2048 tokens, Llama-3.2-3B-Instruct GPTQ W4: reconstruction-only refinement lowers loss more as n grows yet PPL rises "by orders of magnitude".
- Not covered: Hessian estimation error, damping, rotation/codebook parameter count.

**2604.13806 — Robust Ultra Low-Bit PTQ via Stable Diagonal Curvature Estimate (DASH-Q)** (J. Kim, S. Kim, J. Lee, J. Seo; EuroMLSys '26, 2026-04). threat: partially_preempts (parts 1–2, empirically).
- Fig. 1(a): relative ℓ₁ error of Ĥ_n vs Ĥ_4096 for n = 8…2048 (Llama-2-7B layer 10, seq 2048): diagonal stabilises fast, off-diagonal "remains unstable even with much larger sets". Fig. 1(b): error between two independent 128-sample sets over 100 trials (a bootstrap estimate of Ĥ variance — essentially experiment (b) of the idea). Fig. 2: per-entry SNR = |E Ĥ_ij|/Std(Ĥ_ij), off-diagonal dominated by low-SNR tail.
- Defines the linear-shrinkage family H̃(ρ) = D + ρO (l.238–242) and cites Ledoit & Wolf [26]; concludes ρ ≈ 0 (diagonal-only, "increased bias but substantially reduced variance") and reports +7.01% avg / +14.01% max zero-shot over the strongest 2-bit baselines.
- Gap left: no theory for ρ or λ, no n_eff, shrinks toward diag(H) not toward a scaled identity, and the recommended fix (drop the off-diagonals) is the opposite of the idea's (keep them, damp optimally). A LW-optimal λ experiment would have to beat DASH-Q.

**2605.05693 — Saliency-Aware Regularized Quantization Calibration (SARQC)** (Y. Zhao et al.; 2026-05, same group as DRQ). threat: partially_preempts (part 3).
- Thm 3.1 (l.378–405, proof l.2285+): for the finite class Q_R = {Ŵ ∈ Q : ‖Ŵ−W‖_F ≤ R}, ‖X‖₂ ≤ M_X, n i.i.d. calibration inputs, w.p. ≥ 1−δ simultaneously over Q_R: R(Ŵ) − R̂_cal(Ŵ) ≤ R² M_X² √(log(2|Q_R|/δ)/(2n)). They concede "|Q_R| can be very large" and the bound "may be loose".
- This is the uniform-convergence step of idea part (3), but with class size = codebook grid (not rotation/codebook parameter count P), i.i.d. tokens (no n_eff), and no attempt to measure the gap. Table 8/F.5: SARQC "stable across different calibration sizes".

**2511.04063 — DartQuant** (Y. Shao et al.; NeurIPS 2025). threat: supports (part 3) / partially preempts experiment (d) for rotations.
- Table 1 "Impact of overfitting" (l.172–200): end-to-end learned rotations calibrated on Wiki/PTB/C4, evaluated on Wiki/PTB/C4, LLaMA-2-7B: FP 5.47/37.91/7.26; Wiki-calib 5.94/45.13/8.13; PTB-calib 6.02/38.24/8.13; C4-calib 6.05/44.99/8.02. LLaMA-2-13B analogous (PTB: 58.39 vs 49.14 vs 60.59). A ~7-ppl in-domain advantage on PTB is a clean learned-rotation overfitting signature; Wiki/C4 gaps are 0.1 ppl.
- Text (l.259–264): "end-to-end fine-tuning based on calibration sets ... tends to lead to overfitting on the calibration set [34, 27]"; motivation only. No P/n_eff analysis.

### P2 — read; adjust claims

**2603.25385 — GlowQ** (S. An, I. Suh, Y. Kim; ICLR 2026). threat: contradicts (part 2, weakly).
- App. D.1.1, Table 9 (l.3198–3340): Σ̂(α) = (1−α)Σ̂ + α·tr(Σ̂)/d·I (explicitly Ledoit–Wolf-style), α ∈ {0, 0.02, 0.05}, N ∈ {32, 64, 128, 256} seqs; WikiText-2 PPL "essentially flat": +0.01 ppl avg for Llama-3.2-3B/8B, −0.02/−0.01 ppl for Qwen3-8B/14B with shrinkage; N beyond 64 changes ≤ +0.01–0.02 ppl. Caveat: this is the covariance for a low-rank *correction* subspace (top-r eigenvectors), which is far less sensitive to small-eigenvalue noise than GPTQ's H⁻¹. Not a test of the idea's prediction, but a referee will cite it.

**2602.05902 — CoreQ** (S. Cha et al.; 2026-02). threat: partially_preempts (shrinkage framing).
- Thm 3.1 / A.1 (l.532–560, 1665–1690): for the GPTAQ-style mismatch correction S̃ estimated from finite calibration data (unbiased), excess population error R(α) = (1−α)²µ² + α²σ², optimal α* = µ²/(µ²+σ²) < 1; "YR without shrinkage can overfit finite-calibration error". A James–Stein-type argument for *one* calibration-estimated quantity. Not about H or λ, but it is the closest existing "optimal shrinkage from finite calibration" statement in PTQ.

**2602.07465 — MaCa: On the Importance of a Multi-Scale Calibration** (S. Son et al.; ICASSP 2026). threat: supports (n_eff conjecture).
- "regularizes each sequence as an independent sample" and mixes sequence lengths to build H; GPTQ zero-shot +2.46%p max, Qwen3-4B 4-bit 59.93→60.28 (l.502–510). Purely heuristic; no estimate of n_eff. It operationalises the idea's conjecture that the right sample unit is the sequence.

**2604.08118 — Initialisation Determines the Basin: Efficient Codebook Optimisation for Extreme LLM Quantization** (I. Kennedy, N. Moosavi; 2026-04). threat: supports (part 3 for VQ/AQ).
- Table 5 / l.918–928: additive-quantization baseline degradation "scales with domain distance from the C4 calibration set": 1.04× in-domain (C4) → 3.49× on WikiText-2; mechanism: when ρ = N/KM ≫ 1 "codebooks ... partially memorise calibration-specific statistics". l.555–557: wider beam search improves WikiText-2 but worsens C4 (18.64 → 19.00) "suggesting overfitting to the calibration objective". No P/n analysis.

**2405.20835 — Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs** (D. Paglieri et al.; ICML WS 2024). threat: contradicts (premise scale).
- l.366–375: Llama-2/3, Mistral-7B "significantly lower sensitivity to the nature of the calibration set compared to OPT 6.7B"; Williams–Aletras' variations "mostly fall within two standard deviations ... questioning the statistical significance". This is the primary source for the idea's "[2608.01078] newer models less sensitive" line. For GPTQ/AWQ at 4-bit the overfit signal may be below noise → the idea must target 2–3-bit and high-P methods.

**2510.10618 — Preserving LLM Capabilities through Calibration Data Curation** (B. He et al.; NeurIPS 2025). threat: contradicts (monotone-in-n story) / new_hypothesis.
- l.183–190: sweeping 16/32/64/128/256 samples, code generation *drops* with more samples: Qwen2.5-7B GPTQ 57.67% @16 → 34.03% @128; LLaMA3-8B AWQ 46.40% → 38.71%. Non-monotone in n contradicts a pure √(d/n_eff) excess-risk picture and points to covariate shift (part 4) dominating; the idea should predict this via ‖H_T − H_C‖_op rather than fit a monotone power law.

**2405.16406 — SpinQuant** (Z. Liu et al.; ICLR 2025). threat: contradicts (SpinQuant-overfits claim).
- l.1503–1510 & Table 11: 128 vs 800 samples "does not lead to a significant change in the perplexity"; Table 13: C4 vs Wiki calibration gives "consistent results". Calibration = 800 WikiText-2 samples, 100 Cayley iterations (confirms the idea's 800×2048 figure). Together with DartQuant Table 1 (cross-domain PTB gap) the picture is: in-distribution rotations are robust, cross-domain they overfit — i.e. the gap is covariate shift more than sample variance.

### P3 — context / tools

**2507.18553 — The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm** (J. Chen, T. Hoefler, D. Alistarh; ICLR 2026) and **2508.01077** (Birnick, ICLR 2026). threat: tangential/supports.
- Thm 5: ‖X diag(s)z − Xw‖² ≤ ¼ sᵀT⁻¹DT⁻¹s with D from the LDL of the (sample, damped) Hessian; damping λ = tr(XᵀX)/(100c). Deterministic in X — the natural place to plug a Weyl perturbation of D under H−Ĥ. Birnick l.595–598: insufficient regularisation of X gives large integer coordinates, "a symptom of overfitting to the calibration data X" (remark, no analysis).

**2608.21019 — Target-Aware Calibration Data Selection for Preserving Uncertainty** (Z. Yang et al.; EMNLP Findings 2026). threat: tangential (part 4).
- Prop 2/4: |R_T(D) − R_q(D)| ≤ TV(T, q) for bounded losses; coverage n ≥ log(1/δ)/ρ. A covariate-shift bound in total variation, weaker and non-quadratic compared with ‖H_T−H_C‖_op; shows the "mismatch bound" angle is already in print.

**2311.09755 — Williams & Aletras** — see table above; the overfitting attribution should be corrected to Wu et al. 2312.08583.

### Also seen (lower priority)
- 2501.13987 OSTQuant (ICLR 2025): "using CE loss results in overfitting on the calibration set" (Tab 8), fixed by KL-Top; supports part 3, P3.
- 2609.37416 (von Berg, 2026-09): Thm 1 high-probability uniform convergence of RTN loss under an *effective-rank* condition on calibration activations; RMT-adjacent but about grid scale, P4.
- 2604.24008 CoverCal (2026-04): 64 selected samples ≈ 256 random (AWQ INT4, LLaMA-3-8B); submodular coverage theory only, P4.
- 2609.36120 ThinQuant (2026-09): rotation learning with a few hundred selected activations; Thm 1 dimension-free bound for Haar rotations; says SpinQuant "potentially prone to overfitting"; P4.
- 2504.02692 GPTAQ (ICML 2025): k = 128×2048 ≈ 64× n for LLaMA2-7B (l.592–596) — the d/n ratio the idea needs is ~1/64 at token level, which is why n_eff (not n) is the crux.
- 2410.17170 Self-calibration (NAACL 2025), 2410.17711 Beware of calibration data for pruning (ICLR 2025), 2503.01820 RSQ (token importance), 2506.04985 FPTQuant (ICML 2026; 2603.04359 notes it as a case of calibrating on Wikitext and overfitting its characteristics), 2607.07964 KronQ, 2608.06291 FastKron (empirical-Fisher labels make quantization "more strongly adapted ('overfitted') to the calibration dataset", l.1030) — all tangential.
- Not found on arXiv: any paper with Marchenko–Pastur / Ledoit–Wolf *and* GPTQ/Hessian; `abs:Marchenko AND abs:quantization` returns only physics/IT papers; `abs:Ledoit AND (abs:Hessian OR abs:quantization)` returns nothing.

## Implications for the idea text
1. Cite 2508.04853 Remark 3.8 and DRQ Thm 3.1 as the existing decomposition/shift bounds; the contribution must be the *concentration* step (n_eff, κ_eff) and the P-dependence, not the decomposition.
2. Replace the 2608.01078 citation with 2405.20835 (Paglieri); keep 2608.01078 only for the CoT-calibration result.
3. The "fixed 1% damping is ad hoc shrinkage" paragraph must engage DASH-Q (diagonal-only wins at 2-bit) and GlowQ Table 9 (LW shrinkage ≈ no effect); the experiment (c) needs a DASH-Q arm.
4. Part (3)'s uniform-convergence bound exists in SARQC Thm 3.1 over the grid; the new content is parametrising |Q| by P (rotations d(d−1)/2, codebooks K·2^b·d_g) and measuring the gap — DartQuant Table 1 and 2604.08118 Table 5 already give two data points (rotations: ~7 PTB ppl cross-domain; AQ: 1.04×→3.49×).
5. DRQ Thm B.2 and 2510.10618's non-monotone-in-n results mean the success criterion "gap monotone in 1/n_eff" may fail for the wrong reason (proxy mismatch); separate the sample-variance term from the shift term in the experimental design.

## Queries run
arXiv API (relevance sort unless noted):
- abs:calibration AND abs:quantization AND abs:"sample complexity"
- abs:GPTQ AND abs:Hessian AND (abs:damping OR abs:regularization OR abs:estimation)
- abs:"calibration data" AND abs:quantization AND abs:overfitting
- abs:"random matrix" AND abs:quantization AND abs:Hessian
- abs:Marchenko AND abs:quantization
- abs:Ledoit AND (abs:Hessian OR abs:quantization)
- abs:calibration AND abs:quantization AND abs:"distribution shift" AND abs:LLM
- abs:"calibration set" AND abs:size AND abs:"post-training quantization"
- abs:rotation AND abs:quantization AND (abs:overfit OR abs:overfitting) AND abs:LLM
- abs:"calibration data" AND abs:quantization AND abs:LLM AND (abs:"how much" OR abs:"number of samples" OR abs:"sample size" OR abs:"few samples")
- abs:"calibration" AND abs:GPTQ AND (abs:generalization OR abs:"held-out" OR abs:"held out")
- abs:"shrinkage" AND abs:Hessian AND abs:quantization
- abs:"covariance estimation" AND abs:quantization AND abs:LLM
- abs:"effective sample size" AND abs:quantization
- abs:GPTQ AND abs:Babai
- abs:"calibration data" AND abs:quantization AND abs:LLM AND (abs:selection OR abs:selecting OR abs:"self-generated" OR abs:synthetic)
- abs:"calibration" AND abs:quantization AND abs:LLM AND (abs:sensitivity OR abs:robustness) AND abs:GPTQ
- abs:"learned rotation" AND abs:quantization
- abs:SpinQuant AND (abs:generalization OR abs:calibration)
- abs:quantization AND abs:LLM AND abs:"generalization bound" AND abs:calibration
- abs:"Hessian" AND abs:quantization AND abs:LLM AND (abs:"ill-conditioned" OR abs:"condition number" OR abs:"small eigenvalues")
- abs:"calibration" AND abs:quantization AND abs:LLM AND abs:"domain" AND (abs:mismatch OR abs:shift)
- ti:"Beyond fixed-length calibration"; ti:"diminishing effect" AND abs:quantization; ti:RSQ AND abs:quantization
- abs:GPTQ AND (abs:percdamp OR abs:"damping factor" OR abs:"dampening factor" OR abs:"damping ratio")
- abs:"synthetic" AND abs:"calibration data" AND abs:quantization AND abs:LLM AND (abs:pruning OR abs:GPTQ)
- abs:"calibration" AND abs:quantization AND abs:LLM AND abs:"sequence length" AND abs:tokens
- abs:quantization AND abs:LLM AND abs:"calibration" AND (abs:"bias-variance" OR abs:"variance" AND abs:"seed")
- abs:"vector quantization" AND abs:LLM AND abs:codebook AND (abs:overfit OR abs:overfitting OR abs:"calibration data")
- abs:"calibration-free" AND abs:LLM AND abs:quantization AND abs:Hessian
- abs:Hessian AND abs:"calibration data" AND abs:quantization AND abs:LLM AND (abs:noise OR abs:noisy OR abs:"estimation error")
- abs:quantization AND abs:LLM AND abs:"uniform convergence"
- abs:"post-training quantization" AND abs:statistical AND abs:calibration AND (abs:"estimation error" OR abs:"finite-sample" OR abs:"finite sample")
- abs:Wasserstein AND abs:calibration AND abs:quantization AND abs:LLM
- abs:"calibration" AND abs:quantization AND abs:LLM AND abs:"second-moment"
- abs:GPTQ AND abs:"calibration" AND abs:LLM AND (abs:"out-of-distribution" OR abs:"in-distribution")
- abs:quantization AND abs:LLM AND abs:calibration AND abs:"scaling law"
- abs:Hessian AND abs:shrinkage AND abs:quantization; abs:"linear shrinkage" AND abs:quantization
- abs:"off-diagonal" AND abs:Hessian AND abs:quantization AND abs:calibration
- abs:quantization AND abs:calibration AND abs:LLM AND abs:Hessian AND abs:eigenvalue
- abs:quantization AND abs:LLM AND abs:calibration AND (abs:"variable-length" OR abs:"sequence lengths") AND abs:Hessian
- abs:GPTQ AND abs:LLM AND (abs:"calibration data" OR abs:"calibration set") AND abs:stability
- abs:quantization AND abs:LLM AND abs:"calibration" AND (abs:"james-stein" OR abs:"ridge" OR abs:"tikhonov")
- abs:quantization AND abs:LLM AND abs:calibration AND (abs:"random seed" OR abs:"seed" AND abs:variance)
- abs:"Hessian" AND abs:"calibration" AND abs:quantization AND abs:LLM AND abs:"low-rank" AND abs:GPTQ
- abs:quantization AND abs:LLM AND abs:"calibration" AND abs:"influence function"
- abs:quantization AND abs:LLM AND abs:"calibration" AND abs:GPTQ AND abs:"held-out"
- ti:calibration AND abs:quantization AND (abs:LLM OR abs:"language model") --from 20250901 --sort date
- abs:quantization AND abs:LLM AND (abs:AQLM OR abs:QTIP OR abs:"vector quantization" OR abs:"trellis") AND abs:calibration AND (abs:generalization OR abs:overfit OR abs:overfitting)
- abs:"additive quantization" AND abs:LLM AND abs:calibration; abs:quantization AND abs:LLM AND abs:"PV-tuning"
- abs:quantization AND abs:LLM AND abs:calibration AND abs:"fine-tuning" AND abs:overfitting AND abs:"2-bit"
- abs:quantization AND abs:LLM AND abs:Hessian AND abs:"sampling noise"
- abs:quantization AND abs:LLM AND abs:"calibration" AND abs:"token correlation"
- abs:shrinkage AND abs:quantization AND abs:LLM; abs:"data-efficient" AND abs:"post-training quantization" AND abs:LLM
- abs:quantization AND abs:LLM AND abs:calibration AND abs:covariance AND abs:"low-rank"
- abs:quantization AND abs:LLM AND abs:"calibration" AND abs:"overfit" AND abs:rotation
- abs:quantization AND abs:LLM AND abs:"calibration" AND abs:"generalization gap"
- abs:quantization AND abs:LLM AND abs:calibration AND (abs:memorize OR abs:memorise OR abs:memorization)
- abs:quantization AND abs:LLM AND abs:codebook AND abs:calibration AND abs:domain
- abs:quantization AND abs:LLM AND abs:Hessian AND abs:calibration AND abs:concentration

WebSearch:
- "Ledoit-Wolf" GPTQ Hessian damping quantization LLM
- "Marchenko-Pastur" OR "random matrix theory" calibration Hessian GPTQ post-training quantization sample size
- calibration overfitting learned rotation SpinQuant vector quantization codebook held-out perplexity gap 2026
- "effective sample size" OR "effective number of samples" calibration tokens correlated GPTQ Hessian estimation quantization
- openreview pfw3saHzGU calibration data quantization GPTQ AWQ domain
- "calibration" quantization LLM GPTQ AWQ "domain" matching "does not help AWQ" OR "helps GPTQ" openreview 2025 calibration data domain-specific
- learned rotation quantization OSTQuant SpinQuant calibration "overfit" OR "overfitting" WikiText C4 generalization gap rotation training
- GPTQ Hessian "sample covariance" calibration noise "Marchenko" OR "random matrix" OR "shrinkage" blog 2026
- "shrinkage" GPTQ Hessian "identity" covariance quantization LLM arXiv 2026 calibration "Ledoit"

Full texts fetched and grepped: 2311.09755, 2608.01078, 2601.18306, 2606.25519, 2210.17323, 2307.13304, 2508.04853, 2507.18553, 2508.01077, 2610.11226, 2609.37416, 2603.16105, 2604.24008, 2603.04359, 2609.36120, 2602.05902, 2504.02692, 2508.18609, 2601.11200, 2405.20835, 2503.01820, 2506.04985, 2410.17711, 2410.17170, 2604.13806, 2602.07465, 2608.21019, 2605.05693, 2510.10618, 2511.04063, 2603.25385, 2405.14852, 2504.03717, 2604.08118, 2607.07964, 2608.06291, 2501.13987, 2405.16406.
