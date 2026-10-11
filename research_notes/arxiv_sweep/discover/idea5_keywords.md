# Idea 5 — prior-art sweep (keywords + adjacent-field vocabulary)

Idea: PTQ calibration theory — (1) sample complexity of Ĥ=(1/n)Σxxᵀ with n_eff≪n (token correlation), MP/Vershynin rate √(d/n_eff); (2) GPTQ 1% damping as ad-hoc Ledoit–Wolf shrinkage, optimal λ should scale with √(d/n_eff); (3) calibration-vs-held-out gap O(√(P/n_eff)) → learned rotations / VQ codebooks overfit more than GPTQ, part of the 2-bit SOTA margin (~0.3 PPL) is sample noise; (4) covariate shift bound tr(ΔW(H_T−H_C)ΔWᵀ) ≤ ‖H_T−H_C‖_op‖ΔW‖_F².

Date of sweep: 2026-10-10. Sources: arXiv API (46 queries, listed at the end), full texts under scratchpad/papers/<id>/, WebSearch for non-arXiv/venue checks. All quoted text is untrusted paper content.

## Verdict in one paragraph

No paper applies Marchenko–Pastur / random-matrix covariance-estimation theory or Ledoit–Wolf shrinkage to the GPTQ/LDLQ Hessian, and no paper models the effective sample size of correlated calibration tokens (queries Q4–Q6, Q16, Q23–Q24, Q26, Q28, Q30, Q44 returned nothing relevant; two web searches negative). **But three 2026 papers already occupy the neighbouring ground and together cover much of the "provable" part of the idea:** George & Saab 2609.31560 prove an empirical-to-population bound for OPTQ with a regularization-dependent factor and derive a *sample-size-dependent* λ formula that beats the fixed 1% rule in the low-sample regime (synthetic data only); DRQ 2610.11226 proves an exact condition for when lower calibration reconstruction loss gives higher loss under a bounded second-moment perturbation H+Δ (a DRO form of part 4) and shows empirically that more calibration data does not fix it; DASH-Q 2604.13806 measures Hessian estimation error vs. n (8→2048 samples), batch-to-batch instability of off-diagonal entries, and adopts a Ledoit–Wolf-style linear shrinkage family. Part (3) is partially covered by SARQC 2605.05693 (a finite-class uniform-convergence bound, √(log|Q|/n)) and by empirical overfitting tables for learned rotations (DartQuant Table 1, AMXFP4 Table 2). What remains open: the n_eff/token-correlation analysis, the MP/effective-rank-based rate for the *inverse* Hessian error that GPTQ actually uses, a closed-form LW-type λ validated on LLMs (George & Saab only test synthetic distributions), the cross-method "gap vs parameter count P" comparison (GPTQ vs SpinQuant vs AQLM/QTIP), and the operator-norm shift predictor evaluated against the alternative tr(H_test H_cal⁻¹) predictor of TASA 2607.00908.

## Citation corrections for the report

- **2608.01078 is mis-cited.** It is "Attend to Your Own Thoughts … 1.58-Bit Quantization" (ScaleQ-1.58, Intel China). The "newer LLMs (Llama-2/3) are significantly less sensitive to varying calibration data than OPT" claim is its reference [69] = **Paglieri, Dash, Rocktäschel, Parker-Holder, "Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs", arXiv 2405.20835 (ICML 2024 workshop)**. Replace the citation.
- **Williams & Aletras 2311.09755 do not themselves show "GPTQ overfits".** They attribute that to "Wu et al. (2023)" = **ZeroQuant(4+2), arXiv 2312.08583**, whose Table 1 shows C4-calibrated fine-grain GPTQ: C4 PPL 9.34 but PTB PPL 1399.89. What Williams & Aletras measure (1,800 models, 50 calibration sets per setting) is an accuracy *range* across calibration sets: GPTQ 0.9–1.6% (OPT 1.1–1.6%, LLaMA/Vicuna 0.9–1.3%), SpQR 0.6–1.0%, SparseGPT 2.4–4.8%, Wanda 0.6–2.9% (their Sec. 5 / Fig. 4). They explicitly say quantization is *less* sensitive than pruning.
- **OpenReview pfw3saHzGU** could not be fetched (DNS), but the sentence "The domain of calibration data affects GPTQ, but not the rest of the quantization algorithms" is in **arXiv 2504.04823, "Quantization Hurts Reasoning?" (COLM 2025)**, Sec. 5 — cite the arXiv id.
- 2606.25519 ("Quantization Inflates Reasoning") verified: Qwen3-4B INT3 GPTQ, Wikitext→Math-CoT calibration raises macro-average accuracy 28%→65% and cuts CoT length 30k→12k tokens (Fig. 5 text).
- 2601.18306 verified: EACL 2026, multilingual mixes up to 3.52 PPL reduction (abstract).

## Candidates (ranked)

### 1. 2609.31560 — George & Saab, "Generalization behavior of OPTQ and the role of regularization" (2026-09-25, UCSD) — **partially_preempts (1)+(2)**, priority 1
- Theorem 3.1 (empirical→population comparison): m i.i.d. samples with ‖x‖₂ ≤ R a.s., Σ=E[xxᵀ]; if λ > 2R² log(N/ε) then w.p. ≥1−ε, *simultaneously for all (data-dependent) r*: E|xᵀr|² ≤ (1 − √(2R² log(N/ε)/λ))⁻¹ · ((1/m)‖Xr‖² + λ‖r‖²) (their eq. 17). Not tied to OPTQ.
- Corollary 3.2 plugs in the Zhang–Colbert–Saab OPTQ calibration-error bound → population error ≤ factor × (Cδ²/m) Σ_i ‖P⊥_{>i} X̃_i‖².
- Theorem 3.3 / Corollaries 3.4–3.5: distribution-free bound for *stochastic* OPTQ for any finite-second-moment distribution regardless of calibration set, λ>0.
- Remark 4.1: recommended λ = β · ‖X‖_F² / (m^{1/3} · min{m,N}^{2/3}) (eq. 35), vs. GPTQ's λ₁ = 0.01‖X‖_F²/N and Qronos' λ₂ = 10⁻³‖X‖_op. Experiments (Figs. 1–3): synthetic distributions only (uniform Hadamard, non-uniform Hadamard, ReLU network), 30 trials, test error vs m; "existing recommendations … are susceptible to producing large errors in a low sampling regime", β=1 fixes that.
- Explicit framing in their intro: "OPTQ can make the error small on the calibration data by exploiting the empirical geometry of the sampled activations, but this empirical geometry may contain spurious correlations when the calibration set is small. Regularization plays an important role in our analysis, mainly to prevent this form of overfitting."
- Gaps left: no MP / effective-rank rate (their factor is R² log N / λ, a bounded-vector Bernstein argument, not √(d/n)); i.i.d. samples (no n_eff); no LLM experiments; no rotations/VQ; λ formula is bound-minimizing, not Ledoit–Wolf MSE-optimal. The report's experiment (c) must include this λ as a baseline, and the "optimal damping" contribution has to be positioned as LW-vs-George–Saab, validated on real models.

### 2. 2610.11226 — Zhao et al., "When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement (DRQ)" (2026-10-08) — **partially_preempts (4), evidence for (3)**, priority 1
- Theorem 3.1: for Ŵ₁, Ŵ₂ with L(Ŵ₁)<L(Ŵ₂), over input distributions with second moment H+Δ, Δ⪰0, tr(CΔ)≤α, Ŵ₁ stays no worse *iff* for every u with uᵀCu=1: α(‖(Ŵ₁−W)u‖² − ‖(Ŵ₂−W)u‖²) ≤ L(Ŵ₂)−L(Ŵ₁); otherwise Δ=αuuᵀ flips the order. This is the covariate-shift bound of part (4) in a trace-constrained (not operator-norm) ambiguity set, turned into a DRO objective.
- Theorem B.2: with i.i.d. calibration samples, the exact reconstruction minimizer converges a.s. to the population minimizer, whose evaluation loss can be strictly worse — "More calibration data need not improve the selected weights" (law of large numbers; weak but on-point).
- Empirics (Llama-3.2-1B-Instruct, W3): 18/32 integer-code modifications lower calibration loss but raise PPL *on the same calibration data*; of 560 lower-loss replacements, 413 (73.8%, AWQ) and 264 (47.1%, GPTQ) raise held-out WT2 PPL (Fig. 1); Sec. 4.5 sweeps calibration size 16→256 sequences (Llama-3.2-3B, GPTQ W4): reconstruction-only refinement lowers loss more as n grows yet "PPL increasing by orders of magnitude".
- Not covered: sample-complexity rate, damping, parameter-count argument.

### 3. 2604.13806 — Kim et al., "Robust Ultra Low-Bit PTQ via Stable Diagonal Curvature Estimate (DASH-Q)" (EuroMLSys '26) — **partially_preempts (1)+(2) empirically**, priority 1
- Measures ‖Ĥ_n − Ĥ_4096‖₁ separately for diagonal and off-diagonal entries for n = 8…2048 samples (Llama-2-7B layer 10, seq 2048; Fig. 1a): "diagonal entries stabilize quickly … off-diagonal entries remain unstable even with much larger sets". Fig. 1b: relative ℓ1 between two independent 128-sample sets over 100 trials; Fig. 2: SNR_ij = |E Ĥ_ij| / Std(Ĥ_ij) histogram, off-diagonal dominated by low-SNR tail.
- Defines a linear shrinkage family H̃(ρ)=D+ρO (cites Ledoit & Wolf 2004 as [26]); argues ‖O‖₁ ≫ ‖D‖₁ so any ρ>0 inherits the noise; chooses ρ≈0 (diagonal Hessian) + iterative weighted least squares. Reports +7.01% avg zero-shot at ultra-low bits and stability "with very small calibration data".
- Not covered: theory, optimal-λ formula, n_eff, rotation/VQ, covariate shift. But it is the closest *measurement* of part (1) and already frames damping as shrinkage.

### 4. 2607.00908 — Wang et al., "Beyond Activation Alignment: Alignment-Diversity Tradeoff in Task-Aware LLM Quantization (TASA)" (2026-07-01, preprint) — **partially_preempts (4)**, priority 2
- Appendix E.2 (OBS derivation): E_test[ΔL] ≈ (σ_q²/2) tr(H_test H_α⁻¹) — i.e., the test/calibration Hessian mismatch is a *trace-of-ratio*, competing with the report's ‖H_T−H_C‖_op. Theorem 1: f(α)=tr(H_test H_α⁻¹) is strictly convex with a unique optimal mixture α*∈(0,1) when H_task is ill-conditioned; Proposition 1 "Diversity as Hessian regularization"; Remark 1 ties diversity to effective rank. Table 19: per-layer Hessian effective rank / condition number under calibration compositions (e.g., κ 2143 vs 950, ER 3.3 vs 5.8 for pure-math vs wiki).
- Empirical "Alignment-Diversity Tradeoff": target-task-only calibration degrades, mixing general data "stabilizes sensitivity estimation".

### 5. 2605.05693 — Zhao et al., "Saliency-Aware Regularized Quantization Calibration (SARQC)" (2026-05-07) — **partially_preempts (3)**, priority 2
- Theorem 3.1: finite class Q_R={Ŵ: ‖Ŵ−W‖_F≤R}, ‖X‖₂≤M_X, i.i.d. tokens: R(Ŵ) − R̂_cal(Ŵ) ≤ R² M_X² √(log(2|Q_R|/δ)/(2n)) uniformly. A Hoeffding/union bound; the "generalization risk" framing of PTQ calibration is theirs now. Uses it to justify a weight-drift regularizer. Table 8: calibration-size ablation. No parameter count P, no n_eff, no cross-method comparison.

### 6. 2508.04853 — Zhang, Colbert, Saab, "Provable PTQ: Theoretical Analysis of OPTQ and Qronos" (2025-08) — **supports** (foundation), priority 2
- Theorem 3.3: ‖Xw−Xq‖² + λ‖w−q‖² ≤ (δ²/4)·N·C₂(X,λ)², C₂ depends on σ_min of trailing submatrices and λ; Corollary 3.5; Remark 3.4 justifies decreasing-norm ordering. All bounds are against the *fixed* calibration matrix (in-sample); λ guidance is about in-sample stability. Needed as the base for any population bound (George & Saab plug it in).

### 7. 2511.04063 — DartQuant (NeurIPS 2025) — **supports (3)**, priority 2
- Table 1 "Impact of overfitting", end-to-end learned rotation on LLaMA-2-7B calibrated on Wiki / PTB / C4 (FP: Wiki 5.47, PTB 37.91, C4 7.26): PTB PPL = 45.13 (Wiki-cal) / 38.24 (PTB-cal) / 44.99 (C4-cal); Wiki PPL = 5.94 / 6.02 / 6.05; C4 = 8.13 / 8.13 / 8.02. LLaMA-2-13B PTB = 58.39 / 49.14 / 60.59. Direct evidence that learned rotations fit the calibration distribution (≈7–10 PPL on PTB). Also claims SpinQuant/OSTQuant have good PPL but poor zero-shot, "overfitting risks associated with end-to-end fine-tuning". Related: ThinQuant 2609.36120 says SpinQuant is "potentially prone to overfitting to the fine-tuning objective" and shows rotations can be calibrated from orders-of-magnitude fewer activations spanning an r<d subspace (relevant to whether P=d²/2 is the right complexity measure — the effective problem is lower-dimensional); OSTQuant 2501.13987 App. A.4: CE loss "results in overfitting on the calibration set", KL-Top loss alleviates.

### 8. 2411.09909 — AMXFP4 (2024-11) — **supports (3)**, priority 3
- Table 2 "Impact of overfitting: calibration on different data distribution" (PubMed/Enron PPL, PIQA/WinoGrande acc; matched vs mismatched): "QuaRot with GPTQ and SpinQuant … tend to show better accuracy on data observed during calibration … results vary by about 2–3% solely due to different calibration datasets."

### 9. 2405.20835 — Paglieri et al., "Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs" (ICML 2024 WS) — **contradicts (3) for modern models / supports the "GPTQ needs only 128 samples" risk**, priority 2
- Correct source for "newer models are less sensitive" (the report cites 2608.01078, which only cites this as [69]). Not read in full here; must be deep-read to quantify the sensitivity decline and whether it covers 2–3-bit / learned-rotation / VQ methods (it covers GPTQ, AWQ, SmoothQuant at 4-bit per the citing text).

### 10. 2609.37416 — von Berg, Datres, Kneißl, Kutyniok, "Scale Sensitivity in Low-Bit PTQ: Curvature of the Quantization Error Landscape" (2026-09-29) — **tangential / supports**, priority 3
- Uses RMT concentration (Hanson–Wright, sub-gaussian operator norm, Theorems 2–4) with an effective-rank hypothesis on the calibration Gram (H2: r_eff(XXᵀ) ≥ C log⁸ d); Theorem 5 verifies it for wide random MLPs with isotropic Gaussian calibration and sample size growing polylogarithmically. Appendix: 16→256 calibration sequences do not close gaps between scale rules. Treats X as given — not a covariance-estimation analysis — but it is the one PTQ-theory paper that already makes the effective rank of the calibration Hessian the controlling quantity.

### 11. 2602.07465 — Son et al., "On the Importance of a Multi-Scale Calibration (MaCa)" (ICASSP 2026) — **supports (1) n_eff conjecture**, priority 3
- "treats each sequence as an independent sample" and regularizes per sequence in Hessian construction; Hessian "varies significantly with sequence length"; fixed budget 524,288 tokens (256×2048). The unit-of-independence choice is exactly the report's n_eff ≈ number of sequences conjecture, but no estimate of n_eff is given.

### 12. 2508.18609 — Zhou et al., "Task-Stratified Knowledge Scaling Laws for PTQ LLMs" (Findings ACL 2026) — **supports**, priority 3
- −ln(Acc_adj) = A·N^α (log₂B)^β (log₂C_b)^γ G^δ fitted on 293 configurations; calibration-size exponent γ = −0.040 for memorization ("nearly double that of … application"). Only empirical functional form in log₂ C_b; usable as a baseline for the (d/n_eff)^p fit in experiment (a).

### 13. 2609.30820 — Li, "Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness" (2026-09-25, Meta) — **supports (4)**, priority 4
- "Calibration blindness": step-0 Hessian leaves later-recurrence input directions nearly unweighted; one-step GPTQ worse than RTN on 5/9 checkpoints, accumulating H across steps recovers. A clean H_C ≠ H_T instance.

### 14. 2504.04823 — Liu et al., "Quantization Hurts Reasoning?" (COLM 2025) + 2606.25519 — **supports (4)**, priority 3
- "The domain of calibration data affects GPTQ, but not the rest of the quantization algorithms" (Sec. 5); and 2606.25519 Fig. 5 numbers above. Both purely empirical; no shift metric.

### 15. 2311.09755 — Williams & Aletras (ACL 2024) — **supports**, priority 2 (already in report; fix attribution per above)

Also seen, lower priority: 2601.21626 HeRo-Q (Tables 15–16: C4 vs WT2 and 64 vs 128 samples, PPL 6.87–7.01, "relatively stable"); 2510.10618 (NeurIPS 2025, representativeness/diversity in activation space); 2406.12928 (calibration same-distribution as test "not necessarily optimal"); 2604.24008 CoverCal (calibration quality as outlier-channel coverage; with 64 samples matches random-256); 2608.21019 DPQ (mixture-mismatch argument for calibration selection); 2609.14706 WaterKron (Kronecker Hessian chosen by covariance fitting; cites core-shrinkage covariance estimation); 2507.18553 GPTQ=Babai (ICLR 2026; damping only mentioned as full-rank fix); 2601.11663 (survey-style: "PTQ must estimate sensitivity using a limited calibration set, introducing statistical error").

## What this means for the idea

- Keep: the n_eff/token-correlation analysis (no paper), the MP/effective-rank rate on H⁻¹ and its amplification in small-eigenvalue directions (no paper; DASH-Q measures but does not derive), the cross-method gap-vs-P comparison (no paper), and the ‖H_T−H_C‖_op predictor *compared against* tr(H_T H_C⁻¹) (TASA) and DRQ's trace-constrained set.
- Reposition: "optimal damping" must be stated relative to George & Saab's λ ∝ ‖X‖_F²/(m^{1/3} min{m,N}^{2/3}) and DASH-Q's diagonal shrinkage; the novelty is an MSE-optimal (LW) λ with n_eff and an LLM validation, not the observation that 1% is ad hoc.
- Risk raised: Paglieri 2405.20835 and HeRo-Q suggest modern models at 4-bit are insensitive; the overfitting hypothesis must be tested at 2–3 bits with rotation/VQ methods, where DartQuant/AMXFP4 already show multi-PPL and 2–3% effects.
- Sample-complexity statement: use the Koltchinskii–Lounici effective-rank form ‖Ĥ−Σ‖ ≲ ‖Σ‖(√(r(Σ)/n_eff) + r(Σ)/n_eff) rather than d/n — von Berg et al. and TASA both make effective rank the operative quantity, and the activation Hessians are low-rank (QuIP), so the d/n version would be loose and easy to attack.

## Queries run (arXiv API; all with --max 40–80; dates as noted)

Q1 abs:calibration AND abs:quantization AND abs:"sample complexity" (2023+) — 0 hits
Q2 abs:GPTQ AND abs:Hessian AND (abs:damping OR abs:regularization OR abs:estimation)
Q3 abs:"calibration data" AND abs:quantization AND abs:overfitting
Q4 abs:"random matrix" AND abs:quantization AND abs:Hessian — 0
Q5 abs:"Marchenko" AND abs:quantization — 0
Q6 abs:"Ledoit" AND (abs:Hessian OR abs:quantization) — 0
Q7 abs:calibration AND abs:quantization AND abs:"distribution shift"
Q8 abs:"calibration set" AND abs:size AND abs:"post-training quantization"
Q9 abs:rotation AND abs:quantization AND (abs:overfit OR abs:overfitting OR abs:overfits)
Q10 abs:"calibration data" AND abs:quantization AND abs:"large language" AND (abs:sensitivity OR abs:robustness OR abs:impact)
Q11 abs:GPTQ AND (abs:overfit OR abs:overfitting OR abs:generalization) AND abs:calibration
Q12 abs:"calibration-free" AND abs:quantization AND abs:LLM
Q13 abs:quantization AND abs:Hessian AND (abs:"sample size" OR abs:"number of samples" OR abs:"few samples") AND abs:calibration — 0
Q14 abs:"covariance estimation" AND abs:quantization
Q15 abs:quantization AND abs:"vector quantization" AND abs:codebook AND abs:calibration AND (abs:overfit OR abs:overfitting OR abs:generalization)
Q16 abs:"calibration" AND abs:quantization AND abs:"effective sample" — 0
Q17 abs:"calibration data" AND abs:quantization AND abs:LLM --sort date --from 20250601
Q18 abs:GPTQ AND abs:Hessian --sort date --from 20250601
Q19 abs:calibration AND abs:quantization AND (abs:overfit OR abs:overfitting OR abs:"generalization gap") --sort date --from 20250601
Q20 abs:rotation AND abs:quantization AND abs:LLM AND abs:calibration --sort date --from 20250601
Q21 abs:"calibration" AND abs:quantization AND (abs:"domain shift" OR abs:"covariate shift" OR abs:"distribution mismatch" OR abs:"out-of-distribution")
Q22 abs:quantization AND abs:"post-training" AND (abs:"theoretical" OR abs:theory OR abs:bound) AND abs:calibration AND abs:Hessian
Q23 abs:"random matrix theory" AND abs:Hessian AND (abs:pruning OR abs:compression OR abs:quantization OR abs:"Optimal Brain") (2020+)
Q24 abs:shrinkage AND (abs:Hessian OR abs:Fisher OR abs:"second-order") AND (abs:pruning OR abs:quantization OR abs:compression OR abs:"natural gradient") (2020+)
Q25 abs:"effective rank" AND abs:quantization AND (abs:LLM OR abs:"language model" OR abs:transformer)
Q26 abs:"effective sample size" AND (abs:tokens OR abs:"language model") AND (abs:correlated OR abs:autocorrelation OR abs:calibration)
Q27 (abs:Rademacher OR abs:"uniform convergence" OR abs:"generalization bound") AND abs:quantization AND (abs:"post-training" OR abs:calibration OR abs:codebook)
Q28 (abs:"spiked covariance" OR abs:"sample covariance") AND (abs:"neural network" OR abs:LLM OR abs:transformer) AND (abs:quantization OR abs:pruning OR abs:compression) (2022+)
Q29 (abs:coreset OR abs:"importance sampling" OR abs:"data selection") AND abs:calibration AND abs:quantization AND (abs:LLM OR abs:"language model")
Q30 (abs:Tikhonov OR abs:ridge OR abs:"Levenberg") AND abs:Hessian AND (abs:quantization OR abs:"Optimal Brain" OR abs:pruning) AND (abs:LLM OR abs:"language model" OR abs:"neural network") — 0
Q31 au:Williams AND au:Aletras
Q32 (abs:"Optimal Brain Surgeon" OR abs:"OBS") AND abs:Hessian AND (abs:inverse OR abs:"ill-conditioned" OR abs:damping) AND abs:LLM — 0
Q33 abs:"calibration data" AND abs:quantization AND (abs:"less sensitive" OR abs:"robust to the calibration" OR abs:"revisit" OR abs:"revisiting") AND (abs:LLM OR abs:"language model") — 0
Q34 abs:Hessian AND abs:calibration AND abs:quantization AND (abs:noisy OR abs:noise OR abs:variance OR abs:unstable OR abs:stability) AND (abs:LLM OR abs:"language model")
Q35 (abs:"learned rotation" OR abs:"learnable rotation" OR abs:"rotation learning" OR abs:SpinQuant) AND abs:quantization AND (abs:generaliz OR abs:generalization OR abs:overfit OR abs:"calibration set")
Q36 abs:calibration AND abs:pruning AND (abs:"calibration samples" OR abs:"calibration set") AND (abs:"how many" OR abs:"sample size" OR abs:"number of samples" OR abs:"few-shot") AND (abs:LLM OR abs:"language model")
Q37 abs:quantization AND abs:"scaling law" AND abs:calibration
Q38 (abs:"in-sample" OR abs:"out-of-sample" OR abs:"train-test gap") AND abs:quantization AND abs:calibration  → found 2609.31560
Q39 abs:"Hessian" AND abs:quantization AND (abs:"ill-conditioned" OR abs:"condition number" OR abs:"eigenvalue") AND abs:calibration AND (abs:LLM OR abs:"language model") — 0
Q40 abs:quantization AND abs:LLM AND (abs:"calibration tokens" OR abs:"calibration sequences") AND (abs:correlat OR abs:correlated OR abs:redundan OR abs:redundant OR abs:diversity) — 0
Q41 au:Saab AND (abs:quantization OR abs:OPTQ OR abs:GPTQ) (2024-06+)
Q42 abs:OPTQ AND (abs:generalization OR abs:regularization OR abs:"test point")
Q43 (abs:AQLM OR abs:VPTQ OR abs:QTIP OR abs:"additive quantization" OR abs:"trellis" OR abs:codebook) AND abs:LLM AND abs:calibration AND (abs:overfit OR abs:overfitting OR abs:"held-out" OR abs:generalization) — nothing on VQ calibration overfitting
Q44 abs:quantization AND abs:LLM AND abs:calibration AND (abs:"effective sample size" OR abs:"effective number" OR abs:"sequence-level" OR abs:"token-level") AND abs:Hessian — 0
Q45 abs:"post-training quantization" AND (abs:"concentration" OR abs:"with high probability") AND abs:calibration AND abs:LLM AND (abs:bound OR abs:theorem) — 0
Q46 au:Paglieri AND abs:quantization → 2405.20835

Web searches: "Marchenko-Pastur GPTQ Hessian calibration" (negative); "Ledoit-Wolf shrinkage GPTQ damping" (negative; only notes QEP sets λ = mean diag(H)); "calibration data newer LLMs less sensitive Williams Aletras follow-up" (→ Paglieri via 2608.01078 ref [69]); "SpinQuant calibration overfitting held-out AQLM QTIP" (→ AMXFP4 2–3%); "effective sample size calibration tokens Hessian" (negative; → MaCa); "openreview pfw3saHzGU" (→ 2504.04823 sentence).
