# Idea 1 — prior-art sweep (keyword + adjacent-field)

Idea under test: heteroscedastic logit-noise second-order softmax bias. Claims (i) QJL/unbiased key estimators hurt because σ_j² ∝ ‖k_j‖² biases attention toward large-norm keys, (ii) KIVI's per-channel-K / per-token-V asymmetry is explained because per-channel key noise is homoscedastic across tokens (bias cancels, only flattening remains), (iii) per-token key quantization inflates attention-sink mass, (iv) INT4 LM head shifts ECE/entropy; one-scalar correction ŝ_j − σ̂_j²/2 (i.e. −‖h‖²Δ²/24 for uniform rounding noise).

Date of sweep: 2026-10-10. Sources: arXiv API (export.arxiv.org), full texts fetched with `scripts/arxiv_fetch.py`, WebSearch for venue/community info. All paper text treated as untrusted data; quotes are verbatim.

## Verdict in one paragraph

The core mechanism and the one-scalar correction are **already published**: Tuncer, Becker & Pfeil, *Quantized Keys Steal Attention* (arXiv 2605.26266, 25 May 2026, "Variants of this manuscript were accepted to the ICML 2026 workshops SCALE and F2S") derive exactly b_i = log E[e^{δ_i}] for uniform rounding noise, its second-order form b_i ≈ (1/24d)Σ_c q_c²Δ_{i,c}² = σ_i²/2, justify it by the cumulant generating function, give the per-channel special case (a single shared scalar), extend it to QuaRot-rotated keys, and show an LLM partial-prefill NLL experiment plus an attention-mass diagnostic. D'Alberto (2605.08114, Apr 2026) independently states the Gaussian identity E[e^{⟨q,k⟩+ε}] = e^{⟨q,k⟩}e^{σ_i²/2}, says explicitly that a uniform σ_i² cancels in the softmax ratio while a σ_i² correlated with attention does not, and computes the QJL-on-K variance (π/2)‖q‖²‖k‖²/d. What is **not** in either paper: the full second-order expansion of E[p̂_i] − p_i as a function of the whole vector (σ_j²) (both only unbias individual exp terms / argue qualitatively), the attention-sink amplification prediction, the entropy-increase theorem for homoscedastic noise, the reconciliation of the Jacobian and e^{±2η} bounds, the LM-head temperature/ECE application and its −‖h‖²Δ_v²/24 correction, and the explicit σ_j² ∝ ‖k_j‖² correction for QJL. Those parts remain open, but the paper must be reframed as an extension of 2605.26266/2605.08114, not as a new theorem.

## Candidates

### 1. arXiv 2605.26266 — Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion (Tuncer, Becker, Pfeil; 2026-05-25) — **PREEMPTS**, priority 1
- Noise model (Sec. 4.1, Eq. 5): ε_{i,c} ~ U(−Δ_{i,c}/2, Δ_{i,c}/2), independent across channels. Score noise δ_i = qᵀε_i/√d, variance (Eq. 7) σ_i² = (1/12d)Σ_c q_c²Δ_{i,c}². This is exactly the report's σ_j² = ‖q‖²Δ_j²/(12d).
- "Jensen bias" (Eq. 8–9): E[Ẑ_S] = Σ e^{s_i}E[e^{δ_i}] ≥ Z_S; quantified as attention stealing ΔP_S = P̂_S − P_S from cached (quantized) to current (unquantized) chunk.
- Correction (Eq. 10–12): b_i = log E[e^{δ_i}] = Σ_c log(sinh(α_c)/α_c), α_c = q_cΔ_{i,c}/(2√d); Taylor (Eq. 13): b_i ≈ (1/24d)Σ_c q_c²Δ_{i,c}². Quote: "the Taylor correction is exactly half the score-space noise variance: b_i ≈ σ_i²/2. This follows from the cumulant generating function (CGF) ... For zero-mean noise (κ1 = 0), the leading term is κ2/2 = σ²/2, which depends only on the variance and not on the specific noise distribution." (Eq. 14). This is the report's "1-scalar correction" verbatim.
- Grouped per-token form (Eq. 15): b_i ≈ (1/24d)Σ_j Δ_{i,j}²‖q_j‖². Overhead O(QK·d/g), ~5% latency on MAGI-1 with g=32.
- Per-channel case (Appendix E, Eq. 22–23): "The correction reduces to a single scalar shared by all tokens" and "In our experiments, per-token quantization with the token-dependent correction consistently outperforms per-channel quantization with a shared correction." This is the report's "per-channel → bias cancels" observation, though they do not state the homoscedastic/temperature interpretation.
- QuaRot extension (Appendix F): replace q by Hq.
- Experiments: MAGI-1, SkyReels-V2, HY-WorldPlay at INT2 (Table 1): "reaching near-BF16 video quality"; "2.19 effective bits with our method outperforms 4.38 effective bits without correction". Attention-mass shift ΔP_S measured (Fig. 4): correction "slightly over-corrects into negative values".
- LLM experiment (Sec. 5.4, Appendix O): Llama-3.1-8B, Mistral-7B-Instruct-v0.3, Qwen2.5-32B-Instruct, LongBench-Pro, teacher-forced NLL, INT2 KV; "the Taylor correction reduces NLL relative to plain INT2"; attention-mass diagnostic on Llama-3.2-1B (Fig. A11) shows INT2 increases cached-block mass and the correction restores it. Only figures, no numeric table in the text.
- Stated limitations: "Standard single-token decoding offers less headroom for the correction because many cached tokens compete with only one unquantized current token." and "Quantizers with nonuniform or biased error may likewise require a modified derivation." — precisely the regime (all keys quantized, heteroscedastic per-token Δ_j) where the report's cross-token bias would still act; the paper does not analyse it.
- Not covered: attention sinks (grep "sink" → 0 hits), entropy/temperature (0 hits), LM head (0 hits), QJL (0 hits), heteroscedastic/homoscedastic vocabulary (0 hits), full second-order E[p̂_i] formula.

### 2. arXiv 2605.08114 — Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant (D'Alberto; 2026-04-27) — **PARTIALLY PREEMPTS**, priority 1
- Sec. 3.1, Eq. 4: "E[e^{⟨q,k_i⟩+ε_i}] = e^{⟨q,k_i⟩}·e^{σ_i²/2}, for ε_i ~ N(0,σ_i²). If σ_i² is uniform across keys the factor e^{σ²/2} cancels in the softmax ratio. The dangerous regime is when σ_i² is correlated with the attention score: high-attention keys with large quantisation variance produce asymmetric distortion that cannot cancel." — the homoscedastic-cancellation / heteroscedastic-bias dichotomy, stated qualitatively.
- Sec. 4.2, Eq. 12–13 + Observation 1: Var[δ]_{QJL-K} = (π/2)‖q‖²‖k‖²/d; ratio to scalar quantisation (π/2)/ε_B ≥ π/2. Sec. 3.3/4.3: K-vs-V asymmetry argued (V enters linearly, errors average at 1/√S; K enters through exp).
- No correction, no second-order expansion of the softmax vector, no per-channel/per-token rule, no sinks. "A sufficient condition when the Jensen mechanism amplifies superlinearly" is given (Sec. 3.4) as a KL argument. Comment: "the process is highly assisted by AI" — treat as a note, not peer-reviewed.

### 3. arXiv 2608.14191 — KV Cache Compression Through the Lens of Transform Coding (AATC; 2026-08-14) — **PARTIALLY PREEMPTS** (machinery), priority 2
- Assumption 1 (white-noise quantization: zero-mean, independent across tokens/channels, symmetric, odd moments vanish) and Assumption 2 (bounded key error). Theorem 1: E‖o − ô‖² = D_K + D_V + R with D_K, D_V factoring across tokens and channels; proof expands softmax to first and second order (δa_i^{(1)} linear, δa_i^{(2)} quadratic in α_i = qᵀδk_i/√d), E[α_i²] = Σ_c q_c²E[δk_{ic}²] ≤ ‖q‖²σ_K, remainder bounded in Lemma 1.
- This is the same second-order Taylor-of-softmax under the same noise model, but applied to the **second moment** (MSE of attention output), not to the **bias** E[â_i] − a_i. The bias term is implicitly inside their δa_i^{(2)} and could be extracted; a reviewer will ask why the report's Theorem 1 is more than a corollary.

### 4. arXiv 2609.31291 — Softmax Reparameterization for Output-Head Quantization (Kadav; 2026-09-25, "ICLR 2027 submission") — **PARTIALLY PREEMPTS** (LM-head part), priority 2
- Eq. 6: D_KL(p‖softmax(Wh+e)) = ½eᵀF(p)e + O(‖e‖³), F(p) = diag(p) − ppᵀ, eᵀF(p)e = Σ_v p_v(e_v − ē_p)² — the same second-order object as the report's LM-head analysis, used as a diagnostic ("Fisher/2").
- Method: subtract a scalar multiple of the vocabulary-row mean from each output row before quantization (functionally equivalent head), coefficient chosen by validation KL. W4 results: "test KL falls by 93% on XGLM under RTN and by 73–77% on Phi, BLOOM, and BLOOMZ under activation-weighted MSE".
- No noise-variance bias/temperature derivation and no per-row −‖h‖²Δ_v²/24 correction; but it occupies the "cheap, training-free LM-head fix evaluated by KL" niche and must be a baseline.

### 5. arXiv 2508.04257 — KVSink (Su et al.; COLM 2025) — **SUPPORTS**, priority 2
- Table 1 (LLaMA2-7B, RTN, g=16, MSE×100): per-channel dynamic key quantization, groups containing sink tokens have 16.3–29.2% higher error; static per-token quantization excluding sinks reduces key error by up to 81.1% and value error by 68.2%. Mechanism "QKV suppression" (sinks have small Q/K/V norms but high QK cosine).
- Measures MSE, not attention-mass bias; does not predict direction of sink-mass change. Provides the baseline sink-protection method (keep first tokens at full precision) against which the report's prediction (per-token key quantization *inflates* sink mass) can be tested. Note KVSink says sinks have *small* key norms, which cuts against "sink tokens have large Δ_j" — the report's sink claim needs checking on real Δ_j statistics (massive-activation tokens vs BOS sinks are not the same thing).

### 6. arXiv 2605.19660 — OScaR (Su et al.; 2026-05-19, under review) — **CONTRADICTS** (framing), priority 2
- "While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression." Identifies Token Norm Imbalance (TNI); Appendix G Eq. 11: per-channel block reconstruction error "is fundamentally governed by the range of token norms within the block"; Table 2 shows TNI does not amplify error under per-token quantization.
- Also records (via WebSearch snippet of the paper) that TurboQuant+ omits QJL "as it has been shown to increase variance that is subsequently amplified by the softmax function". Supports the QJL part, contradicts an unconditional "keys must be per-channel": the honest framing is a bias (per-token) vs variance/MSE (per-channel under TNI) trade-off.

### 7. arXiv 2606.13233 — ReSET: NVFP4 reasoning via step-aware temperature scaling (2026-06-11) — **CONTRADICTS** (sign of entropy shift), priority 2
- "quantization increases incorrect sampling at low-entropy symbolic tokens, while causing over-concentration on a small set of tokens in high-uncertainty reasoning steps"; "NVFP4 compresses the entropy of high-entropy tokens in uncertain steps ... so a temperature above the default is needed". Mixed-sign entropy shift; the report's homoscedastic prediction (entropy rises) only covers one direction. NVFP4 here is whole-model, not LM-head-only.

### 8. arXiv 2607.09683 — Ablation, Statistical Inference, and Validation for KV-Cache Compression (D'Alberto; 2026-06-14) — **SUPPORTS**, priority 3
- "The softmax is convex in its argument: E[exp(s + ε)] = exp(s)·exp(σ²/2)"; "K-path gap (KL, relMSE K) is larger for TQ-QKV, confirming Jensen amplification" with permutation tests. Same author, same qualitative argument; empirical support for "QJL on K hurts".

### 9. arXiv 2606.08635 — SpectrumKV (2026-06-07) — **SUPPORTS**, priority 3
- Lemma 2 (softmax amplification bound): for ‖η‖∞ ≤ η and S the set of near-max logits, |p̃_j − p_j|/p_j ≤ e^{2η} − 1 + (|S|e^{2η} − |S|)/(1 + |S|e^{−2η}); "scales roughly as e^{2η}/|S|". Conjecture 1: entropy-dependent INT4 tolerance threshold H* ≈ log n − 2. The proof is a sketch ("The tightest bound arises when the partition is sharp"). This is the "sharp attention is more sensitive" worst-case bound the report wants to reconcile with the Jacobian bound.

### 10. arXiv 2608.20988 — Jacobian-guided Noise Injection (2026-08-21) — **SUPPORTS**, priority 3
- First-order: ‖Δa‖ bounded through ‖J_S(z)‖₂, J_S = diag(p) − ppᵀ. Eq. 10: E_{ε~N(0,σ²I)}[L(z+ε)] ≈ L(z) + (σ²/2)tr∇²L — a second-order-in-expectation argument, but for the *loss* under *isotropic* noise (training-time regularizer), not for softmax-output bias under heteroscedastic noise. This is the "flat attention is more sensitive" (Jacobian) intuition.

### 11. arXiv 2608.12026 — SoftWater: Class-Aware Rate Allocation for Softmax Quantization (2026-08-12) — **SUPPORTS**, priority 3
- Second-order KL Hessian for the head: E[(diag(p) − ppᵀ) ⊗ XXᵀ]; separable surrogate λ̄_k = E[p_k(1−p_k)]; allocates "fine grids to frequent, low-variance classes and coarse grids to rare ones". Rate allocation, not bias correction; its per-class curvature weighting is the natural comparator for the report's rare-token (large Δ_v) bias claim.

### 12. arXiv 2609.07901 — Quantization Amplifies Determinism, Not Bias (2026-09-07) — **CONTRADICTS** (naive temperature story), priority 3
- Qwen3-8B W4A16 AWQ vs bf16, ~71k paired completions: same-brand agreement +5.1pp (Holm p = .023), TTR −0.011 (std. effect −0.51) — int4 makes outputs *less* diverse at 8B. AWQ keeps the LM head in FP16, so not a direct LM-head test, but it is evidence that body quantization sharpens rather than flattens.

### 13. arXiv 2604.08976 — Quantisation Reshapes the Metacognitive Geometry of Language Models (2026-04-10) — **TANGENTIAL** (citation needs correction), priority 3
- The report cites this as "GPTQ 4-bit worsens ECE". The abstract is Llama-3-8B-Instruct at Q5_K_M (llama.cpp) vs f16, metacognitive M-ratio (Spearman ρ = 0.00 across domains) with Type-2 AUROC stable (ρ = 1.00). Not GPTQ, not 4-bit, not ECE. Fix the citation or drop it.

### 14. arXiv 2309.01729 — Softmax Bias Correction for Quantized Generative Models (Pandey, Fournarakis, Patel, Nagel; 2023-09-04) — **TANGENTIAL**, priority 4
- Bias is on the *output* side: 8-bit softmax probabilities round to zero ("up to 99% of the values are rounded to zero"; sums as low as 0.3); correction is an offline offset absorbed into the zero-point. Not logit-noise Jensen bias. Cited by 2605.26266 as prior "attention sensitivity and correction" work.

### 15. arXiv 2605.20868 — Runtime-Certified Bounded-Error Quantized Attention (2026-05-20) — **TANGENTIAL**, priority 4
- Deterministic per-head per-step bounds on attention-distribution distortion from INT8 keys (Theorem 4.3) and value error (Theorem 4.1), with FP16 fallback. Worst-case, not expectation; no bias term.

Also looked at and set aside: 2606.31519 RaBitQCache (unbiased cosine estimator used only for top-p selection; Theorem A.1), 2605.01910 SANTA (unbiased post-softmax value sampling, Prop. A.3 variance 1/S), 2604.19528 (RaBitQ vs TurboQuant reproducibility), 2510.05373 KVLinC (Observation 3: raw keys per-channel + Hadamard values per-token is best; no explanation offered), 2302.00787 FAVOR# / 2009.14794 Performers (random-feature variance, different object), 2411.04013 kNN Attention Demystified (sampling estimators). Econometrics analog: Bhat (1995) heteroscedastic extreme-value (HEV) logit — heteroscedastic Gumbel utilities break IIA; this concerns P(argmax), not E[softmax], and is not on arXiv; cite as motivation only.

## Community/engineering evidence
- vLLM TurboQuant docs (all versions v0.20–latest): "QJL is intentionally omitted — community consensus (5+ independent groups) found it hurts attention quality by amplifying variance through softmax." The "5+ groups" are not linked.
- github.com/TheTom/turboquant_plus: "QJL hurts, PolarQuant works"; "K precision is the dominant quality factor because it controls attention routing via softmax."

## What remains defensible for Idea 1 after this sweep
1. The full second-order vector formula E[p̂_i] − p_i ≈ (p_i/2)[σ_i²(1−2p_i) − Σ_j p_jσ_j²(1−2p_j)] and its homoscedastic specialisation σ²p_i(‖p‖² − p_i); neither 2605.26266 nor 2605.08114 writes it (2608.14191 has the ingredients but only for the second moment).
2. The entropy-increase statement for homoscedastic noise (Chebyshev-sum argument) — not found anywhere.
3. The attention-sink prediction under per-token key quantization and its measurement — not found; but KVSink's "sinks have small K norm" needs to be squared with "sinks have large Δ_j".
4. The QJL-specific correction σ̂_j² = (π/2)‖q‖²‖k_j‖²/m subtracted at zero cost — D'Alberto computes the variance but does not propose the correction; 2605.26266 says FP/non-uniform quantizers "require a modified derivation".
5. The LM-head −‖h‖²Δ_v²/24 correction and ECE/entropy measurement — not found; 2609.31291 and 2608.12026 are the mandatory baselines, and 2604.08976 must not be cited as ECE evidence.
6. The reconciliation of Jacobian (2608.20988) and e^{2η} (2606.08635) bounds via the second-order term — not found.
The report's "novelty 높음 / 선행 논문 사실상 없음" judgement is wrong for the core correction; the honest position is "extension of Tuncer et al. 2026 to the all-cached, heteroscedastic, QJL and LM-head regimes with a full second-order theorem".

## Queries run (arXiv API unless noted)
Seed/direct (2023-01-01 to 2026-10-31, relevance):
1. `abs:softmax AND abs:quantization AND (abs:bias OR abs:Jensen)`
2. `abs:"attention sink" AND (abs:quantization OR abs:quantized) AND abs:key`
3. `abs:"unbiased estimator" AND abs:attention AND (abs:quantization OR abs:quantized OR abs:sketch)`
4. `abs:heteroscedastic AND abs:softmax`
5. `abs:"per-channel" AND abs:"per-token" AND abs:"KV cache" AND (abs:theory OR abs:theoretical OR abs:analysis)`
6. `abs:"lm head" AND (abs:quantization OR abs:quantized) AND (abs:temperature OR abs:calibration OR abs:entropy)`
7. `abs:logit AND abs:noise AND abs:softmax AND abs:temperature`
8. `abs:"bias correction" AND abs:softmax AND (abs:quantization OR abs:quantized OR abs:compression)`
9. `abs:"KV cache" AND (abs:quantization OR abs:quantized) AND abs:softmax AND (abs:variance OR abs:bias)`
10. `abs:"Johnson-Lindenstrauss" AND abs:"KV cache" AND (abs:quantization OR abs:quantized)`
Date-sorted, --from 20250601:
11. `abs:softmax AND (abs:quantization OR abs:quantized) AND (abs:bias OR abs:Jensen OR abs:unbiased)`
12. `abs:"attention sink" AND (abs:quantization OR abs:quantized OR abs:"low-bit")`
13. `abs:"KV cache" AND (abs:quantization OR abs:quantized) AND (abs:theory OR abs:theoretical OR abs:"error bound" OR abs:analysis) AND abs:softmax`
14. `(abs:"lm head" OR abs:"output layer" OR abs:"unembedding") AND (abs:quantization OR abs:quantized) AND (abs:calibration OR abs:entropy OR abs:temperature OR abs:"ECE")`
15. `abs:QJL OR abs:TurboQuant OR abs:PolarQuant OR (abs:"Johnson-Lindenstrauss" AND abs:"KV cache")`
Adjacent-field vocabulary:
16. `abs:"Jensen gap" AND (abs:softmax OR abs:"log-sum-exp" OR abs:logsumexp)` (2018–) — 0 hits
17. `abs:softmax AND abs:noise AND (abs:"noisy logits" OR abs:"logit noise" OR abs:"perturbed logits") AND (abs:expectation OR abs:expected OR abs:bias OR abs:temperature)` — 0 hits
18. `(abs:dither OR abs:dithering OR abs:"noise shaping" OR abs:"sigma-delta") AND abs:attention AND (abs:quantization OR abs:quantized)`
19. `(abs:"random features" OR abs:"random feature" OR abs:sketch OR abs:sketching) AND abs:softmax AND abs:attention AND abs:variance AND (abs:unbiased OR abs:bias)`
20. `(abs:"extreme value" OR abs:Gumbel) AND abs:attention AND (abs:noise OR abs:quantization) AND (abs:softmax OR abs:argmax)`
21. `abs:heteroscedastic AND (abs:logits OR abs:logit) AND (abs:classification OR abs:calibration OR abs:softmax)`
22. `(abs:lognormal OR abs:"log-normal" OR abs:"cumulant generating") AND abs:softmax AND (abs:attention OR abs:noise)`
23. `abs:"importance sampling" AND abs:attention AND abs:softmax AND (abs:variance OR abs:bias) AND (abs:quantization OR abs:approximation OR abs:estimator)`
24. `(cat:cs.IT OR cat:eess.SP OR cat:stat.ML OR cat:math.OC OR cat:math.ST) AND abs:softmax AND (abs:quantization OR abs:quantized OR abs:"additive noise") AND (abs:bias OR abs:expectation OR abs:variance)`
25. `abs:"attention entropy" AND (abs:quantization OR abs:quantized OR abs:"low-bit" OR abs:"KV cache")`
26. `abs:"temperature" AND abs:"KV cache" AND (abs:quantization OR abs:quantized) AND (abs:softmax OR abs:attention)` — 0 hits
27. `(abs:"stochastic rounding" OR abs:"unbiased quantization" OR abs:"unbiased quantizer") AND abs:attention AND (abs:softmax OR abs:"KV cache")` — 0 hits
28. `(abs:"expected calibration error" OR abs:ECE OR abs:miscalibration OR abs:overconfiden*) AND (abs:quantization OR abs:quantized) AND (abs:LLM OR abs:"language model")`
29. `abs:"massive activation" AND (abs:quantization OR abs:quantized) AND abs:attention AND (abs:key OR abs:"KV")`
30. `(abs:"first-order" OR abs:"second-order") AND abs:softmax AND abs:perturbation AND abs:attention AND (abs:quantization OR abs:error OR abs:noise)` — 0 hits
31. `abs:softmax AND abs:"additive noise" AND abs:attention` — 0 hits
32. `(abs:"token entropy" OR abs:"next-token entropy" OR abs:"output entropy") AND (abs:quantization OR abs:quantized OR abs:NVFP4 OR abs:FP4)`
33. `abs:key AND abs:"per-channel" AND (abs:quantization OR abs:quantized) AND abs:"KV cache" AND (abs:explain OR abs:why OR abs:theoretical OR abs:optimal)`
34. `abs:attention AND (abs:logits OR abs:scores) AND abs:noise AND (abs:flatten OR abs:flattening OR abs:sharpen OR abs:sharpening OR abs:"effective temperature")`
35. `(abs:convexity OR abs:convex) AND abs:exponential AND abs:attention AND (abs:quantization OR abs:quantized OR abs:compression) AND (abs:inflate OR abs:inflation OR abs:bias)`
36. `abs:"attention" AND abs:"sketch" AND abs:"KV cache" AND (abs:variance OR abs:unbiased) AND abs:"inner product"` — 0 hits
37. `ti:softmax AND (ti:quantization OR ti:quantized OR ti:"low-bit" OR ti:"low-precision")`
38. `(abs:heteroscedastic OR abs:heteroskedastic) AND (abs:logit OR abs:"random utility" OR abs:"discrete choice" OR abs:"Gumbel")` (2015–)
39. `(abs:"noisy softmax" OR abs:"softmax with noise" OR abs:"expected softmax" OR abs:"expectation of the softmax" OR abs:"expectation of softmax")` (2015–)
40. `abs:"output head" AND (abs:quantization OR abs:quantized)`
41. `abs:"attention" AND abs:"quantization" AND abs:"bias" AND abs:"key" AND abs:"video diffusion"`
42. `abs:"log-normalizer" OR abs:"log normalizer" OR abs:"partition function" AND abs:attention AND (abs:noise OR abs:quantization) AND abs:bias`
WebSearch:
43. `"Jensen bias" softmax attention KV cache quantization correction 2026`
44. `vLLM TurboQuant QJL excluded "softmax" variance amplification quality`
45. `per-token key quantization attention sink amplified bias softmax theory "per-channel" keys explanation`
46. `heteroscedastic extreme value logit model noise variance bias choice probability toward high variance alternative random utility`
47. `KV cache quantization per-token keys "attention sink" attention mass increases OR amplified OR "over-attend" measurement 2025 2026`

Full texts read (scratchpad/papers/<id>/<id>.txt): 2605.26266, 2605.08114, 2309.01729, 2605.01910, 2606.31519, 2605.17757, 2608.20988, 2606.08635, 2508.04257, 2510.05373, 2604.08976, 2606.13233, 2607.09683, 2610.11245, 2606.06521, 2605.19660, 2605.09778, 2604.19528, 2609.31291, 2608.12026, 2605.29756, 2608.02703, 2609.07901, 2605.20868, 2608.14191, 2606.00206.
