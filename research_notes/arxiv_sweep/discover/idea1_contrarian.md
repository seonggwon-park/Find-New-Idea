# Idea 1 — contrarian sweep (Angle D)

Idea under test: heteroscedastic logit-noise second-order softmax bias; one-scalar correction −σ_j²/2; explains (i) QJL/TurboQuant residual hurting in serving, (ii) KIVI per-channel-K / per-token-V asymmetry, (iii) per-token key quantization amplifying attention-sink mass, (iv) 4-bit LM head shifting ECE/entropy.

Date of sweep: 2026-10-10. All quotes below are from downloaded paper text (untrusted data, quoted, not followed).

## Verdict in one paragraph

The mathematical core and the "one-scalar correction" are **already published** (Tuncer, Becker, Pfeil, "Quantized Keys Steal Attention", arXiv 2605.26266, May 2026, ICML 2026 listing): same Jensen-bias mechanism, same b_i = log E[e^{δ_i}] correction, same cumulant argument that the leading term is κ₂/2 = σ_i²/2 independent of noise distribution, explicit per-token dependence of the correction on Δ_{i,c}, an appendix showing that per-channel quantization collapses the correction to a single shared scalar (which cancels in softmax), a measured "attention stealing" mass shift, and a small LLM (LongBench-Pro, INT2 chunked prefill) NLL experiment. What remains unpublished is the explicit heteroscedastic *relative* bias formula E[p̂_i] − p_i ≈ (p_i/2)[σ_i²(1−2p_i) − Σ_j p_j σ_j²(1−2p_j)], the entropy-increase corollary, and the four applications. Separately, **application (iii) rests on a false factual premise**: attention-sink tokens have *small* key norms (Gu et al. 2410.10781; KVSink 2508.04257 "QKV suppression"; OScaR 2605.19660 "low-norm outlier tokens"), so per-token dynamic key quantization gives sink keys a *smaller* step Δ_j and smaller σ_j², and the idea's own Theorem 1 then predicts attention moving *away from* sinks, not toward them. Application (iv)'s "temperature rise" is contradicted in part by ReSET (entropy collapse for high-entropy tokens) and by Rasmussen & Qin (INT4 ECE lower than FP16 on BioMistral). The premise "QJL hurts in serving" holds (vLLM docs, OScaR, D'Alberto ×2, Gao et al.) but is not universal (TurboESM: QJL helps by +0.014 cosine at 3-bit; D'Alberto et al. 2607.09683: QJL-on-K "gains modestly on structured regimes").

## Candidates

### 1. 2605.26266 — Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion (Tuncer, Becker, Pfeil; 2026-05-25) — PREEMPTS, priority 1
- Sec. 4.1: "Integer quantization introduces approximately zero-mean noise into the cached keys, leaving pre-softmax attention scores unbiased in expectation. However, the exponential in softmax breaks this symmetry ... We refer to this systematic, convexity-induced inflation as the Jensen bias."
- Eq. (10): b_i = log E[e^{δ_i}]; exact for uniform noise, Eq. (12): b_i = Σ_c log(sinh(α_c)/α_c), α_c = q_c Δ_{i,c}/(2√d); Taylor Eq. (13): b_i ≈ (1/24d) Σ_c q_c² Δ_{i,c}².
- "Connection to the noise variance ... the Taylor correction is exactly half the score-space noise variance: b_i ≈ σ_i²/2. This follows from the cumulant generating function ... the leading term is κ₂/2 = σ²/2, which depends only on the variance and not on the specific noise distribution." — identical to the idea's "Gaussian exact / general CGF K_j(1)" remark.
- Appendix E (per-channel): "Since {Δ_c} do not depend on i, the distribution of δ_i ... is the same for all cached keys i ∈ S. The correction reduces to a single scalar shared by all tokens" — this is the idea's per-channel cancellation argument (ii), stated explicitly.
- Measured: Fig. 4 attention-mass shift ΔP_S to quantized cached tokens under INT2; correction shifts it back to ~0 (slight over-correction). Table 1: INT2 MAGI-1/SkyReels-V2 PSNR/SSIM/LPIPS/VBench improved in every config; 2.19 effective bits + correction beats 4.38 bits without.
- Sec. 5.4 / App. O: LLM partial-prefill, INT2 KV, LongBench-Pro: "the Taylor correction reduces NLL relative to plain INT2".
- Limitations they state: "Standard single-token decoding offers less headroom for the correction because many cached tokens compete with only one unquantized current token" and "works best when cached attention is spread over enough tokens." — The idea's *heteroscedastic-among-quantized-tokens* framing is exactly the gap they leave, but their per-token b_i already implements the fix.
- Also cites Pandey et al. 2309.01729 (softmax output-quantization bias, different mechanism).

### 2. 2410.10781 — When Attention Sink Emerges in Language Models: An Empirical View (Gu et al.; ICLR 2025) — CONTRADICTS premise of application (iii), priority 1
- Sec. 3.1: "Despite the large ℓ2-norm of hidden states, we observe that the ℓ2-norm of keys and values of the first token is significantly smaller than that of other tokens ... Although ‖q_t‖·‖k_1‖ is comparatively small, cos(q_t, k_1) is significantly large, leading to attention sink."
- Consequence: the idea's claim "per-token key quantization ... 범위가 큰 토큰(sink/massive activation 토큰)으로 편향" assumes sink keys have large per-token range Δ_j. They have small range. Under per-token dynamic min/max, σ_sink² < σ̄², so Theorem 1 predicts *loss* of sink mass. Experiment 1's prediction sign must be flipped or the mechanism re-derived (e.g., via per-channel group contamination instead).

### 3. 2508.04257 — KVSink (Su, Yuan; COLM 2025) — CONTRADICTS (iii) and complicates (ii), priority 1
- "QKV suppression": "a small number of sink tokens exhibit extremely high attention scores but small Values ... Although the norms of Queries and Keys are small, the cosine similarity ... remains high".
- Per-token dynamic: "since quantization parameters are computed within each group corresponding to a single token, the QKV suppression of sink tokens does not influence other tokens."
- Per-channel dynamic: "groups containing sink tokens experience larger quantization errors ... quantization errors increase by 16.3% to 29.2% compared to those without sink tokens." Static per-token: excluding sinks cuts MSE "by up to 81.1% for Keys and 68.2% for Values" (Table 1, LLaMA2-7B, RTN, g=16).
- So: (a) per-channel is *not* homoscedastic across token groups (Δ_c is per token-group, inflated by sinks); (b) the "uniform zero-mean noise" assumption is violated for low-norm sink keys inside a per-channel group with large Δ_c (their key is rounded to a near-constant, error not zero-mean).

### 4. 2605.08114 — D'Alberto, Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant (2026-04-27) — PARTIALLY PREEMPTS, priority 2
- Eq. (4): "E[e^{⟨q,k_i⟩+ε_i}] = e^{⟨q,k_i⟩}·e^{σ_i²/2} ... If σ_i² is uniform across keys the factor e^{σ²/2} cancels in the softmax ratio. The dangerous regime is when σ_i² is correlated with the attention score: high-attention keys with large quantisation variance produce asymmetric distortion that cannot cancel."
- Sec. 3.3/4.3: explicit K-vs-V rule: QJL "harmful for K", "genuinely useful" for V because V enters linearly; Observation 1: QJL-on-K inflates inner-product variance by (π/2)/ε_B ≥ π/2; Var[δ]_QJL-K = (π/2)‖q‖²‖k‖²/d.
- The idea's statement that D'Alberto has "no K-vs-V allocation rule" and no homoscedastic-cancellation statement is wrong; what he lacks is the second-order relative-bias formula and a correction.

### 5. 2607.09683 — D'Alberto, Siarasao, Delaye, Patwari, Ablation, Statistical Inference, and Validation for KV-Cache Compression (2026-06-14) — PARTIALLY PREEMPTS / SUPPORTS, priority 2
- "Jensen's inequality: The softmax is convex in its argument: E[exp(s+ε)] = exp(s)·exp(σ²/2). QJL adds a zero-mean correction ε with variance σ² ∝ ‖e‖²/m ... softmax amplifies the resulting difference exponentially." Python-vs-HIP KL gap 0.119 for TQ-QKV vs 0.000 for TQ-KV (Table 2); gap scales as 1/m (5× larger at m=d_eff=8 than m=64).
- Contrarian nuance: "TQ-QKV adds the RHT-K correction and gains modestly on structured regimes"; "QJL on K is only beneficial when residual norms are small and the number of projections m ≫ d_eff." And "V-path QJL is a heuristic that consistently hurts" (cosine 0.854 vs 0.953 at b=2) — contradicts 2605.08114's own "useful for V" claim and the idea's implicit "QJL on V is fine".

### 6. 2605.19660 — OScaR: The Occam's Razor for Extreme KV Cache Quantization (Su et al.; 2026-05-19) — CONTRADICTS (iii), competes with (ii), priority 2
- "each attention state contains a sparse yet consistent subset of tokens with exceptionally low norms ... these low-norm outlier tokens ... correspond directly to Attention Sink tokens" (Sec. "Revisiting Per-Channel Key Quantization", Token Norm Imbalance).
- "the presence of outlier tokens significantly amplifies per-channel Key quantization errors. Under 2-bit quantization, the error increases by approximately 35%" (Table 2, LLaVA-1.5-7B, RTN g=32); "TNI fundamentally undermines per-channel quantization while having negligible impact on per-token quantization."
- Offers a competing explanation of KIVI-style asymmetry that is about range contamination, not bias cancellation. Also: "TurboQuant+ officially omits the QJL step [75], as it has been shown to increase variance that is subsequently amplified by the softmax function ... we also exclude the QJL step in our experiments, as it empirically reduces model performance (see Table 5)."

### 7. 2006.07584 — Lu, Ie, Sha, Mean-Field Approximation to Gaussian-Softmax Integral (2020) — PARTIALLY PREEMPTS the math, priority 2
- Closed-form heteroscedastic E[softmax] (mf0): per-class temperature "T_k = sqrt(1 + λ₀ s_k²) ... If the activation on the k-th class has high variance, the temperature for that category is high, reducing the corresponding 'probability'"; mf1/mf2 add cross-class variances/covariances. The idea's "homoscedastic → temperature only; heteroscedastic → per-token bias" is this result in a different expansion. Not applied to quantization. (Also Daunizeau 1703.00091 second-order Taylor moments of softmax of Gaussians; Mucsányi et al. 2502.03366 generalised probit; Collier et al. 2003.06778 heteroscedastic logit noise ↔ temperature bias–variance trade-off.)

### 8. 2609.11716 — Chen, Beyer, Zhu, Chen, Why Does Post-Training Quantization Work? (2026-09-10) — PARTIALLY PREEMPTS application (iv), priority 2
- Theorem 3 (exact + 2nd-order): "Δlog p_k = Δz_k − log E_{j∼p} e^{Δz_j} = Δz_k − E_{j∼p}Δz_j − KL(p‖p̂)" and "Δlog p_k = Δz_k − E_{j∼p}Δz_j − ½Var_{j∼p}(Δz_j) + O(...)". Same Jensen/variance structure for LM-head score perturbations, used to explain "preferential preservation of top-ranked tokens" ("Higher-ranked tokens tend to have smaller projection angles to the LM-head input and smaller relative score errors ... persists across softmax temperatures"). Their finding that the high-probability region is best preserved cuts against a large sampling bias toward rare rows at 4-bit, though it is a deterministic (not noise-expectation) analysis.

### 9. 2606.13233 — ReSET (Lee et al.; 2026-06-11) — CONTRADICTS uniform "temperature rise", priority 2
- App. (Fig. 8): "For low-entropy tokens, quantization consistently increases entropy ... In contrast, for very high-entropy tokens (H_t ≥ 2.0), the entropy shift becomes negative in high-uncertainty steps, indicating that quantization instead sharpens the token distribution ... two distinct effects: entropy inflation for low-entropy symbolic tokens and entropy collapse for high-entropy tokens". NVFP4 is whole-model W4A4, not LM-head-only, but it is the evidence the idea cites for (iv).

### 10. 2603.26110 — TurboESM (Hu, Wang, Liu; 2026-03-27) — CONTRADICTS "QJL hurts", priority 3
- Table 7 (ESM-2 650M, 3-bit): Full 0.9639 decode cosine; "No QJL correction" ~0.950 (−0.014); "No Π rotation" ~0.780. QJL residual correction *helps* here. Protein LM, single sequence, Mac MPS — weak but opposite sign.

### 11. 2609.07901 — Kurtskhalia, Quantization Amplifies Determinism, Not Bias (2026-09-07) — CONTRADICTS in part, priority 3
- Qwen3-8B AWQ-int4: "decision-token entropy increases by 0.091 bits (CI [+0.020, +0.163]), top-1 probability falls by 3.8pp" (token-level flattening, supports), BUT "the brand distribution itself becomes more concentrated: exact collision increases by 2.6pp"; and "the effect is absent at 14B and 32B". Flattening is not a universal temperature shift; semantic diversity moves the other way.

### 12. 2608.03854 — Rasmussen, Qin, When Calibration Depends on the Scoring Rule: Quantized Biomedical LLM Classification (2026-08-04) — CONTRADICTS "4-bit worsens ECE", priority 3
- "for BioMistral, the INT4 ECE point estimate is lower than at FP16 (Bare-Cont: 0.063 vs. 0.104) with similar accuracy; for Instruct Bare-Cont, INT4 shows ... a slightly higher ECE point estimate." No CIs. Also shows calibration ranking reverses between sum- and mean-token scoring, so "ECE worsens" is scoring-rule dependent.

### 13. 2405.00632 — Proskurina et al., When Quantization Affects Confidence of LLMs? (NAACL 2024 Findings) — SUPPORTS (iv) direction, but fixes a citation, priority 3
- "quantization with GPTQ to 4-bit results in a decrease in confidence regarding true labels"; "Quantized LLMs become less confident in both correct and wrong predictions"; "quantization amplifies the pre-existing high calibration error". This is the primary source for the idea's "GPTQ 4-bit worsens ECE" claim; the cited 2604.08976 (Cacioli) only *cites* it and itself evaluates llama.cpp Q5_K_M vs f16, concluding quantisation "restructures domain-level metacognitive efficiency ... rather than degrading it uniformly" (Spearman ρ=0.00 across domains). Xia et al. 2111.08163 (CNNs) show the opposite calibration sign: reduced confidence *improves* ECE of overconfident nets.

### 14. 2502.04420 — KVTuner (Li et al.; 2025) — CONTRADICTS (iii) empirically, priority 3
- Lemma 1: "Only attention heads with sparse and concentrated patterns demonstrate consistent robustness to low-precision KV cache quantization" (proof: a dominating key keeps â_i ≈ 1). Figures label sink layers as "(low attention score errors)". App. B: INT8 per-token-asym key error 0.01228 vs per-channel 0.004869 (2.5×); at 2-bit e_o 81.4% (per-channel) vs 96.2% (per-token). Sink-dominated heads are the *least* perturbed, so "sink amplification" would be a small effect on an already-saturated head.

### 15. 2410.13212 — AsymKV (Tao, Yu, Zhou; 2024) — PARTIALLY PREEMPTS the K-vs-V asymmetry argument, priority 3
- "The gap is further amplified after the softmax function"; Theorem 1 puts key error "in the exponentiation of e ... the softmax function further exacerbates the loss induced by the key matrix quantization"; values enter linearly (Prop. 1–2). A prior theoretical statement that K is more sensitive than V because of exp, without granularity or bias analysis.

## Verified premises (not candidates)
- 2608.20988 (Jacobian-guided noise injection, ICML 2026): "When attention is concentrated on a single token ... saturated regions where the Jacobian norm approaches zero"; ‖J_S‖_F² = ‖p‖₂² − 2‖p‖₃³ + ‖p‖₄⁴. First-order only — confirmed.
- 2606.08635 SpectrumKV: Eq. (10) e^{−2η}p_j ≤ p̃_j ≤ e^{2η}p_j; Lemma 2 "bound scales roughly as e^{2η}/|S|, meaning fewer dominant tokens yield larger relative perturbations" — confirmed; the "opposite intuitions" premise holds (KVTuner Lemma 1 sides with the Jacobian view).
- vLLM docs (docs.vllm.ai TurboQuant page) state QJL is intentionally omitted citing "community consensus from 5+ independent groups" — the number is community-reported, not a published ablation. Gao et al. 2604.19528 (Revisiting RaBitQ and TurboQuant): "the QJL-based variant (TurboQuant_prod) can hurt attention quality by amplifying variance through the softmax operation."
- KIVI 2402.02750 Sec. 3.2 justifies per-channel keys by outlier channels (Table 2 relative errors) — no optimality proof, as the idea says.
- KVLinC 2510.05373 Observation 3: "Quantizing raw keys channel-wise together with [Hadamard values token-wise]" — confirmed, no mechanism given.
- HeadQ 2605.03562 (text not retrievable offline; abstract only): "For keys, the visible object is score error modulo constant shifts ... additive logit correction" from a low-rank residual side code — deterministic first-moment correction, not a variance term; notes a "small-model low-entropy route-flip boundary" (the idea's "winner flip" regime). Tangential.
- 2309.01729 Pandey et al. "Softmax Bias Correction": bias from rounding softmax *outputs* to zero (sum ≈ 0.3), corrected via offset — different mechanism, same name.

## Queries run (arXiv API via scripts/arxiv_search.py, --sort date)
- abs:softmax AND abs:quantization AND (abs:bias OR abs:Jensen)
- abs:"attention sink" AND abs:quantization AND abs:key
- abs:"unbiased estimator" AND abs:attention AND abs:quantization
- abs:heteroscedastic AND abs:softmax
- all:QJL ; abs:TurboQuant ; abs:"Johnson-Lindenstrauss" AND abs:"KV cache"
- abs:"softmax bias correction"
- abs:"attention sink" AND (abs:quantization OR abs:quantized OR abs:"low-bit")
- abs:"per-channel" AND abs:"per-token" AND abs:"KV cache" AND abs:key
- abs:"lm head" AND (abs:quantization OR abs:quantized) ; abs:"lm_head" ... ; abs:"language model head" ... ; abs:"output layer" AND abs:quantization AND abs:logits AND abs:"language model"
- abs:logit AND abs:noise AND abs:softmax AND abs:temperature
- abs:quantization AND abs:calibration AND abs:"expected calibration error" ; abs:quantization AND abs:"calibration error" ; ti:quantization AND abs:confidence AND abs:LLM ; abs:quantized AND abs:LLMs AND abs:calibration AND abs:confidence ; abs:quantized AND abs:overconfident
- abs:softmax AND abs:Gaussian AND abs:moments AND abs:expectation ; abs:"mean-field" AND abs:softmax AND abs:Gaussian ; abs:"expected softmax" OR abs:"probit approximation" OR abs:"Laplace bridge"
- abs:quantization AND abs:entropy AND abs:"output distribution" AND (abs:temperature OR abs:sampling) ; abs:quantization AND (abs:"sampling temperature" OR abs:"token entropy" OR abs:"output entropy") ; abs:quantization AND abs:LLM AND abs:diversity AND abs:entropy AND abs:sampling
- abs:"KV cache" AND abs:quantization AND abs:softmax AND (abs:theoretical OR abs:theory OR abs:bound)
- abs:attention AND abs:quantization AND abs:"second-order" AND abs:softmax ; abs:"quantization noise" AND abs:attention AND abs:softmax AND abs:variance ; abs:"quantization error" AND abs:softmax AND abs:attention AND abs:exponential
- abs:"KV cache" AND abs:quantization AND abs:asymmetric AND abs:key AND abs:value ; abs:"KV cache" AND abs:quantization AND abs:"more sensitive"
- abs:"attention entropy" AND (abs:quantization OR abs:quantized)
- abs:"Jensen gap" AND abs:softmax ; abs:"Jensen" AND abs:attention AND abs:quantization
- abs:"per-channel" AND abs:key AND abs:quantization AND abs:"KV cache" AND (abs:optimal OR abs:why OR abs:theoretical)
- abs:"stochastic rounding" AND abs:"KV cache"
- abs:"attention sink" AND abs:key AND abs:norm ; abs:"attention sink" AND (abs:"low norm" OR abs:"low-norm" OR abs:"small norm") ; ti:sink AND (abs:quantization OR abs:quantized) ; abs:quantization AND abs:"attention sink" AND (abs:amplifies OR abs:amplified OR abs:amplification)
- abs:"KV cache" AND abs:quantization AND abs:bias AND abs:correction ; abs:"KV cache" AND abs:quantization AND abs:"attention weights" AND (abs:shift OR abs:distort OR abs:drift) ; abs:"score-space" AND abs:quantization
- abs:"log-partition" AND (abs:quantization OR abs:quantized)
- WebSearch: "vLLM TurboQuant QJL excluded softmax variance amplification"; "quantized keys attention sink amplified KV cache quantization softmax bias large norm keys 2026"; "Jensen bias softmax quantization noise correction σ²/2 attention logits LLM 2026 arxiv"
