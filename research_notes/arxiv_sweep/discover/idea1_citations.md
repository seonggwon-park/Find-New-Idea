# Idea 1 — citation-neighbourhood sweep (Angle C)

Idea: heteroscedastic logit-noise second-order softmax bias; one-scalar correction −σ_j²/2; explains QJL harm, KIVI per-channel-K asymmetry, sink amplification, LM-head temperature shift.

Date of sweep: 2026-10-10. Sources: full texts fetched via `scripts/arxiv_fetch.py` into the scratchpad `papers/` dir; arXiv API searches; web search for vLLM/community material. Quotes below are from downloaded paper text (untrusted data, quoted only).

## Verdict in one paragraph

The core deliverable of Idea 1 — "subtract σ_j²/2 from each quantized key's score, with σ_j² = ‖q‖²Δ_j²/(12d) for per-token uniform quantization, derived from E[e^{s+ε}] = e^{s}·e^{σ²/2} / the cumulant-generating function" — is **already published**: Tuncer, Becker & Pfeil, *Quantized Keys Steal Attention* (arXiv 2605.26266, 25 May 2026, ICML 2026 workshops SCALE and F2S). They derive the exact uniform-noise correction Σ_c log(sinh α_c/α_c), its Taylor form b_i ≈ (1/24d)Σ_c q_c²Δ_{i,c}² = σ_i²/2, note the CGF generality, measure the attention-mass shift, show the correction recovers INT2 video-diffusion quality, run an LLM partial-prefill NLL diagnostic on three LLMs, and in Appendix E show that per-channel key quantization collapses the correction to one scalar shared by all tokens. Independently, D'Alberto (2605.08114, §3.1 and §3.3; follow-up 2607.09683) already states the heteroscedastic/homoscedastic distinction ("if σ_i² is uniform across keys the factor cancels in the softmax ratio; the dangerous regime is when σ_i² is correlated with the attention score") and the K-vs-V rule (V enters linearly, no Jensen amplification), contrary to the idea text's claim that D'Alberto has neither. What remains *not* done in the literature: (a) the full second-order bias of the **normalized** softmax including the denominator (the idea's Theorem 1), hence the homoscedastic "temperature-only" entropy-increase result; (b) the explicit sink-amplification prediction ½(σ_sink² − σ̄²) under per-token keys and its measurement in LLM decoding; (c) the QJL-specific σ_j² ∝ ‖k_j‖² key-norm bias and a correction using the stored norm; (d) the LM-head/vocabulary version and its ECE prediction. These are now increments on 2605.26266, not a new theorem.

## Key papers named in the idea — what they actually prove / claim

### 2406.03482 QJL (Zandieh, Daliri, Han; AAAI 2025)
- Lemma 3.2: ProdQJL unbiased for ⟨q,k⟩. Lemma 3.5: distortion bound; the proof uses Fact 3.3 — the projected key coordinate "is a normal random variable with mean zero and variance ‖k‖₂²" (line ~364), so estimator variance ∝ ‖q‖²‖k‖²/m. Theorem 3.6: (1±ε) relative distortion of attention scores w.h.p.
- Nothing on the softmax expectation; no bias analysis of the attention distribution. The idea's "σ_j² ∝ ‖k_j‖²" follows directly from Lemma 3.5's proof. SUPPORTS the premise.

### 2605.08114 D'Alberto (27 Apr 2026, "process highly assisted by AI")
- §3.1 Eq. (4): "E[e^{⟨q,k_i⟩+ε_i}] = e^{⟨q,k_i⟩}·e^{σ_i²/2} for ε_i ~ N(0,σ_i²). If σ_i² is uniform across keys the factor e^{σ²/2} cancels in the softmax ratio. The dangerous regime is when σ_i² is correlated with the attention score: high-attention keys with large quantisation variance produce asymmetric distortion that cannot cancel."
- §3.3 "Implication for V. The value tensor enters the output linearly ... There is no exponential, so variance is not amplified ... making QJL's unbiased correction genuinely useful here without the 2π variance penalty that makes it harmful for K."
- §4: QJL on K inflates inner-product variance by (π/2)/ε_B ≥ π/2. Empirical (synthetic Beta/fat-tail data, not LLMs): KQV beats QKQV in KL at every budget; crossover in geometric K error at n∈{2,3,5}.
- "Sufficient condition when the Jensen mechanism amplifies superlinearly" is argued, not a theorem with proof; no correction formula; no token-range/sink link; no uniform-noise model. PARTIALLY PREEMPTS the motivation (heteroscedastic vs homoscedastic, K-vs-V rule). The idea text's claim that D'Alberto lacks "systematic bias from token heterogeneity ... K-vs-V allocation rule" is wrong and must be revised.

### 2607.09683 D'Alberto follow-up (14 Jun 2026)
- "Jensen's inequality: The softmax is convex in its argument: E[exp(s+ε)] = exp(s)·exp(σ²/2). QJL adds a zero-mean correction ε with variance σ² ∝ ‖e‖²/m" (residual norm²/m). "QJL on K is only beneficial when residual norms are small and the number of projections m ≫ d_eff." "V-path corrections, which enter T linearly through a weighted sum, do not trigger Jensen amplification." Python-vs-HIP KL gap 0.119 for TQ-QKV vs 0.000 for TQ-KV; gap scales as 1/m (5× larger at m=8 than m=64).
- Synthetic data only. PARTIALLY PREEMPTS the QJL application (variance ∝ residual-norm², m-dependence) but no correction.

### 2402.02750 KIVI (ICML 2024)
- Justification is purely outlier-channel: "per-channel quantization can confine the quantization error to each individual channel" (§3.2); empirical: per-token K gives "almost 5× larger attention score error than per-channel" (line ~605). No softmax-bias theory. SUPPORTS premise that the asymmetry lacks a theoretical explanation.

### 2510.05373 KVLinC (Oct 2025)
- Observation 3: raw keys per-channel + Hadamard values per-token is best; explanation is distributional (Fig. 2), no theory. Linear correction adapters are learned. SUPPORTS premise.

### 2608.20988 Jacobian-guided noise injection (ICML 2026 AdaptFM workshop)
- Theory: Jacobian J_S = diag(p) − pp^T; bounding its norm bounds degradation (first-order); noise std σ_i = α‖J_S‖²_{F,i}. No second-order bias. TANGENTIAL.

### 2606.08635 SpectrumKV (Jun 2026)
- Lemma 2 "Softmax amplification bound" with e^{−2η}p_j ≤ p̃_j ≤ e^{2η}p_j; "We state this as a conjecture rather than a theorem because the precise threshold H* ..." (line ~341). Sink protection at FP16. TANGENTIAL; the idea's reading is accurate.

### 2606.13233 ReSET (Jun 2026)
- NVFP4 "increases incorrect sampling at low-entropy symbolic tokens, while causing over-concentration on a small set of tokens in high-uncertainty reasoning steps"; step-aware temperature scaling, +~2 points. Empirical only. SUPPORTS (already cited correctly).

### 2604.08976 Cacioli (Apr 2026) — MIS-CITED in the idea
- Actually: Llama-3-8B-Instruct at **Q5_K_M vs f16** (llama.cpp GGUF), metacognitive M-ratio profiles uncorrelated across formats (ρ=0.00) while Type-2 AUROC stable (ρ=1.00). It is **not** "GPTQ 4-bit worsens ECE". CONTRADICTS the idea's citation sentence; replace or drop.

### 2605.17757 OSCAR (May 2026)
- Theorem 1: optimality of spectral rotation variants under diagonal residuals; attention-aware covariance; no softmax bias. TANGENTIAL (the idea's "adjacent" note is fine but overstated).

### 2605.26266 Quantized Keys Steal Attention (Tuncer, Becker, Pfeil; 25 May 2026) — THE PREEMPTING PAPER
- Noise model Eq. (5): ε_{i,c} ~ U(−Δ_{i,c}/2, Δ_{i,c}/2), independent across channels (Widrow). Eq. (7): σ_i² = (1/12d)Σ_c q_c²Δ_{i,c}².
- Eq. (8)–(9): "Jensen bias" E[Ẑ_S] ≥ Z_S; attention stealing ΔP_S = P̂_S − P_S.
- Eq. (10): b_i = log E[e^{δ_i}]; Eq. (12) exact: b_i = Σ_c log(sinh(α_c)/α_c), α_c = q_cΔ_{i,c}/(2√d); Eq. (13): b_i ≈ (1/24d)Σ_c q_c²Δ_{i,c}².
- "Comparing Eq. (13) with Eq. (7), the Taylor correction is exactly half the score-space noise variance: b_i ≈ σ_i²/2. This follows from the cumulant generating function ... the leading term is κ₂/2 = σ²/2, which depends only on the variance and not on the specific noise distribution."
- Appendix E (per-channel): "Since {Δ_c} do not depend on i ... The correction reduces to a single scalar shared by all tokens" (Eq. 22–23). "In our experiments, per-token quantization with the token-dependent correction consistently outperforms per-channel quantization with a shared correction."
- Appendix F: extends to QuaRot (q → Hq). Overhead ≈5% latency at g=32. Taylor overestimates at large α (Fig. A1) → mild over-correction at INT2.
- Experiments: MAGI-1, SkyReels-V2, HY-WorldPlay at INT2 (RTN, QuaRot+RTN, with QVG); "using 2.19 effective bits with our method outperforms 4.38 effective bits without correction". Fig. 4: ΔP_S strongly positive under INT2, correction centres it near zero.
- §5.4 / App. O: LLM partial prefill (chunked), Mistral-7B-Instruct-v0.3, Llama-3.2-1B, Qwen2.5 on LongBench-Pro; teacher-forced NLL: INT2 worsens, Taylor correction improves over plain INT2; Fig. A11 (Llama-3.2-1B): "quantization increases attention weights in the cached block of tokens and decreases them in the current chunk".
- Stated limitations: "Standard single-token decoding offers less headroom for the correction because many cached tokens compete with only one unquantized current token." "It works best when cached attention is spread over enough tokens for score perturbations to average out." FP/MXFP/NVFP open.
- Gap they leave (where Idea 1 can still live): they unbias the *numerator* only; no expansion of the normalized softmax, no entropy result, no sink analysis, no QJL/key-norm case, no LM head. In pure decoding with all keys per-token-quantized, their correction is exactly the idea's; with per-channel keys their correction is a constant and cancels — i.e., they implicitly contain the idea's "per-channel ⇒ no relative bias" claim.

## Other candidates found

- **2608.14191 "KV Cache Compression Through the Lens of Transform Coding" (Aug 2026)**: Theorem 1 gives E‖o−ô‖² = D_K + D_V + R under a white-noise model (Assumption 1, zero-mean, independent across channels and tokens), D_K = Σ_{i,c} a_i²‖v_iW_O − o‖² q_c² E[(δk_{ic})²]; Lemma 1 bounds the second-order softmax Taylor remainder (|r_i| ≤ 7A₀³). Same expansion machinery, but it computes the second moment (MSE), not the first-moment bias of p̂, and assumes E[(δk_{ic})²] "the same across tokens". PARTIALLY PREEMPTS the "second-order Taylor of softmax under KV white noise" framing; the bias result is complementary.
- **2508.04257 KVSink (COLM 2025)**: Table 1 — static per-token quantization excluding sink tokens cuts error "up to 81.1% for Keys and 68.2% for Values"; per-channel dynamic K: groups containing sinks have 16.3–29.2% larger error; Fig. 7: quantization visibly changes sink attention scores and the "attention bias" Σ_{i∈S}p_it v_i, more at lower bits. No per-token-vs-per-channel sink-mass direction, no mechanism. PARTIALLY PREEMPTS Experiment 1's measurement; the idea must cite it and state the sign prediction as the new content.
- **2608.03854 (AAAI 2026 Fall Symposium)**: NF4 INT4 ECE changes "mixed, no reliable direction"; fitted T = 1.25–1.67 (>1 means over-peaked, the opposite of the idea's "4-bit raises temperature"). CONTRADICTS the LM-head direction weakly (backbone quantization, not LM head).
- **2609.07901 "Quantization Amplifies Determinism, Not Bias" (Sep 2026)**: AWQ-int4 Qwen3-8B: "decision-token entropy increases by 0.091 bits (CI [+0.020, +0.163], p=.015), top-1 probability falls by 3.8pp (p=.024)" while semantic collision +2.6pp. SUPPORTS the flattening/temperature-shift prediction, though AWQ typically leaves lm_head unquantized, so it is backbone noise, not the idea's LM-head mechanism.
- **1703.00091 Daunizeau 2017**: second-order Taylor approximations to E[softmax(x)] for Gaussian x via gradient/Hessian of (log-)softmax (eqs. ~28–30), plus moment-matched sigmoid approximations. The idea's Theorem 1 is this delta-method expansion applied with diagonal Σ. The *math* is not new; the application is. TANGENTIAL / partially preempts the math.
- **2006.07584 Lu et al. mean-field Gaussian-softmax**: per-class adaptive temperature T_k depending on class-k variance ("If the activation on the k-th class has high variance, the temperature for that category is high"). Different (non shift-invariant) approximation; relevant as prior framing of "heteroscedastic logit noise = per-class temperature". TANGENTIAL.
- **2606.23406 HyperQuant (Jun 2026)**: proves subtractive lattice dither is strictly per-vector inner-product unbiased (Appendix A), contrasts "QJL alone is biased per-vector" (A.5); applies bias correction to the KV cache. NEW HYPOTHESIS: Idea 1 predicts that dither-unbiased *keys* should hurt via softmax exactly like QJL; HyperQuant reports all bias-correction variants within 0.04 PPL at ≥2.5 bps, so this is a cheap falsification test.
- **2410.13212 AsymKV**: Proposition 1 (linear error propagation); empirical: key-quantization MSE gap "further amplified after the softmax". SUPPORTS K-vs-V asymmetry, no bias.
- **2605.20868 Runtime-Certified Bounded-Error Quantized Attention (May 2026)**: Theorems 4.1/4.3/4.4 with exp(2Δ) multiplicative bounds (same family as SpectrumKV), per-channel INT8 keys. TANGENTIAL.
- **2309.01729 Pandey et al. (Qualcomm, 2023) "Softmax Bias Correction for Quantized Generative Models"**: bias from rounding small *post-softmax* probabilities to zero in 8-bit; offline offset. Different mechanism. TANGENTIAL (but cite to avoid "bias correction for softmax" confusion).
- **2605.19660 OScaR (Su et al., May 2026)**: "QJL is excluded as it degrades performance. See Appendix N." Token-norm imbalance as per-channel bottleneck. SUPPORTS motivation.
- **2609.11716 "Why Does PTQ Work?" (Sep 2026)**: Theorems 2–3 on LM-head geometry: top-ranked tokens have smaller relative log-prob change under isotropic rotation of h_LM. Error is in h, not W; TANGENTIAL to the LM-head claim but a competing explanation for rank-dependent effects.
- **Non-arXiv**: vLLM TurboQuant config docs / PR #38479: "QJL is intentionally omitted — community consensus (5+ independent groups) found it hurts attention quality by amplifying variance through softmax." Intel community blog: disabling QJL is optimal in production (~30% less decode overhead, no measurable accuracy loss). Dissent: AmesianX llama.cpp fork re-enabled QJL for head_dim=64 ("without QJL: repetition loops at turn 3-4"). The "5+ groups" count is self-reported.

## Not found
- No paper derives the second-order bias of the normalized softmax under heteroscedastic noise and the resulting entropy increase under homoscedastic noise, in the KV-quantization context.
- No paper derives a σ_j² ∝ ‖k_j‖² attention bias for QJL or proposes using the stored key norm to correct it.
- No paper derives an LM-head per-vocabulary-row correction ‖h‖²Δ_v²/24.
- No paper measures sink-mass change per-token vs per-channel key quantization at matched bits in LLM decoding.
- arXiv searches for "heteroscedastic AND softmax", "Jensen gap AND softmax", "noisy logits AND attention", "logit AND bias correction AND quantization" returned nothing relevant.

## Recommended revision of Idea 1
1. Cite 2605.26266 as the primary prior work; reposition the idea as (i) the normalized-softmax second-order theorem (bias + entropy), (ii) the sink-amplification prediction and test, (iii) the QJL key-norm case with ‖k_j‖-based correction, (iv) the LM-head analogue. Drop "one-scalar correction" as a headline claim.
2. Fix the D'Alberto characterization (he states the heteroscedastic cancellation and the K-vs-V rule).
3. Replace the 2604.08976 citation (it is Q5_K_M metacognition, not GPTQ-ECE); cite 2608.03854 and 2609.07901 with their mixed/flattening findings.
4. Add 2608.14191 and Daunizeau 2017 as the methodological antecedents of the Taylor expansion.
5. Add the HyperQuant-dither falsification test.

## Queries run
arXiv API (`scripts/arxiv_search.py`):
- abs:softmax AND abs:quantization AND (abs:bias OR abs:Jensen)
- abs:"attention sink" AND (abs:quantization OR abs:quantized) AND abs:key
- abs:"unbiased estimator" AND abs:attention AND (abs:quantization OR abs:quantized OR abs:sketch)
- abs:heteroscedastic AND abs:softmax
- abs:"Jensen" AND abs:attention AND (abs:quantization OR abs:quantized OR abs:noise)
- abs:"per-channel" AND abs:"per-token" AND abs:"KV cache" AND (abs:theory OR abs:theoretical OR abs:analysis)
- abs:"lm head" AND (abs:quantization OR abs:quantized) AND (abs:temperature OR abs:calibration OR abs:entropy)
- abs:logit AND abs:noise AND abs:softmax AND abs:temperature
- abs:"KV cache" AND (abs:quantization OR abs:quantized) AND (abs:"bias correction" OR abs:"debias" OR abs:"unbiased")
- abs:quantization AND abs:softmax AND abs:"second-order" AND abs:attention
- (abs:QJL OR abs:TurboQuant OR abs:PolarQuant) AND (abs:softmax OR abs:variance OR abs:attention)
- abs:"attention entropy" AND (abs:quantization OR abs:quantized) AND abs:"KV cache"
- abs:"massive activation" AND abs:"KV cache" AND (abs:quantization OR abs:quantized)
- abs:"expected calibration error" AND (abs:quantization OR abs:quantized) AND (abs:LLM OR abs:"language model")
- abs:"Jensen gap" AND (abs:softmax OR abs:"log-sum-exp" OR abs:logsumexp)
- abs:softmax AND abs:"Gaussian noise" AND abs:logits AND (abs:expectation OR abs:expected OR abs:bias)
- abs:key AND abs:value AND abs:asymmetr* AND abs:"KV cache" AND abs:quantization AND (abs:sensitivity OR abs:theory)
- abs:"KV cache" AND abs:quantization AND abs:sink AND (abs:"per-token" OR abs:"per-channel")
- abs:temperature AND abs:quantization AND abs:entropy AND abs:sampling AND abs:"language model"
- abs:"stochastic rounding" AND abs:attention AND abs:softmax AND abs:bias
- abs:"log-sum-exp" AND abs:noise AND abs:bias AND abs:variance
- ti:softmax AND (ti:noise OR ti:noisy OR ti:perturbation) AND (ti:bias OR ti:expectation)
- abs:"attention weights" AND (abs:"quantization noise" OR abs:"quantization error") AND abs:bias AND abs:softmax
- abs:"query norm" AND abs:"KV cache" AND (abs:quantization OR abs:quantized) AND abs:correction
- abs:quantized AND abs:logit AND abs:entropy AND (abs:flatter OR abs:flatten OR abs:sharper OR abs:"distribution shift") AND abs:LLM
- abs:"noisy logits" AND (abs:softmax OR abs:attention)
- abs:"softmax" AND abs:"expected" AND abs:"Gaussian" AND abs:"probit" AND abs:approximation
- abs:"KV cache" AND abs:quantization AND abs:theory AND abs:softmax AND (abs:error OR abs:perturbation) AND abs:bound
- abs:"attention sink" AND abs:quantization AND (abs:amplif* OR abs:inflat* OR abs:"attention mass")
- ti:"KV cache" AND ti:quantization AND (ti:theory OR ti:theoretical OR ti:analysis OR ti:understanding)
- abs:"mean-field" AND abs:softmax AND abs:Gaussian AND (abs:logits OR abs:"pre-activation")
- abs:"KV cache" AND (abs:quantization OR abs:quantized) AND abs:"key norm"
- (abs:"lm_head" OR abs:"output head" OR abs:"unembedding" OR abs:"LM head") AND (abs:quantization OR abs:quantized) AND abs:LLM
- abs:logit AND abs:quantization AND abs:"temperature scaling" AND abs:calibration AND (abs:LLM OR abs:"language model")
- abs:"attention sink" AND abs:"KV cache" AND (abs:quantization OR abs:quantized)
- abs:TurboQuant AND abs:QJL
- abs:logit AND abs:"bias correction" AND (abs:quantization OR abs:quantized)
- abs:softmax AND (abs:"normal variables" OR abs:"Gaussian variables" OR abs:"Gaussian random") AND (abs:moments OR abs:expectation)
- ti:"Gaussian-softmax" OR ti:"softmax-Gaussian" OR (ti:softmax AND ti:Gaussian AND ti:integral)
- abs:"key cache" AND abs:quantization AND abs:"attention scores" AND abs:bias AND abs:softmax

Web searches:
- vLLM TurboQuant QJL excluded softmax amplifies variance community consensus
- "Jensen bias" softmax attention KV cache quantization correction 2026
- heteroscedastic logit noise softmax expected value bias high-variance tokens correction sigma^2/2
- LM head quantization INT4 calibration ECE entropy temperature shift sampling bias rare tokens
- expected softmax under Gaussian logit noise closed form second-order Taylor bias toward higher variance class
- KIVI per-channel key per-token value quantization theoretical explanation softmax bias attention sink amplification per-token key
- quantized LLM output distribution flatter higher entropy 4-bit weights temperature effect token probabilities study
- noise injection attention logits increases attention entropy expected softmax flattening analysis "temperature" equivalence
- per-token key quantization biases attention toward tokens with large quantization range attention sink softmax 2026 blog OR preprint
- QJL TurboQuant key norm large-norm keys attention bias softmax variance proportional to key norm analysis

Full texts fetched and grepped: 2406.03482, 2605.08114, 2402.02750, 2510.05373, 2608.20988, 2606.08635, 2606.13233, 2604.08976, 2605.17757, 2605.26266, 2309.01729, 2508.04257, 2606.31519, 2605.01910, 2608.02703, 2605.29756, 2410.13212, 2602.23200, 2504.19874, 2605.20868, 2608.14191, 2607.09683, 2610.11245, 2608.03854, 2605.19660, 2602.17691, 2609.07901, 2610.11214, 2605.06675, 2606.02288, 2502.03366, 2606.06521, 2501.19392, 2606.23406, 1703.00091, 2006.07584, 2609.11716.
