# Serving-side quantization: 10 new candidates (generated 2026-10-11)

Lens: KV cache, attention kernels, SSM/hybrid state, speculative decoding, long context, batching/prefill-decode, MoE serving.
Inputs read: report §0 and §0.4 (preempted list), `arxiv_sweep/recent/*.md`, `hypotheses/triage.md`, limitation sections of `theorems/{2602.10718,2605.17757,2606.24033,2608.02691,2604.12374,2402.02750}.md`.
Tool budget: about 55 calls. Every candidate below passed at least 4 `arxiv_search.py` queries, including at least 2 with `--sort date --from 20250601` and 1 in adjacent-field vocabulary. The query lists are recorded per candidate. Paper text is quoted as data.

Read this first:
- "Not found" means not found by these literal arXiv-API queries. The API does no stemming, and a few compound queries returned 0 hits, so these screens are weaker than a full citation sweep.
- The search killed or narrowed many ideas (list at the end). The most important case: the "TV is first-order and KL is second-order, so acceptance and flips track TV" law **is already published**. It appears in 2610.00694 (flip rate ≈ TV, median ratio 1.05, and TV predicts vLLM greedy draft acceptance within 1.1–2.4%) and in 2605.02404 (the EAR = 1 − TV metric). Candidate 1 is therefore reduced to the Bayes-optimal draft theorem.
- Rough strength ranking (my judgement): 9 (V-cache noise shaping) ≈ 2 (winner's curse in quantized selection) > 6 (bias amplification in hybrids) ≈ 4 (common-mode error in parallel sampling) > 3 (fan-out allocation) ≈ 5 (EVT context shrinkage) > 7 (dispatch/combine) > 1 (posterior-quantile draft) > 8 (token-vs-precision α) ≈ 10 (route-flip tie theorem).

---

## 1. Posterior-quantile drafts: the acceptance-optimal draft under known quantization noise (speculative decoding)

**Hypothesis.** A quantized self-draft observes noisy logits ẑ = z + δ, with heteroscedastic noise δ_x ~ N(0, σ_x²) across the vocabulary. The draft that maximizes expected acceptance is not p̂ = softmax(ẑ). It is a per-token posterior **quantile** of p(x). To first order this is q*(x) ∝ p̂(x)·exp(−z_λ·s_x), where s_x is the delta-method std of log p(x).

This is a first-order (σ) shrinkage of uncertain tokens. It is different in order and in shape from the −σ²/2 mean correction (2605.26266) and from global temperature tuning. When the noise is homoscedastic, the gain is exactly zero.

**Gap evidence.**
- 2610.00694 (§1, l.36, l.61–64) and 2605.02404 (l.302–307, EAR) establish acceptance = 1 − TV(p, q) and use it as a *metric* of the quantized model. Neither optimizes the draft given knowledge of the noise.
- 2605.26266 and 2608.14191 correct the *mean* bias with −σ²/2. That is a KL/MSE-motivated correction, and acceptance is a TV (L1) objective.
- 2610.10411 optimizes drafters for expected decoding rounds, with no observation noise.
- FlexDraft 2605.20022 calibrates using bonus tokens, not a noise posterior.
- Queries for "optimal draft" combined with uncertainty or posterior returned 0 hits.

**Math core.** Theorem (provable, elementary):
- Setup: let P(x) be the random target probability, distributed as its posterior given the draft's observation, and let q lie in the simplex.
- Concavity: Φ(q) = E Σ_x min(P(x), q(x)) is concave, and ∂Φ/∂q(x) = Pr[P(x) > q(x)].
- KKT: the optimum satisfies Pr[P(x) > q*(x)] = λ on supp q*. Hence q*(x) = F_x^{-1}(1 − λ), with λ fixed by normalization.
- Corollary: if s_x is constant over the support, then q* = p̂.
- Greedy target: the optimal proposal is argmax_x Pr[x = argmax z | ẑ], which differs from argmax ẑ when margins are below the noise scale.
- Assumptions: Gaussian logit noise with known per-token variance (estimated offline from paired FP/quantized logits), a flat prior on z, continuous marginals.
- Additional result (also provable): the optimal (draft bits b, draft length γ) given cost(b) and α(b).

**Decisive experiment (2–4 GPU-days).**
- Target: Llama-3.1-8B / Qwen3-8B in BF16.
- Drafts: the same weights at W3 GPTQ, at W2 (QTIP), and with KIVI-2 KV.
- σ_x: estimated on 512 calibration sequences.
- Tasks and settings: MT-Bench, GSM8K and HumanEval; T ∈ {0, 0.7}; γ ∈ {3, 5}.
- Arms: p̂; best global temperature; −σ²/2 correction; posterior quantile.
- **Fails if** the quantile draft gains less than 2% mean accepted length over the best-temperature p̂ at both W3 and W2, or if its gain is reproduced by a global temperature.

**Nearest papers.**
- 2610.00694 and 2605.02404 (acceptance/TV as a measurement; they do not shape the draft).
- QSpec/Quasar 2603.01399 and SpecQuant 2609.21704 (systems).
- 2605.26266 (mean-bias correction).

**Queries.**
- `abs:"speculative decoding" AND (abs:quantized OR abs:quantization) AND abs:draft` (date, ≥20250601)
- `abs:"acceptance rate" AND (abs:quantization OR abs:quantized) AND (abs:"total variation" OR abs:theory OR abs:bound)` (date) → 0
- `(ti:speculative OR abs:"speculative decoding") AND (abs:"logit noise" OR abs:"noisy logits" OR abs:"draft distribution") AND (abs:optimal OR abs:theorem)` (date)
- `abs:"speculative decoding" AND abs:"optimal draft" AND (abs:uncertainty OR abs:noisy OR abs:posterior)` → 0
- adjacent: `abs:"maximal coupling" AND (abs:"noisy" OR abs:"estimated") AND abs:proposal`
- `(abs:"speculative") AND (abs:"draft temperature" OR abs:"temperature calibration" OR abs:"calibrated draft" OR abs:"draft calibration")` (date)
- `(abs:"self-speculative") AND (abs:quantiz OR abs:"low-bit") AND (abs:theory OR abs:analysis OR abs:bound)` (date)
- `(abs:"speculative decoding") AND (abs:"total variation") AND (abs:"compressed" OR abs:"quantized" OR abs:"approximate")` → found 2610.00694, which kills the "first-order law" half

**Risk.** In real quantized models, the vocabulary heterogeneity of s_x may be small, so the gain would be second-order in Var_x(s_x). The bound on the gain should be computed before running GPUs.

---

## 2. Winner's curse in quantized-key selection: empirical-Bayes ranking for sparse attention, eviction and recompute selection

**Hypothesis.** Many methods select tokens with noisy, quantized scores ŝ_i = s_i + δ_i:
- DeepSeek-V3.2 lightning-indexer-style FP8 scoring;
- Quest / RaBitQCache / Self-Indexing page or token top-k from compressed keys;
- H2O-style eviction with quantized K;
- deviation-guided recompute.

When σ_i is heteroscedastic (∝‖k_i‖·Δ_i, or an estimator variance), ranking by ŝ systematically over-selects high-noise tokens and loses attention mass. Ranking by the posterior expected mass E[e^{s_i}|ŝ_i] recovers the mass. An *unbiased* score estimator does not remove this selection bias, and neither does a Hadamard rotation, which preserves ‖k‖.

**Gap evidence.**
- RaBitQCache 2606.31519 (l.23, l.113) presents an *unbiased* attention-score estimator as the cure for biased proxies. Unbiased estimation ≠ unbiased top-k selection.
- Fathom 2609.17652 uses score variance to set read depth but still ranks by the estimate.
- BreadthKV 2610.05685: an oracle keeping 10% of tokens still loses about 13% of attention mass, so selection mistakes cost mass directly.
- No selection-under-heteroscedastic-noise treatment was found.

**Math core.**
- Theorem (exact; follows from linearity): suppose s_i | ŝ_i ~ N(m_i, v_i) independently, with m_i = (τ²ŝ_i + σ_i²μ)/(τ²+σ_i²) and v_i = τ²σ_i²/(τ²+σ_i²) under a per-query prior N(μ, τ²) fit by empirical Bayes from that query's ŝ histogram. Then the size-k set maximizing E Σ_{i∈S} e^{s_i} is the top-k by m_i + v_i/2.
- Corollary 1: ranking by ŝ is optimal iff σ_i is constant.
- Corollary 2 (winner's curse): E[s_i − ŝ_i | ŝ_i] = −σ_i²(ŝ_i − μ)/(τ²+σ_i²), which is strongly negative in the upper tail.
- Corollary 3: the mass loss of naive ranking is a closed-form sum over swapped pairs.
- Assumptions: Gaussian prior per query, independent noise, σ_i computable from stored scales (per-token INT: σ_i² ≈ ‖q‖²Δ_i²/12).

**Decisive experiment (3–5 GPU-days).**
- Models and tasks: Llama-3.1-8B and Qwen3-8B; RULER/LongBench at 32–128K.
- Selection keys: 1/2/4-bit (KIVI per-channel, per-token INT, RaBitQ 1-bit).
- Budgets: Quest-style page selection and token top-k at 1–10% budgets.
- Metrics: captured exact mass, oracle top-k recall, accuracy.
- Arms: ŝ; m + v/2 (EB); ŝ + σ²/2 (flat prior).
- **Fails if** the EB ranking captures less than 1 percentage point more mass than naive ranking in every setting, or if the gain is uncorrelated with the dispersion of σ_i.

**Nearest papers.** RaBitQCache 2606.31519, Self-Indexing KVCache 2603.14224/2609.13205, Fathom 2609.17652, PQ-HSA 2609.33746, KVBoost 2608.21362, Quest, ZipCache. None models selection bias under heteroscedastic noise.

**Queries.**
- `(abs:"sparse attention" OR abs:"top-k attention" OR abs:"token selection" OR abs:indexer) AND (abs:quantized OR abs:"low-bit" OR abs:FP8 OR abs:"1-bit" OR abs:"4-bit") AND (abs:keys OR abs:key)` (date)
- `(abs:"KV cache" OR abs:"sparse attention") AND (abs:"selection bias" OR abs:"winner" OR abs:shrinkage OR abs:"empirical Bayes" OR abs:"noisy scores" OR abs:"estimation error")` (date)
- `(abs:"lightning indexer" OR abs:"DeepSeek Sparse Attention" OR abs:DSA) AND (abs:FP8 OR abs:quantiz OR abs:precision)` (date ≥20250901)
- adjacent: `(abs:"winner's curse" OR abs:"selection bias") AND (abs:"top-k" OR abs:ranking) AND (abs:heteroscedastic OR abs:"unequal variance" OR abs:"noisy estimates")` → 0
- `(abs:"heavy hitter" OR abs:eviction) AND abs:"KV cache" AND (abs:quantiz) AND (abs:"attention score" OR abs:saliency) AND (abs:bias OR abs:"inaccurate")` → 0

**Note.** That the DeepSeek-V3.2 indexer runs in FP8 is my recollection of its tech report. It was not re-verified this session.

---

## 3. Fan-out-aware precision: the value of a bit scales with coherent read fan-out (prefix caching, CLA/MLA sharing, SWA vs global, elastic downgrade)

**Hypothesis.** The marginal value of a bit in a cached tensor is set by its *coherent fan-out*: the Gram of the read-Jacobians, i.e. how many outputs read the same stored error with aligned influence. Under a byte budget, this gives:
- a prefix shared by N requests or samples gets about +½log2(N_coh) bits over per-request suffixes;
- SWA-layer KV gets ½log2(T_rem/W) fewer bits than global-layer KV;
- under memory pressure, requests near completion are downgraded first.

**Gap evidence.**
- Mix-Quant 2605.20315 reports that prefill/prompt-side quantization has "substantial quantization redundancy". A naive fan-out view predicts the opposite, and no paper resolves this. The resolution proposed here: independent per-token errors are averaged by attention (weight Σp_i²), and only *coherent* reuse amplifies.
- MorphServe 2506.02006 and DPS 2609.34380 change precision under load without a per-tensor value rule.
- KV allocators (Block-GTQ 2606.24033 Sec. 7 "multi-GPU and batched serving are open directions"; ReadKV 2610.11245; OSCAR 2605.17757) treat each request's cache in isolation.

**Math core.**
- Lemma: for a cached entry with error e that is read by queries r with first-order maps a_r J_r, the output error is E‖Σ_r a_r J_r e‖² = tr(G Σ_e), with G = (Σ_r a_r J_r)ᵀ(Σ_r a_r J_r). The effective weight ranges from incoherent Σ_r‖a_rJ_r‖² to coherent (Σ_r‖a_rJ_r‖)².
- Theorem: reverse water-filling with storage multiplicity c_j and weights w_j = tr(G_j)/d_j gives b_j = ½log2(w_j/(λc_j)) on active components.
- Assumptions: high-rate quantization, independent errors across entries, first-order linearization.

**Decisive experiment (≈1 GPU-week).**
- Models: Qwen3-8B / Llama-3.1-8B.
- (a) Shared prefix with N ∈ {1, 8, 32} samples on GSM8K/MATH self-consistency, comparing a prefix/suffix bit split with a uniform split at equal bytes.
- (b) Gemma-3: local vs global KV bit split.
- (c) Measure tr(G) coherence directly with JVPs.
- **Fails if** the fan-out split beats uniform by less than 0.5 pt (or KL-equivalent) at N = 32, there is no gain in the SWA/global split, or measured coherence does not rank the gains.

**Nearest papers.** Mix-Quant 2605.20315, MorphServe 2506.02006, DPS 2609.34380, Block-GTQ 2606.24033, ReadKV 2610.11245, CommonKV 2508.16134 (cross-layer sharing, no precision rule).

**Queries.**
- `(abs:"KV cache") AND (abs:quantization OR abs:quantized) AND (abs:"prefix sharing" OR abs:"shared prefix" OR abs:"prefix caching" OR abs:"parallel sampling" OR abs:"best-of-n" OR abs:"self-consistency")` (date)
- `(abs:quantization OR abs:quantized) AND (abs:"self-consistency" OR abs:"majority voting" OR abs:"test-time scaling" OR abs:"parallel sampling") AND (abs:correlated OR abs:"common" OR abs:diversity)` (date)
- `(abs:"KV cache") AND abs:"bit allocation" AND (abs:prompt OR abs:prefix OR abs:"generated tokens")` → 0
- adjacent: `(abs:"correlated errors" OR abs:"common-mode") AND (abs:"majority vote" OR abs:ensemble) AND (abs:quantization OR abs:compression OR abs:"shared")`
- `(abs:"cross-layer" AND abs:"KV") AND (abs:sharing OR abs:shared) AND (abs:quantiz)` (date)
- `(abs:"KV cache") AND (abs:"under load" OR abs:"bursty" OR abs:"load spike" OR abs:"memory pressure") AND (abs:"precision" OR abs:quantiz OR abs:"bit-width")` (date)
- `(abs:"KV cache") AND (abs:requantiz OR abs:"re-quantiz" OR abs:"progressive quantization" OR abs:"adaptive quantization") AND (abs:serving OR abs:scheduler OR abs:preemption)` (date)

---

## 4. Common-mode quantization error defeats self-consistency; lazy prompt-KV quantization fixes it (parallel sampling / test-time scaling)

**Hypothesis.** The quantization error in a shared prompt KV is *common-mode* across N parallel samples. It is also concentrated in the early decode steps, while sample queries are still similar. Consequences:
- (i) The majority@N gap between quantized and FP KV does not shrink with N, and may grow. Per-sample independent error at equal per-sample KL does shrink.
- (ii) Keeping the prompt KV unquantized only for the first k decode steps ("lazy quantization", transient memory shared across samples) recovers most of the gap.

**Gap evidence.**
- Prior quantized-reasoning work, e.g. 2606.00206 and 2610.00694, measures per-sample degradation only.
- No paper found separates common-mode from idiosyncratic quantization error under test-time scaling.
- Prefix sharing for n > 1 sampling is the default in vLLM/SGLang, so the common-mode regime is the deployed one.

**Math core.**
- Proposition (provable): let problem-level correctness probability be π(c + ξ_s), with c common and ξ_s independent. As N → ∞, majority accuracy → 1{π̄(c) > ½}, where π̄(c) = E_ξ π(c + ξ). The asymptotic gap is therefore Pr[sign(π_FP − ½) ≠ sign(π̄(c) − ½)]. That is first-order in c, weighted by the density of problems with π ≈ ½. Independent ξ enters only through E_ξ, at second order.
- Lemma: the cross-sample correlation of logit errors at step t equals the alignment of the samples' queries against the shared ΔK. It decays as the samples diverge, which justifies a finite lazy window.

**Decisive experiment (≈1 GPU-week).**
- Models and tasks: Qwen3-8B and R1-Distill-Qwen-7B on AIME24/25, MATH500 and GPQA.
- Arms:
  - KIVI-2/3 shared prompt KV;
  - control: per-sample re-quantization with independent dither seeds at matched KL.
- N ∈ {1, 4, 16, 64}; lazy window k ∈ {0, 16, 64, 256}.
- Also measure cross-sample error correlation vs step.
- **Fails if** under the shared prompt the N = 64 gap is ≤ 50% of the N = 1 gap (i.e., it averages out), and lazy k = 64 recovers less than 30%.

**Nearest papers.** 2606.00206 (quantized reasoners think longer), 2610.00694 (flips/TV per sample), 2605.20315 (prefill-side precision). None treats cross-sample correlation.

**Queries.**
- `(abs:quantization OR abs:quantized) AND (abs:"self-consistency" OR abs:"majority voting" OR abs:"test-time scaling" OR abs:"parallel sampling") AND (abs:correlated OR abs:"common" OR abs:diversity)` (date)
- `(abs:quantization OR abs:quantized) AND (abs:"majority" OR abs:"pass@k" OR abs:"best-of-N") AND (abs:reasoning) AND abs:"KV"` (date)
- `(abs:"KV cache") AND (abs:quantization OR abs:quantized) AND (... abs:"parallel sampling" OR abs:"best-of-n" OR abs:"self-consistency")` (date)
- adjacent: `(abs:"correlated errors" OR abs:"common-mode") AND (abs:"majority vote" OR abs:ensemble) AND (...)`
- `(abs:GRPO OR abs:"group relative") AND (abs:quantiz OR abs:FP8) AND (abs:"KV cache" OR abs:prefix OR abs:"shared prompt")` (date) → nothing relevant

**Note.** An RL twist was considered and dropped. Group-normalized advantages cancel common-mode *reward* shifts, but not importance-ratio mismatch, so the claim was not sharp.

---

## 5. KV quantization shrinks usable context polynomially: N*_q = (N*_FP)^{1/(1+r)} (long context)

**Hypothesis.** In a retrieval head with N distractors, KV logit noise of variance σ_q² inflates the distractor maximum to √(σ_d²+σ_q²)√(2 ln N). The critical context length therefore obeys ln N*_q = ln N*_FP/(1+r), with r = σ_q²/σ_d². Consequences:
- quantization looks lossless far below the model's native retrieval limit and fails sharply near it;
- the bits needed diverge as N → N*_FP: b(N) ≈ ½log2(c·ln N/(σ_d²(ln N*_FP − ln N)));
- the weight-quantization gap shrinks with position (consistent with 2609.04098, "32K perplexity gap shrinks with position"), while the KV-quantization gap grows (KVarN 2606.03458, pseudo-decode accumulation);
- length-dependent inverse temperature (2605.12697) scales noise and margin equally, so it cannot fix hard retrieval.

**Gap evidence.**
- 2609.22101 Theorem 1 proves that the margin must scale as Ω(√log N) against distractors, with no quantization.
- WitCert 2607.28699's bound is worst-case and vacuous at 1–2 bits.
- KV long-context degradation reports are purely empirical. Queries for a length law returned 0.

**Math core.**
- Theorem (Gaussian EVT; extends the 2609.22101 setting): needle score μ + m, distractors iid N(μ, σ_d²), and iid N(0, σ_q²) quantization noise on all scores. Then Pr[retrieve] → 1 iff m > (1+o(1))√(2(σ_d²+σ_q²) ln N). Hence ln N* = m²/(2(σ_d²+σ_q²)).
- Extension to heteroscedastic σ_q,i via a Gumbel mixture.
- This is a corollary-level theorem. The contribution is the length law and its test.

**Decisive experiment (≈1 GPU-week).**
- Models: Llama-3.1-8B-Instruct (128K) and Qwen3-8B.
- Tasks: RULER NIAH with controlled same-format hard negatives (as in 2609.22101), N from 4K to 128K.
- KV settings: INT2/3/4 and FP8.
- Measure per-retrieval-head m, σ_d and σ_q directly, then fit N*_q against N*_FP.
- **Fails if** the measured ln-length ratio deviates from 1/(1+r) by more than 30% across bit-widths, or if the onset does not move with r.

**Nearest papers.** 2609.22101 (EVT poisoning, no quantization), 2605.12697 (temperature scaling), WitCert 2607.28699, KVarN 2606.03458.

**Queries.**
- `(abs:"KV cache") AND (abs:quantization OR abs:quantized) AND (abs:"context length") AND (abs:"extreme value" OR abs:"scaling law" OR abs:"log n" OR abs:theory OR abs:theorem)` (date) → 0
- `(abs:"needle" OR abs:retrieval) AND (abs:"KV cache" OR abs:attention) AND (abs:quantization OR abs:"low-bit") AND (abs:margin OR abs:"distractor")` (date)
- adjacent: `(abs:softmax OR abs:attention) AND (abs:"extreme value" OR abs:Gumbel OR abs:"sqrt(2 log n)" OR abs:"maximum of") AND (abs:"long context" OR abs:"sequence length" OR abs:dilution)` → 2609.22101, 2605.12697
- `(ti:quantization OR ti:quantized) AND (ti:"long context" OR ti:"long-context") AND (abs:theory OR abs:law OR abs:predict)` (date ≥20250101) → 0

**Risk.** This is a corollary of 2609.22101. Reviewers may call it thin unless the measured law holds quantitatively across models.

---

## 6. Slow recurrent channels amplify upstream quantization *bias*, not variance: resolving the GDN-survives vs RTN-state-explodes conflict (hybrid SSM/attention)

**Hypothesis.** In hybrids, an upstream error with conditional mean μ over a channel's timescale is amplified by 1/(1−a). Zero-mean error is amplified only by 1/√(1−a²). The bias/variance harm ratio is therefore (μ²/v)(1+a)/(1−a). Upstream sources include W4A4 in_proj, attention-output errors from KV quantization, and state writes.

This explains three published results:
- NVFP4 W4A4 on all 496 linear layers of a GDN hybrid matches BF16, and its gap shrinks with position (2609.04098). Those errors are quasi-zero-mean.
- RTN state-cache writes are up to 400× worse than Weyl dither (2609.39185).
- Nemotron 3 Super needs Philox SR on its SSM cache to avoid verbosity, and "Naive INT16 quantization did not improve verbosity" (2604.12374 §4.3).

The last two are *biased* (deadband) errors.

Prediction: on hybrids, bias correction or dithering of upstream quantizers buys gains ∝ Σ_c 1/(1−a_c)², which are small on dense models.

**Gap evidence.**
- 2609.30950 Prop. 1 assumes "uncorrelated noise", so there is no bias term.
- Nemotron 3 Super Eq. (3) unrolls the recurrence with no stochastic model and no bound (theorems note).
- No paper reconciles the three results above. Searches found none.

**Math core.**
- Proposition (provable): for x_t = a x_{t−1} + b u_t with input error e_t = μ_t + ξ_t (μ slowly varying, ξ white with variance v), the stationary error has mean bμ/(1−a) and variance b²v/(1−a²).
- Extension to delta-rule GDN, (I − βkkᵀ)α: bias along repeatedly written key directions is erased, and out-of-span bias persists. This predicts GDN is less sensitive than Mamba-2.
- Assumptions: linearization; slowly varying μ relative to 1/(1−a).

**Decisive experiment (≈1 GPU-week).**
- Models: Mamba-2-2.7B, Nemotron-H-8B or Falcon-H1-7B, a GDN hybrid, and a dense control (Llama-3.1-8B).
- (1) Measure per-channel windowed mean error μ_c for RTN / SR / dithered W4A4 and KV4.
- (2) Regress ΔNLL at 32K on Σμ_c²/(1−a_c)² vs Σv_c/(1−a_c²).
- (3) Intervention: inject zero-mean vs biased noise of *equal MSE* into the in_proj outputs.
- **Fails if** the biased/zero-mean ΔNLL ratio is below 2× on slow-decay hybrids, or is not larger than on the dense control.

**Nearest papers.** 2609.30950 (Gramian, uncorrelated noise), 2609.39185, 2607.23390, 2609.04098, 2604.12374, STEPQuant 2609.38169, DAMP 2608.27513. This candidate is about *upstream* bias, not state-quantizer design, which is preempted.

**Queries.**
- `(abs:hybrid AND (abs:Mamba OR abs:SSM OR abs:"linear attention")) AND (abs:quantization OR abs:quantized) AND (abs:"KV cache" OR abs:attention) AND (abs:bias OR abs:accumulat OR abs:sensitivity)` (date)
- `(abs:Nemotron-H OR abs:Jamba OR abs:Zamba OR abs:"Qwen3-Next" OR abs:Falcon-H1 OR abs:hybrid) AND (abs:"post-training quantization" OR abs:PTQ) AND (abs:"unbiased" OR abs:"stochastic rounding" OR abs:"error propagation")` (date) → 0
- adjacent: `(abs:"recurrent" OR abs:"state space") AND (abs:"input bias" OR abs:"DC offset" OR abs:"systematic error") AND (abs:"low-pass" OR abs:integrator OR abs:"slow modes")`
- `ti:quantization AND (ti:hybrid OR ti:Mamba OR ti:"state space")` (date)

---

## 7. Dispatch vs combine in expert-parallel MoE: common-mode vs independent quantization error (MoE serving)

**Hypothesis.** The two communication paths behave differently:
- Dispatch quantization error ε is one per token, shared by all k experts, so it propagates *coherently*: Σ_e g_e J_e ε.
- Combine errors η_e are independent per expert and gate-weighted: Σ_e g_e²σ².

Once combine-side outliers are handled with per-expert-output scales, combine tolerates about ρ⁻¹·k_eff times more noise, where k_eff = (Σg)²/Σg² and ρ is the measured Jacobian coherence. Current practice (FP4/FP8 dispatch with BF16 combine) may have the communication precision inverted.

**Gap evidence.**
- 2603.02731 l.234: "the combine phase retains BF16 accumulation to preserve reduction precision", with no ablation.
- DeepSeek-V3 does FP8 dispatch with BF16 combine. That is from memory and not re-read.
- No paper found quantizes combine or analyzes the asymmetry.
- Adjacent: 2610.03741 does over-the-air (noisy) MoE aggregation in wireless.

**Math core.**
- Proposition (provable, first order): E‖Δy‖²_dispatch = εᵀ(Σg_eJ_e)ᵀ(Σg_eJ_e)ε and E‖Δy‖²_combine = Σg_e²E‖η_e‖². Their ratio is set by ρ = ‖Σg_eJ_e‖_F²/Σg_e²‖J_e‖_F² and by output dynamic range.
- Byte-constrained allocation between the two streams by reverse water-filling.
- The outlier (massive-activation) effect on the combine scale is empirical.

**Decisive experiment (3–5 GPU-days).**
- Models: Qwen3-30B-A3B and DeepSeek-V2-Lite, with emulated EP.
- Arms: dispatch and combine quantized separately to FP8 / NVFP4 / MXFP4, with per-token or per-expert-output scales.
- Measure KL, PPL, GSM8K and MMLU, plus ρ per layer.
- **Fails if** NVFP4 combine with per-expert scales has higher KL than NVFP4 dispatch in most layers, or if ρ does not predict the per-layer winner.

**Nearest papers.** 2603.02731 (FP4 dispatch in training), DeepSeek-V3 report, QUADS 2607.15810 (dual-side alignment is rollout vs training, not dispatch vs combine), 2607.18745 Thm 9.1 (operand bit split for a single matmul; no fan-out coherence).

**Queries.**
- `(abs:"all-to-all" OR abs:dispatch OR abs:"expert parallel") AND (abs:MoE OR abs:"mixture-of-experts") AND (abs:FP8 OR abs:FP4 OR abs:quantiz OR abs:"low-precision communication")` (date)
- `(abs:combine AND abs:dispatch) AND (abs:quantiz OR abs:FP8 OR abs:compress)` (date ≥20250101)
- adjacent: `(abs:"activation compression" OR abs:"communication compression") AND (abs:"tensor parallel" OR abs:"expert parallel" OR abs:"model parallel") AND abs:inference` → 0
- `(abs:MoE OR abs:"mixture-of-experts") AND (abs:"combine" OR abs:"expert outputs" OR abs:"weighted sum") AND (abs:quantiz OR abs:FP8 OR abs:FP4 OR abs:"low precision")` (date)
- `(abs:"expert parallelism" OR abs:"all-to-all") AND (abs:"communication") AND (abs:"FP8" OR ...) AND (abs:inference OR abs:serving OR abs:decoding)` (date) → 0

---

## 8. Evict or quantize? A closed-form head-wise rule from the attention tail exponent (KV eviction × precision)

**Hypothesis.** Fit the sorted attention weights of a head as p_(i) ≈ C·i^{−α}. At byte budget B, the choice between keeping all tokens at low precision and evicting while keeping precision flips at a critical α_c(B/N). α_c has a closed form via the Lambert W function: minimize A·n^{2−2α} + K·4^{−B/n} over n. A head-wise policy predicted from α beats uniform policies at equal bytes, and it explains the regimes reported by "More Tokens, Lower Precision" (2412.12706) and BreadthKV (2610.05685).

**Gap evidence.**
- 2412.12706 is empirical; only its title and abstract were seen this session.
- 2610.05685: an oracle keeping 10% of tokens loses about 13% of mass.
- ThinKV 2510.01290 mixes eviction and quantization by "thought type" heuristics.
- No α-based closed form was found.

**Math core.**
- Eviction distortion ≈ Var_v·τ(n)², where τ(n) = Σ_{i>n} p_(i) ≈ C n^{1−α}/(α−1).
- Quantization distortion ≈ κ4^{−b}Σ_{i≤n}p_i², with nb = B.
- The first-order condition gives n* via Lambert W. Keep-all iff f'(N) ≤ 0.
- Provable under the power-law and independent-error model. Its validity is empirical.

**Decisive experiment (3–5 GPU-days).**
- Models and tasks: Llama-3.1-8B and Qwen3-8B on LongBench, RULER and long-CoT traces.
- Measure α per head; compare uniform KIVI-2/3/4, uniform SnapKV/H2O and the α-predicted head-wise mix at matched bytes.
- **Fails if** the head-wise policy gains less than 1 pt over the best uniform policy, or if the predicted regime matches the per-head empirical optimum for fewer than 70% of heads.

**Nearest papers.** 2412.12706, ThinKV 2510.01290, BreadthKV 2610.05685, MiKV/ZipCache.

**Queries.**
- `(abs:"KV cache") AND (abs:eviction OR abs:"token dropping" OR abs:pruning) AND (abs:quantiz) AND (abs:"trade-off" OR abs:tradeoff) AND (abs:precision OR abs:"bit-width")` (date ≥20250101) → 0
- `(abs:"power law" OR abs:"heavy-tailed" OR abs:"tail exponent") AND (abs:attention) AND (abs:"KV cache") AND (abs:compression OR abs:eviction OR abs:quantiz)` (date) → 0
- adjacent: `(abs:"number of coefficients" OR abs:"sparse approximation") AND (abs:"bits per coefficient" OR abs:"bit allocation") AND (abs:"power law" OR abs:compressible)` → 0
- `(abs:"KV cache") AND (abs:"token budget" OR abs:"number of tokens" OR abs:"eviction") AND (abs:"precision" OR abs:"bit") AND (abs:"head-wise" OR abs:"per-head" OR abs:"attention sparsity" OR abs:"attention distribution")` (date) → ThinKV
- `ti:"More Tokens, Lower Precision" OR ti:"token-precision"`

**Risk.** 2412.12706 may already contain a similar analysis. Read its full text before investing.

---

## 9. Token-axis noise shaping of the V cache: zero-storage error feedback (KV cache storage)

**Hypothesis.** Write V with first-order error feedback along the token axis:

q_i = Q(v_i − u_{i−1}), u_i = q_i − (v_i − u_{i−1})

The attention output error then becomes Σ_i u_i(p_i − p_{i+1}) instead of Σ_i p_i e_i:
- the error power shrinks by the factor R = Σ(Δp)²/Σp²; R ≈ 1/N for diffuse heads, and R ≤ 2 in the worst case (a single spike);
- the stored codes are ordinary b-bit codes, and the carry exists only at write time. This contrasts with recurrent-state ΣΔ, which "doubles the size of a state cache" (2609.39185, l.594–597);
- choosing per head via an offline R statistic should give about ½–1 bit of V-equivalent gain at 2 bits;
- K-side shaping is predicted **not** to help: its logit errors are weighted by p_i(v_i − o), which is not smooth, and the doubled per-element variance increases softmax flattening.

**Gap evidence.**
- Error feedback / ΣΔ has been applied only to recurrent states (2609.39185, 2607.23390, 2609.04490) and to weights (SDQ-LLM 2510.03275).
- KV theory assumes white token-wise noise: AATC 2608.14191, Assumption 1. That is a design choice, not a constraint.
- No KV noise-shaping paper was found.

**Math core.**
- Proposition (provable with subtractive dither, so that the u_i are iid with variance σ²): E‖ε_o‖² = dσ²·Σ_i(p_i − p_{i+1})² (plus boundary terms), vs dσ²Σp_i² without shaping.
- Order-L shaping replaces Δ with Δ^L, and the worst case grows to C(2L, L).
- Stability: first-order error feedback stays in range with one step of headroom.

**Decisive experiment (2–3 GPU-days).**
- Models: Llama-3.1-8B, Qwen3-8B and Mistral-7B.
- Setup: V at INT2/INT3 per-token (KIVI-V), with or without first/second-order token-axis feedback; K fixed at 4 bits.
- Compute R per head from FP attention on calibration data; heads with R > 1 fall back to RTN.
- Metrics: output MSE, PPL, LongBench, GSM8K, RULER.
- **Fails if** the attention-weighted mean R exceeds 0.7 at 4–32K, or if the V2 gain is below 25% of the V2→V3 gap.

**Nearest papers.** 2609.39185 / 2607.23390 (ΣΔ on recurrent states, with the storage objection), SDQ-LLM 2510.03275 (weights), AATC 2608.14191 (white-noise D_V), VC-Attention 2609.15810 (V smoothing by block mean, a different mechanism).

**Queries.**
- `(abs:"KV cache") AND (abs:"noise shaping" OR abs:"error feedback" OR abs:"error diffusion" OR abs:"sigma-delta" OR abs:"delta-sigma")` (date ≥20250101) → 0
- `(abs:"value cache" OR abs:"KV cache") AND abs:quantiz AND (abs:"along the token" OR abs:"token dimension" OR abs:"temporal correlation" OR abs:"adjacent tokens" OR abs:"neighboring tokens")` (date)
- adjacent: `(abs:"error diffusion" OR abs:"noise shaping") AND (abs:attention OR abs:"weighted average" OR abs:"low-pass") AND (abs:quantization)`
- `(abs:"sigma-delta" OR abs:"error feedback") AND (abs:quantization) AND (abs:LLM OR abs:transformer OR abs:"language model") AND (abs:activation OR abs:cache OR abs:inference)` (date)
- `(abs:"attention probabilities" OR ...) AND (abs:"stochastic rounding" OR abs:"error diffusion" OR abs:"error feedback" OR abs:dither) AND (abs:quantiz OR abs:FP4 OR abs:FP8)` (date) → 0

**Risk.** LLM attention is lumpy (sinks, punctuation), so R may be near 1 for many heads. The R statistic is cheap to measure first, on CPU, from saved attention maps.

---

## 10. Why route-flip harm is at chance: a tie-indifference theorem for quantized MoE serving

**Hypothesis.** Quantization-induced route flips happen at near-ties of an approximately loss-calibrated router. Three consequences follow:
- the first-order loss change of a flip has conditional mean ≈ 0, so its sign is unpredictable from router statistics;
- the net route-mediated damage is second order: ∝ Σ_flips g_tie²‖E_k(x) − E_{k+1}(x)‖²_H;
- gate-continuous blending near ties removes the damage, while pinning the FP route cannot recover more than this second-order term.

**Gap evidence.** 2608.11212 reports on OLMoE (top-8, norm_topk_prob = False, 4-bit KV):
- RMF ≈ 0.31;
- flip detection AUC 0.772;
- harm-given-flip AUC 0.490, with P(harmful | flip) = 0.572;
- normalization convention acts as a "damage-magnitude moderator".

It explicitly calls the barrier "an empirical barrier, not a theorem". Separately, 2606.19036 analyzes discontinuity geometry but not loss sign.

**Math core.**
- Proposition: assume near ties that r_k − r_{k+1} = −c·E[L(k) − L(k+1) | x] + o(m). Then E[ΔL | flip at margin m] = O(m) + ½g_tie²ΔEᵀHΔE.
- The calibration assumption is empirical and must be measured, so this is half theorem, half finding.

**Decisive experiment (3–5 GPU-days).**
- Models: OLMoE-1B-7B (unnormalized), Qwen1.5-MoE-A2.7B / Qwen3-30B-A3B (normalized), DeepSeek-V2-Lite.
- Noise: 4-bit KV and W4A8.
- Per flip, compute teacher-forced ΔL, g_tie and the JVP-based ‖ΔE‖_H; also test a soft-blend fix.
- **Fails if** the second-order term explains R² < 0.3 of the mean flip damage, or the soft blend recovers less than 50% of RMF.

**Nearest papers.** 2608.11212 (empirical barrier), SmoothSMoE 2606.19036 (the smoothing fix already exists), VSRAQ 2606.05688 (routing-consistent PTQ). The novelty is only the explanation and predictions, so this is moderate-low.

**Queries.**
- `(abs:"mixture-of-experts" OR abs:MoE) AND (abs:quantization OR abs:quantized) AND (abs:"routing flip" OR abs:"router flip" OR abs:"expert selection" OR abs:"routing consistency" OR abs:"routing shift")` (date)
- `(abs:MoE) AND abs:router AND abs:margin AND (abs:quantization OR abs:perturbation OR abs:noise)` (date)
- adjacent: `(abs:"top-k" AND abs:"discontinu") AND (abs:gating OR abs:routing) AND (abs:smooth OR abs:noise)`
- `ti:2608.11212 OR (abs:OLMoE AND abs:"KV cache" AND abs:routing)`

---

## Screened out this session (preempted or thin)

| Idea | Killed by |
|---|---|
| Bayes blend of top-k and k+1 experts at quantization-noisy route margins | SmoothSMoE 2606.19036 (ε-strip smoothing with an extra expert) |
| FP4/FP8 P: normalize by Σ Q(P) and restore the V mean | HiFA4 2607.04302 (P-Reordering: normalizer from quantized P; "every tile exhibits net probability-mass loss"), VC-Attention 2609.15810 (V-Smooth plus Prop. 3.1) |
| Cross-request leakage and interference via batch-shared dynamic FP8 scales | Quantamination 2604.26505; also 2609.04748 |
| Quantized SSM state snapshots for prefix caching | DASC 2608.30386 (decay-aware state checkpoint compression) |
| Deferred, windowed state write-back | LeapQuant 2609.38166 (per-window quantization with high-precision buffered updates); 2609.30950 (chunked write-back) |
| Retraining EAGLE/MTP drafters for quantized targets | 2607.04244 (two-stage drafter training BF16 → INT4 target) |
| Low-precision prefill, high-precision decode | Mix-Quant 2605.20315; 2610.07443 |
| Occupancy- or gate-aware any-precision expert reads | SliceMoE 2512.12990, DynaExq 2511.15015, DyMoE 2603.19172 |
| "Acceptance is first-order (TV), PPL second-order" law; acceptance rate as a fidelity monitor | 2610.00694, 2605.02404 (EAR); runtime certification 2605.20868, WitCert 2607.28699 |
| Token-axis shaping of P in FP4 PV | Helps only for smooth V (video), and is 2× worse for uncorrelated text V; mass is already handled by HiFA4 |
| Interaction-aware cross-layer KV allocation | First-order errors from independent noise are additive in expectation, so the premise is weak; CoopQ 2509.15455 covers the game-theoretic weight version |
| Stale KV after in-flight RL weight updates absorbed by the FP8 cell | CacheReforge 2609.30884; quantization-cell theorem 2609.21834 (already preempted in the report) |
