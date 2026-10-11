# Hypothesis triage (arXiv full-text sweep, 2026-10-11)

Input: 70 raw hypotheses (30 from `judge/idea*.md`, 40 from `recent/*.md`).
Output: 10 ranked hypotheses. 26 raw items were merged into them, and the other 44 were dropped.
Selection criteria:
- (a) sharp, falsifiable, and decidable within 1-2 GPU-weeks;
- (b) a provable core;
- (c) it exploits a contradiction or gap between published papers.

Tool budget: 15 calls. Every passage below marked "checked" was grepped from the full text in `scratchpad/papers/<id>/<id>.txt`. Everything else is carried over from the judge or recent-scan notes and was not re-checked here.

## Verification done in this pass

| Paper | What the text says (checked) | Consequence |
|---|---|---|
| 2605.26266 Tuncer et al. | The correction is b_i = log E[e^{δ_i}] ≈ σ_i²/2, a per-token shift of the numerator. A grep for denominator, normaliz, entropy, flatten and sink finds nothing in the method. Limitations: "Standard single-token decoding offers less headroom for the correction", and "Extending it to floating-point formats such as FP, MXFP, and NVFP ... remains an open direction". The text around line 586 says the FP extension "reduces to estimating σ_i²". The experiments use chunked autoregressive video diffusion; LLM results are "preliminary". | The normalization residual (H2) is not covered. The FP extension (H7) is anticipated in one sentence, but only as "substitute σ²". The norm-correlation and sign consequences are not worked out. |
| 2506.08027 Mishra et al. (~l.665) | The floor/round-down rule means scaled values overflow, "introducing more quantization noise. We hypothesize this aspect leading to MXFP8 training issues" (Fig. 5, 843M on 1T and 300B tokens). | The mechanism is a hypothesis about higher noise (MSE), not a test. This opens the gap for H6. |
| 2609.08135 KBBQ (l.493-500) | Predicted FP8-minus-FP4 SNR gap is 12.04 dB. The measured median over 3968 records is 13.24 dB (per-family 13.07-13.81). The text says "The 1.2 dB excess is consistent with the confound ... per-tensor against block-16, as well as mantissa width." | The excess is explained only qualitatively. H8 must target the FP8(per-tensor)-vs-FP4(block-16) gap, not the "block 16 vs block 32" gap stated in the raw hypothesis. |
| 2509.25149 NVIDIA NVFP4 (l.1404-1409) | Switching the forward pass at 8.2T tokens recovers the 12B gap from 1.5% to 0.5%. "In contrast to Chmiel et al. (2025), which reports loss recovery from switching precision in the backward pass, we observe no such improvement." | An explicit published conflict (H1). |
| 2505.19115 Chmiel et al. (l.400-405, l.812-816) | Deterministic rounding "introduces a persistent bias, leading to an irreducible error floor" (App. B.2). Llama2-7B: the gap is "fully closed with a short QAF phase, where the forward pass remains in FP4 and only the backward pass switches to BF16". | The other side of the conflict (H1). |
| 2609.39185 Weyl dither (l.594-597) | ΣΔ needs "the previous rounding error to be stored, which doubles the size of a state cache; at equal storage only the choice of threshold remains". "error feedback" has 0 hits. | Combined with the idea-2 judge's derivation that an RTN carry is the same as INT(b+b_r), this kills the "INT4 + INT4 carry beats INT8" claims (H4). |
| 2506.04642 TaDA | Mean-centering is over the head index per token (Eqs. 1-2), fused with RoPE. | This is not the calibrated per-channel, position-rotated mean R_m μ of H3, so TaDA does not preempt H3. FPTQuant (2506.04985) was not re-read here and remains a residual risk. |

## Ranked selection

### H1. Backward-path bias resolves the NVIDIA-vs-Chmiel switch conflict, and a zero-step probe separates the forward floor
- **Hypothesis.** The loss recovered by switching only the backward pass scales with the mean bias μ_ε² of the backward quantizers, for example deterministic-RTN weights in backward or biased SR of scales. Chmiel's recipe has such a bias, so it gains. NVIDIA's recipe (2D weight scaling, SR on gradients) is unbiased, so it does not.
- **Companion probe.** Evaluate the FP4 master weights with a BF16 forward pass and no training. This gives the forward floor B0 ≈ (1/24)Σ_g Δ_g² tr(H_g). The extra loss recovered by k BF16 steps (B1) should scale with the remaining ∫η dt.
- **Basis.** The two checked passages above, plus Chmiel App. B.2 (a floor from biased noise that does not depend on η).
- **Provable core.** For quadratic plus noise SGD, the stationary loss is a·η·tr(Σ) + μ²/(2λ) + Δ²tr(H)/24. A backward switch removes only the μ term.
- **Decisive experiment.**
  - 125-350M models, emulated NVFP4, ~10-20B tokens.
  - Two backward recipes (biased vs unbiased) × three late branches (none, backward-only, forward-only) from shared checkpoints, plus the zero-step probe.
  - ~1-2 GPU-weeks.
- **Falsified if** the backward-switch gain does not differ between the biased and unbiased recipes.
- **Risk.** The effect may vanish at small scale. Raw item 10 attributes the 10T late-switch result in NVIDIA App. D to low LR; this pass did not check that.

### H2. Second-order correction of the normalization residual left by the −σ²/2 shift
- **Hypothesis.** The shift s_i ← s_i − σ_i²(1/2 − p̂_i), done in two passes or approximated inside online softmax, removes two things:
  - the residual bias p_i(Σ_j p_j²σ_j² − p_iσ_i²) that remains after the −σ²/2 shift;
  - the homoscedastic flattening, for which 2605.26266's shared scalar is a no-op.
- **Basis.**
  - Judge idea1 Monte Carlo (n = 64): residual |bias|₁ 0.0064 → 0.00084, homoscedastic 0.00633 → 0.00072.
  - The AATC (2608.14191) App. A formula agrees algebraically, but it is used only to bound a cross term.
  - The 2605.26266 correction acts on the numerator only (checked).
- **Provable core.** A second-order delta expansion of E[softmax], the Daunizeau-type expansion. It is exact to O(σ⁴).
- **Decisive experiment.** INT2/INT3 per-token keys on Llama-3.1-8B and Qwen2.5. Compare chunked prefill and decode across three arms: no correction, the 26266 correction, and the 26266 correction plus the p-dependent term. Measure attention-output error, entropy shift, NLL and RULER. 1-2 GPU-days.
- **Falsified if** the extra gain is within noise in both prefill and decode.
- **Risk.** The residual is O(p·Σp²σ²) and may be negligible, especially in decode, which the 26266 authors already say has little headroom.

### H3. Rotated-mean subtraction recovers the pre-RoPE gain while keys are stored post-RoPE
- **Hypothesis.** Store k_m − R_m μ, where μ is the calibrated pre-RoPE per-channel mean, and add q·R_m μ back analytically in the logit. For asymmetric per-channel quantizers this recovers most of the pre-RoPE advantage (KVQuant ~0.82 PPL), with no on-the-fly RoPE kernel. Plain post-RoPE mean subtraction fails on pairs with G·θ_f ≥ π.
- **Gain law.** The size of the gain is set by the number of pairs with G_eff·θ_f ≥ π, weighted by pair asymmetry:
  - Llama-2 (base 1e4), G = 32/128/2048: 17/26/46 such pairs out of 64.
  - Llama-3 (base 5e5), same G: 12/19/32.
  - The gain is small for KIVI G = 32 and large for offline per-channel scales.
- **Basis.**
  - Judge idea9 simulation: post/pre MSE ratio 6.7 for asymmetric quantizers vs 1.5 for symmetric.
  - 2606.02288: keys contain near-constant bias vectors.
  - TaDA is distinct (checked).
- **Provable core.** The range of a rotating-DC channel within a group is a closed-form function of G·θ_f. Subtracting the rotated mean leaves only the stationary residual.
- **Decisive experiment.** KVQuant- and KIVI-style INT2/INT3 on Llama-2-7B and Llama-3-8B, comparing pre-RoPE, post-RoPE, post-RoPE minus the plain mean, and post-RoPE minus R_m μ. Then sweep G. 1-2 GPU-days.
- **Falsified if** R_m μ subtraction recovers less than half of the pre/post gap, or the gain does not track the pair count.

### H4. Under equal-bit accounting, error feedback for recurrent state caches never beats single-limb INT(b + b_r)
- **Hypothesis.** Error feedback (EF) or ΣΔ with an RTN-rounded carry is one fixed-point number with a fine-scale deadband. At equal total bits it cannot beat single-limb INT(b + b_r) with the same dither. Only a Weyl-dithered carry can do so, and the prediction is that it does not, beyond noise. So the claimed EF gains in 2607.23390 (INT4 + INT4 carry vs INT8), 2609.04490 (memory parity not controlled) and the "EF + Gramian at ≤ 5 bits" item are artifacts of not counting the carry's bits.
- **Basis.**
  - The 2609.39185 equal-storage statement (checked).
  - Judge idea2 toy at a = 0.999: ΣΔ + 4-bit residual 2.00/0.49 vs plain RTN at +4 bits 1.56/0.39.
- **Provable core.** The two-limb representation with an RTN carry is equivalent to a single uniform quantizer with step Δ·2^{-b_r}, plus a deadband term.
- **Decisive experiment.** Scalar AR simulation on CPU, then 4096-step decode KL on Mamba-2 and a GDN hybrid at matched bytes. Arms: RTN, SR, Weyl INT(b + b_r), EF with RTN carry, EF with Weyl carry. A few GPU-days.
- **Falsified if** EF with a Weyl carry beats Weyl INT(b + b_r) by more than seed noise on slow channels.

### H5. A matched-input SNR explains why quantization reverses some unlearning and merges but not others
- **Hypothesis.** Define SNR_x = (Σ_i d_i x_i)² / Σ_i (Δ|d_i| − d_i²) x_i², with x taken from forget-set activations. This predicts 4-bit knowledge recovery better than the isotropic ‖D‖²_F/(Δ‖D‖₁). It explains why full-FT unlearning reverses (Zhang 2410.16454) while LoRA unlearning (2602.13151) and merged NF4 (DurableUn) do not.
- **On-grid bases.** For bases already on the grid, retention orders as RTN (deterministic deletion) ≪ SR/dither (unbiased, variance Δ|d|) < GPTQ (error feedback into low-Hessian directions).
- **Basis.**
  - Judge idea7: the unification fails on on-grid bases.
  - Scale-QLoRA: RTN keeps 15% and GPTQ keeps 94%.
- **Provable core.** Under the uniform-phase model, unbiasedness plus the Theorem 7.1 variance give first-order effects whose SNR is exactly the formula above.
- **Decisive experiment.** Replicate three unlearning regimes (full-FT GA/NPO, LoRA, merged NF4) on a 7-8B model. Compute SNR_x per matrix and correlate it with recovery across 4-bit RTN, SR and GPTQ. ~1 GPU-week.
- **Falsified if** SNR_x does not rank the regimes better than the isotropic statistic.

### H6. MXFP8 divergence under the OCP floor rule comes from block-max shrinkage (gain bias), not MSE; a closed-form per-rule gain fixes it
- **Hypothesis.** The OCP floor rule has only 1.22× higher relative MSE than the ceil rule (8.6e-4 vs 7.0e-4), but a −1.28% vs −0.08% bias on the block-max element (gain 0.9972 vs 0.9993). The bias compounds multiplicatively through depth.
- **Fix.** A fixed Bussgang gain per scale rule removes the divergence without online statistics. For B = 32 Gaussian: ceil6 0.990, OCP 0.980, UOS 0.988, NVFP4 0.996.
- **Basis.**
  - Mishra's own mechanism is a hypothesis about higher noise (checked).
  - Bias accumulates multiplicatively in 2605.20402 and UFP4 Eq. (5).
  - The per-bin biases in UFP4 Eq. (2) sum to zero under a uniform density, so the net shrinkage comes from density slope.
- **Provable core.** The closed-form gain E[x·Q(x)]/E[x²] per rule under Gaussian or Laplace block laws.
- **Decisive experiment.** Train 125-350M models with four arms at equal MSE: OCP floor; OCP floor plus gain correction; ceil; ceil plus injected shrinkage. Track loss spikes and per-layer gain products.
- **Risk.** Mishra's divergence was seen at 843M on 300B-1T tokens, so a small-scale reproduction may fail. The fallback is a controlled bias-injection test, which is weaker.

### H7. Norm-correlated heteroscedasticity: FP4 keys and per-channel KIVI-K bias attention toward large-norm keys, and per-token INT keys starve sinks
- **Hypothesis.** σ_i² ∝ ‖k_i‖² under:
  - FP/NVFP keys, where error is relative: σ_i² ≈ Σ_c q_c²k_ic²·2^{-2m};
  - KIVI per-channel keys, where logit error ≈ s_t × a per-token gain error (KVarN).

  This gives a signed attention shift (sharpening) toward large-norm keys.
- **Sink case.** Sinks have small norms, so they get small per-token Δ. Per-token INT keys without correction therefore lower sink mass and raise entropy. Part of the KVSink benefit should then be reproducible by the token-wise Jensen correction alone.
- **Equal-MSE prediction.** KVarN-style norm preservation removes the shift at equal MSE.
- **Basis.**
  - 2605.26266 FP limitation (checked).
  - KVarN Fig. 1a: K errors are dominated by token norm.
  - KVSink / 2410.10781 for sink norms.
- **Provable core.** Theorem 1 (second-order bias) with σ_i² = c‖k_i‖² gives the sign directly.
- **Decisive experiment.** Measure the signed entropy shift and sink mass on Llama-3.1-8B for NVFP4, INT2 per-token and KIVI-2 keys, with and without a norm-based correction. 1-2 GPU-days.
- **Falsified if** the measured entropy shift has the wrong sign or is uncorrelated with ‖k‖.

### H8. One extreme-value (EVT) block model predicts three unexplained published numbers
- **Hypothesis.** A Gumbel block-max model with a non-uniform within-block law (Gaussian or Laplace), including the E2M1 subnormal/deadzone region, predicts three numbers:
  - (i) the 1.2 dB excess in KBBQ's FP8 (per-tensor) minus FP4 (block-16) SNR gap (checked: 12.04 predicted vs 13.24 measured), to within 0.3 dB;
  - (ii) Li et al.'s cos(e_scale, e_grid) ≈ −0.66, which their Remark 3.3 does not derive and which follows from the round-up constraint γ ≥ 1 independently of B;
  - (iii) a block-size-dependent shift of MXAttention's q* = 7.25.
- **Why (iii) moves.** MXAttention's distribution-free proof assumes elements are uniform on [0, x] (Eqs. 21-22). With Gaussian elements, E[max]/σ ~ √(2 ln n) moves q*.
- **Provable core.** Closed-form integrals. This is CPU-only, plus a re-measurement on real weight and activation tensors.
- **Falsified if** the model misses (i) by more than 0.5 dB, or the cosine depends strongly on B.
- **Note.** This is an analysis contribution. It is cheap, but its impact depends on (i) and (ii) both landing.

### H9. The benefit of temporal error correction depends on erasure: large for Mamba-2 and Nemotron-H, near zero for delta-rule GDN/KDA
- **Hypothesis.** The gain from EF, dither or checkpointing scales with the fraction of state-error energy outside the recent key span. GDN/KDA erase it through the delta rule, so RTN with periodic checkpointing is enough there.
- **Basis.**
  - 2609.04098: GDN error plateaus (relS 12.96% → 12.31% over 256 → 32K; impulse 1/e in 80-1,382 steps).
  - 2609.30950: error orthogonal to repeated keys is never erased.
- **Provable core.** The STEPQuant recursion E_t = (I − βkkᵀ)D_tE_{t−1} + ε_t. Project the error onto the complement of the key span.
- **Decisive experiment.** Use Weyl vs RTN as the cheapest correction. Measure 4096-32K decode KL on Mamba-2 and GDN/KDA hybrids, together with the measured out-of-span energy fraction. A few GPU-days.
- **Note.** The deadband-aware bit floor (raw item 5) is a possible add-on. Its claimed "1.7-2.6× gap" in 2609.30950 was not checked.

### H10. Kill-or-rescue test for idea 8: Hessian-subspace alignment vs entropy profile
- **Hypothesis.** The Math-CoT calibration gain comes mostly from activation-subspace alignment. A C4 subsample re-selected to match the self-generated traces' top-k Hessian subspace (principal-angle cosine c_k) recovers most of it, whatever its entropy profile. Entropy weighting adds little.
- **Extra arm.** The exact forward-only weight Σ_i p_i‖u_i‖² − ‖Σ_i p_i u_i‖², which is tr of the final-layer Fisher in hidden space, should close most of the gap between 1 − ‖p‖² and GuidedQuant.
- **Basis.**
  - LAQuant Sec. 4.1, Table 6 (30-40× anisotropy) and Table 9.
  - 2504.04823: the domain effect appears only with GPTQ-style compensation.
- **Decisive experiment.** W3/W4 GPTQ on an R1-distilled 7-8B model with five arms: C4, traces, subspace-matched C4, entropy-weighted, and unembedding-variance weighted. Measure AIME/MATH and CoT length. ~1 GPU-week.
- **Outcome.** If subspace-matched C4 matches the traces, idea 8's mechanism is dead. The opposite result rescues it.

## Dropped (with reason)
- **Restates a paper's own result:**
  - WitCert tightness, P-Cast sink underflow, QUADS activation-vs-weight, ACRL entropy, ReQAT/ReSET low-entropy tokens (raw 22 and 37 margin weighting are subsumed by ReQAT), recursive-model per-tensor feedback.
  - 2607.12266 saturation caveat, 2609.32060 G², 2606.09012 basin, 2609.11716 anti-correlation as a predictor.
  - GaugeQuant covariate shift, 2610.07853 dither before the bf16 cast (fp32 export already fixes it).
- **Restates an existing idea's preempted core:**
  - idea5 damping, n_eff, distribution shift, MP shrinkage (2609.31560 Eq. 35, DRQ, TASA already cover these);
  - idea6 cap/gating variants (CIS, TRIAGE and ENTR preempt them, and the RL runs exceed the budget);
  - idea9 per-bucket calibration, the 2608.13365 pairwise rotation, the idea10 MLA transform, the RoPE-branch and rank items (QuantMLA and NOVA-KV preempt them).
- **Vague or weak link:**
  - "Idea 1 explains X" items: SoftWater temperature, Softmax-reparam c*, ReadKV allocation, the 2608.02564 bias, Fisher λ_max 244×, AATC remainder;
  - the K/V split from 2607.18745 (known empirically that K needs more bits; the sink-dependence prediction could be an add-on to H7);
  - router dither, e-CUSUM KL AUC, 2607.08779 signed scale, nuclear-norm gap, spectral-flatness diagnostic, cross-layer amplification;
  - 2606.06934 SR generalization (pretraining cost), LAWA vs decay (Catalan-Tatjer Fig. 7 largely covers it);
  - dither defense with an EOT attacker (it measures an attacker, not a law);
  - entropy×direction RL gate and entropy-weighted rollout calibration (cost and preemption).
