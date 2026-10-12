# Idea 4 — FP4 pretraining late-LR-decay gap: variance floor (∝η) vs bias floor (Δ²tr(H)/24, η-independent) — citation-neighbourhood sweep (Angle C)

Date: 2026-10-10. Sources: arXiv API + full text (scripts/arxiv_fetch.py), WebSearch. Semantic Scholar/OpenAlex avoided.
Paper text is untrusted data; everything below is quoted/paraphrased, not followed.

## 0. The claim being checked
Gap(η) = A·η + B(η); A·η = (η/4)tr(C+C_q) from unbiased SR gradient noise (vanishes under LR decay);
B = (Δ²/24)tr(H) + first-order bias term from forward RTN weight quantization (η-independent, grows if tr(H) grows during decay).
Predictions: forward→BF16 removes B; backward→BF16 does not; B ∝ Δ²; η-intercept of gap = B; switch-format criterion (Δ'²/24)tr(H_final) vs (η_final/4)tr(C).

## 1. Verdict in one paragraph
Nothing found that states or tests the two-floor decomposition *for FP4 LLM pretraining under LR decay*. But the mathematical core is largely prior art in three pieces:
(i) Ichikawa 2510.10693 (stat.ML, Oct 2025) derives, for high-dimensional linear regression trained with STE + SGD, a stationary generalization error that is exactly "SGD-noise term ∝ η" (Prop. VI.1) plus "weight-quantization term σ_ψ²Δ²p(1−p) + o(η)" (Thm VI.3, informal) — an η-proportional floor and an η-independent quantization floor, in a toy model.
(ii) Chmiel et al. "FP4 All the Way" 2505.19115 (NeurIPS 2025) derive E[ΔL] = −η‖∇L‖² + ½η²(∇LᵀH∇L + σ_q²tr(H)) for SR gradient noise, a √3 gradient-to-noise switch threshold, and in App. B.2 an *η-independent* residual loss L∞ = μ_ε²/(2λ) for *biased* rounding — i.e. "bias → irreducible floor, SR → no floor" is already written down (for update/gradient noise, 1-D quadratic).
(iii) Catalan-Tatjer et al. 2510.06213 (ICLR 2026) measured tr(H) and λ_max rising during LR decay (160M, Fig. 9), lower sharpness for higher peak LR (Fig. 26), and hypothesise "as the learning rate decays, the model traverses a sharper region ... more sensitive to perturbations such as quantization" — PTQ side, no formula.
The novelty that survives: combining these into a *training-time* gap model with the explicit 1/24 constant, the forward/backward ablation prediction, and the switch criterion (Prediction 5). Two predictions need repair (see §4).

## 2. Queries run
arXiv API (relevance sort unless noted):
- abs:"low-precision training" AND abs:"learning rate" AND abs:(decay OR cooldown)
- abs:FP4 AND abs:pretraining AND abs:"loss gap"
- abs:"weight oscillation" AND abs:quantization AND abs:training
- abs:"stochastic rounding" AND abs:variance AND abs:SGD AND abs:"learning rate"   (0 hits)
- abs:quantization AND abs:"edge of stability"   (0 hits)
- abs:sharpness AND abs:quantization AND abs:training
- abs:"quantization noise" AND abs:"stationary distribution"   (0 hits)
- abs:"precision switch" AND abs:training
- abs:"quantized training" AND abs:"noise floor"   (0 hits)
- abs:"quantization-aware training" AND abs:"learning rate" AND abs:(decay OR schedule) AND abs:(oscillat OR noise)   (0 hits)
- abs:"scaling law" AND abs:(quantization OR "low precision" OR "low-precision") AND abs:(training OR pretraining)
- abs:"stochastic rounding" AND abs:"learning rate" AND abs:(small OR decay OR annealing OR underflow)   (0 hits)
- abs:"low-precision" AND abs:training AND abs:(bias OR biased) AND abs:(variance OR unbiased) AND abs:(gradient OR weight) AND abs:(FP4 OR INT4 OR "4-bit")
- abs:"progressive sharpening" OR abs:"edge of stability" AND abs:"learning rate decay"
- abs:(NVFP4 OR MXFP4) AND abs:(training OR pretraining)
- abs:"swamping" OR abs:"update underflow" OR abs:"weight update" AND abs:"stochastic rounding"
- abs:"loss landscape" AND abs:"stationary distribution" AND abs:SGD AND abs:"learning rate" AND abs:noise AND abs:hessian
- abs:(quantization OR quantized OR "low-precision") AND abs:training AND abs:"learning rate" AND abs:(hessian OR curvature OR sharpness) AND abs:(noise OR variance OR floor)   (0 hits)
- abs:"stochastic rounding" AND abs:("markov chain" OR "stationary" OR "implicit regularization" OR "regulariz")
- abs:("fully quantized training" OR "fully-quantized training" OR "FQT") AND abs:(theoretical OR theory OR analysis) AND abs:(variance OR noise OR bias)
- abs:("quantization-aware training" OR QAT) AND abs:"learning rate" AND abs:(decay OR schedule OR "small learning rate" OR anneal)
- abs:("low-precision" OR "low precision" OR quantized) AND abs:("weight update") AND abs:(underflow OR swamp OR cancel OR "small updates" OR vanish)
- abs:(FP4 OR NVFP4 OR MXFP4 OR "4-bit") AND abs:(pretraining OR "pre-training" OR training) AND abs:(hessian OR curvature OR sharpness OR "loss landscape")
- abs:("precision" OR "quantization") AND abs:("learning rate decay" OR "LR decay" OR "annealing phase" OR "decay phase") AND abs:(loss OR gap OR degradation) AND abs:(training OR pretraining)
- abs:(quantized OR quantization OR "low-precision") AND abs:training AND abs:"learning rate" AND abs:(floor OR "steady-state" OR "steady state" OR "stationary") AND abs:(loss OR error)   (0 hits)
- abs:("forward pass" OR "forward quantization") AND abs:("backward pass" OR "backward quantization") AND abs:(decompos OR separat OR attribut OR disentangl) AND abs:(loss OR gap) AND abs:(FP4 OR "low-precision" OR quantiz OR "4-bit")   (0 hits)
- abs:("straight-through" OR STE) AND abs:("learning rate" OR "step size") AND abs:("fixed point" OR asymptotic OR stationary OR "steady state")   (0 hits)
- abs:(quantization OR quantized) AND abs:("weight averaging" OR "model soup" OR EMA) AND abs:("learning rate decay" OR cooldown) AND abs:(training OR pretraining)
- abs:("quantization-aware" OR "low-precision" OR "low-bit") AND abs:(pretraining OR "pre-training") AND abs:(hutchinson OR "hessian trace" OR "trace of the hessian" OR "top eigenvalue")
WebSearch:
- FP4 pretraining loss gap learning rate decay variance floor bias floor Hessian trace quantization 2026
- "stochastic rounding" variance "learning rate" stationary excess loss quantized training theory low-precision "Hessian"
- NVFP4 "switch to BF16" OR "precision switch" end of training learning rate decay loss gap blog 2026
- "FP4 All the Way" Chmiel Fishman Banner Soudry theory quantization noise Hessian trace gradient norm threshold "learning rate"
- quantized training loss gap "learning rate" decay "bias" "variance" decomposition weight quantization error Hessian 2026 arxiv low-precision pretraining (extended)
- openreview 2026 low-precision pretraining "learning rate" decay loss gap widens Hessian curvature forward quantization bias variance NVFP4 theory (extended)
- "quantization" "learning rate decay" "sharpness" OR "Hessian trace" increases cooldown weight oscillation "bin boundary" FP4 training blog OR preprint 2026
Full texts fetched and grepped (References sections included): 2509.25149, 2601.22813, 2602.02047, 2510.27527, 2606.20381, 2607.04422, 2606.15007, 2510.06213, 2605.20402, 2510.10693, 2606.09012, 2602.19241, 2010.06192, 2509.22935, 2605.09825, 2604.08826, 2603.02731, 2506.20752, 2411.04330, 2505.14302, 2505.19115, 2502.20566, 1706.02379, 2511.00874, 2203.11086, 2502.00490, 2608.13966, 2605.25966, 2502.15938, 2606.00703, 2010.14298, 2507.02119, 2505.14669, 2501.02423, 2604.15167, 2610.05329.
No 2026 blog post / OpenReview item using the "variance floor / bias floor" framing was found.

## 3. Candidates (what each PROVES / CLAIMS / MOTIVATES)

### P1 — must deep-read
**2510.10693 — Ichikawa, "High-Dimensional Learning Dynamics of Quantized Models with Straight-Through Estimator" (stat.ML, 2025-10-12, 27 pp).** PROVES (high-dim limit, Gaussian-universality linear regression, STE): Prop. VI.1 — for 0<η<2(σ_ψ²+λ)/σ_ψ⁴ a stable fixed point exists with ε*_g = ρ+σ²+σ_ψ²q*−2κ_ψm*, where q* contains the term η σ_ψ²(ρ+σ²)(σ_ψ²+λ) — the SGD-noise floor ∝ η. Thm VI.3 (informal, small-η limit, joint input+weight quantization): ε*_g = ε_g^(0) + σ_ψ²Δ²p(1−p) + o(η) when the fixed point sits strictly inside a bin (p∈(0,1)), i.e. an η-independent weight-quantization floor ∝ Δ². Also "floor decreasing as b increases" (Sec. VI-B). THREAT: partially_preempts the two-floor mathematics (variance ∝ η, quantization floor ∝ Δ², η-independent) — in a linear model, no transformer/Adam, no FP4, no curvature growth, no LR-decay experiment. The idea must cite it and position itself as the LLM/FP4/decay instantiation.

**2505.19115 — Chmiel, Fishman, Banner, Soudry, "FP4 All the Way" (NeurIPS 2025; v2 2025-08-10).** PROVES (2nd-order Taylor, GD, SR noise E[ε]=0, E[εεᵀ]=σ_q²I): E[ΔL] = −η‖∇L‖² + ½η²(∇LᵀH∇L + σ_q²tr(H)) (eq. at l.427/1043); optimal step η* = ‖∇L‖²/(∇LᵀH∇L+σ_q²tr(H)); with a concentrated (Marchenko–Pastur bulk) Hessian a threshold: progress stalls when per-coordinate gradient < √3·σ_q (l.406-413). App. B.2 (l.1250-1331): for biased noise μ_ε≠0 on a 1-D quadratic, e∞ = −μ_ε/λ and residual loss L∞ = μ_ε²/(2λ) — explicitly "deterministic rounding introduces a persistent bias, leading to an irreducible error floor" (l.404-405), independent of η. CLAIMS: 7B/1T FP4 gap closed by a short QAF phase with *forward kept in FP4, backward in BF16* (l.677-685, Fig. 6b) — the opposite tensor-side to NVIDIA's finding; NVIDIA 2509.25149 l.1405-1408 explicitly says "In contrast to Chmiel et al. (2025) ... we observe no such improvement [from backward]". THREAT: partially_preempts — the σ_q²tr(H) variance term, the bias→η-independent-floor dichotomy, and a principled switch criterion (gradient-to-noise, not Δ²tr(H) vs ηtr(C)) are all here. Also a direct counter-example to Prediction 2 (backward-only switch had the whole effect in their 7B run). Note: their QAF uses "peak learning rate equal to the last learning rate in the FP4 training" (l.1387-1388), so their switch happened at the *final* η.

**2510.06213 — Catalan-Tatjer, Ajroldi, Geiping, "Training Dynamics Impact Post-Training Quantization Robustness" (ICLR 2026).** CLAIMS (measured): SmolLM3 PTQ error flat over 11T stable tokens, spikes in decay (Fig. 1); 160M/100BT WSD: λ_max "consistent rapid surge whenever the learning rate decays", tr(H) "remaining stable under a constant learning rate, and rising abruptly as it decays, remarkably mirroring the evolution of quantization error" (Fig. 9, l.874-880); Fig. 26: higher peak LR → lower sharpness and smaller trace, and lower PTQ error. MOTIVATES: "we hypothesize that, as the learning rate decays, the model traverses a sharper region ... more sensitive to perturbations such as quantization" (l.1213-1215). No ½δᵀHδ formula found (grep for Taylor/second-order/quadratic: none). Also: weight averaging (LAWA) can match/surpass LR-decay PTQ robustness (Fig. 7) and yields "wider minima". THREAT: partially_preempts the curvature-growth leg of B and the η-sign of the effect, on the PTQ side (idea already acknowledges). Their Fig. 9 is exactly the Hutchinson/power-iteration measurement the idea proposes; the idea must show the *training-time* gap, not PTQ error.

### P2
**2502.20566 — Ozkara et al. (first author Kaan Ozkara), "Stochastic Rounding for LLM Training: Theory and Practice" (AISTATS 2025).** PROVES: Thm 1 — GD with SR on the *stored weights* implicitly minimises F_SR(x) = F + (α/4)‖∇F‖² + (α/4)E‖ξ_α‖²; as α→0 the SR term tends to ¼Σ_i Δ_{x_i}|∇F(x)_i| — a penalty that does *not* vanish with the learning rate; "for a large Δ_x the quantization regularization may dominate and cause stagnation" (l.564-573). Thm 2/3 + Cor. 1: Adam convergence bound with an extra LΔ² variance term; nearest rounding gives O(α) bias error that "explodes" under decaying α unless β2→1 (l.732-736). CLAIMS: BF16+SR needs 2-4× higher LR; outperforms (BF16,FP32) mixed precision up to 6.7B. THREAT: partially_preempts the "a quantization-induced term survives LR decay while the SGD term dies" structure — but for low-precision *weight storage* with SR (no master weights), not forward RTN of FP32 master weights. Must be cited and distinguished.

**2505.14669 — Castro, Panferov, Tabesh, Alistarh et al., "Quartet: Native FP4 Training Can Be Optimal" (2025-05-20).** CLAIMS (fitted scaling law, eq. l.245-246): L = ... + A/(N·effN(P_forward))^α + B/(D·effD(P_backward))^β — forward-pass precision reduces *effective parameter count* (η-independent, capacity-type), backward-pass precision reduces *effective data* (noise-type); "effN appears to correlate heavily with MSE" of the forward quantizer; RTN best for backward effD, QuEST best for forward effN. THREAT: partially_preempts the forward=bias / backward=variance dichotomy at the scaling-law level (no LR dependence, cosine schedule only). The idea's Prediction 2 is the dynamical version of this.

**2509.25149 — NVIDIA, "Pretraining LLMs with NVFP4" (2025-09-29).** CLAIMS: 12B/10T, relative loss error "<1% stable ... slightly above 1.5% as the learning rate is decayed" (l.284-286); App. D Fig. 7: switch at 8.2T (18% of training) matches FP8; forward-only switch recovers "from 1.5% to 0.5% relative error"; backward-only: "no such improvement"; switch at 10T "fails to fully recover, presumably because the learning rate is too low for the weight updates" (l.1344-1350, 1405-1410). Also l.1812-1818: activation fwd/bwd inconsistency "only impacted in the later stages during the learning rate decay". THREAT: supports (the idea's main empirical anchor; no ablation of cause, as the idea says).

**2510.27527 — Chen et al., "TetraJet-v2" (v3 2026-05-11).** CLAIMS: "In the final stage ... the learning rate (LR) drops to near zero ... when the master weights converge as the LR is approaching zero, there are still drastic changes among the quantized weights. This phenomenon is weight oscillation" (l.276-286); oscillation suppression (reset to bin centre) is applied "when the learning rate decays to a relatively low value and oscillation problems emerge" (l.479-481). Fig. 6: switching to FP6×FP4 in the last 50B tokens of OLMo2-370M gives "substantial improvement", FP6×FP6 more (l.1236-1242). THREAT: supports (B's dynamic/oscillation component); offers an alternative mechanism (flip noise that does not shrink with η) which the idea must separate from the static Δ²tr(H)/24 term — note Nagel 2203.11086 App. A.3 shows decreasing LR "reduces the amplitude of the oscillations but do not affect their frequency" in the toy model.

### P3
**1706.02379 — Li, De, Xu, Studer, Goldstein, "Training Quantized Nets: A Deeper Understanding" (NIPS 2017).** PROVES: Thm 5 — for SR on quantized weights the Markov chain's stationary distribution is, for small α, nearly invariant to α and "not concentrated on minimizers"; BC (master weights) concentrates as α→0 (Cor. 1, Fig. 2). THREAT: partially_preempts the "η-independent floor from quantization" in the low-precision-weight regime (classic; cite).

**2604.15167 — "When Flat Minima Fail: Characterizing INT4 Quantization Collapse After FP32 Convergence" (2026-04-16).** CLAIMS (Pythia-160m, 154 checkpoints, calibration-free per-group INT4 PTQ probe): three-phase structure; "this divergence begins not when the learning rate starts decaying, but precisely when FP32 perplexity converges"; INT8 "entirely immune". THREAT: contradicts the idea's framing that LR magnitude is the driver (PTQ side, one small model); suggests the η-intercept experiment (Prediction 4) must control for convergence state, and that B may be grid-coarseness-specific.

**2605.09825 — Cim et al., "Pretraining LLMs with MXFP4 on Native FP4 Hardware" (2026-05-11).** CLAIMS (Llama-3.1-8B/C4, MXFP4): "quantizing Wgrad is the primary driver of convergence degradation, whereas FP4 in Fprop and Dgrad alone introduces only modest additional token requirements"; SR and random Hadamard "fail to stabilize training once Wgrad is quantized", deterministic Hadamard does. THREAT: contradicts the premise that forward quantization carries most of the gap (MXFP4, not NVFP4; degradation ≠ final decay-phase gap).

**2010.06192 — Zamirai, Zhang, Aberger, De Sa, "Revisiting BFloat16 Training" (2020).** CLAIMS: nearest-rounding weight updates "cancels small updates", degrading convergence as LR shrinks; SR or Kahan summation fixes. THREAT: contradicts/alternative explanation for "late switch fails because LR too low" (update swamping) — not applicable when master weights are FP32, but the idea should state that assumption explicitly.

**2509.22935 — Dremov, Hägele, Kosson, Jaggi, "Compute-Optimal QAT" (ICLR 2026).** CLAIMS: optimal QAT fraction grows with tokens-per-parameter-byte; proposes "QAT & Learning Rate Cooldown Fusion": "learning rate cooldown performs low-magnitude adjustments to weights, we speculate that a substantial part of updates during learning rate cooldown gets [lost]" (l.801-803). THREAT: new_hypothesis / tangential — reverse-direction switch (FP→QAT during cooldown) with a timing law; a natural comparison for Prediction 5.

**2606.15007 — NVIDIA, "Nemotron 3 Ultra" (2026-06).** CLAIMS: 550B-A55B NVFP4/20T WSD; BF16-switch ablations from 5T/10T/16T: start gap 0.27/0.28/0.25%, after 74B BF16 tokens 0.33%/0.34%/0.03% (l.483-490); "Switching all tensors to BF16 did not resolve the training divergence" (Fig. 3 bottom; divergence attributed in §2.7 to output-layer local gradient accumulation precision). THREAT: supports the idea's "instability ≠ floor" classification. Note: gap *increased* after switch from 5T/10T — not predicted by the idea (both runs at same η?), worth checking.

**2605.20402 — Li et al., "Decomposing MXFP4 quantization error for LLM RL: reducible bias, recoverable deadzone, and an irreducible floor" (2026-05-19).** PROVES: exact 3-way decomposition (scale bias / deadzone / grid noise), Lemma 3.2 orthogonality, grid noise invariant to scale precision → "irreducible floor"; CLT check std_theory=2.0 vs 1.97 measured. "Grid noise adds variance but no systematic bias to gradients". THREAT: tangential (RL post-training, MXFP4) but the vocabulary "bias vs irreducible floor" and the decomposition style overlap; cite to avoid the appearance of novelty inflation.

**2606.20381 — Zhao et al., "Rethinking Shrinkage Bias in LLM FP4 Pretraining ... UFP4" (2026-06-18).** CLAIMS: E2M1 non-uniform grid gives systematic negative (shrinkage) bias compounding across layers; E1M2 uniform grid (UFP4) reduces BF16-relative loss degradation; no LR-decay analysis found (grep). THREAT: supports the idea's first-order ∇Lᵀb̄ term; the idea's Prediction 3 (Δ² scaling) should be tested on a uniform grid where b̄≈0.

### Secondary (not in the 15, noted)
- 2602.02047 Dong et al.: SFT gap "continues to widen ... particularly during the learning rate decay phase", attributed to "diminishing learning rate, which amplifies the relative impact of quantization noise"; NVFP4 gradient norms higher late. supports/competing explanation.
- 2601.22813 Quartet II: SR at micro-scales inflates variance; MS-EDEN unbiased; 15-25% relative gap reduction. supports (variance channel).
- 2602.19241 Zhang et al.: Thm 4.2 additive quantization error is "a fixed floor" in sketched linear regression. supports (floor exists), no LR.
- 2606.09012 Li et al.: toy PTQ loss gap "grows linearly in the sharpness parameter μ". supports.
- 2605.25966 Thomassen: optimal warmdown fraction bit-width-agnostic (FP16/INT8/INT6; INT4≥50M wd33); "mechanism is not grid-snapping". tangential (consistent with B being schedule-independent).
- 2610.05329 Wang et al.: weight averaging as lag–noise trade-off for PTQ; "lower precision favors less noise". tangential.
- 2010.14298 Chen et al.: FQT gradient is an unbiased estimator of the QAT gradient; variance analysis (C_q). supports (A).
- 2502.15938 Straight to Zero: LR decay = gradient-noise averaging (A-term mechanism). supports (standard).
- 2502.20853 Oscillation-Reduced MXFP4 (ViT): "a large amount of latent weights lies around the quantization thresholds at the end of training". supports (B dynamic).
- 2203.11086 Nagel: LR reduces oscillation amplitude not frequency (toy). supports.
- 2606.00703: info-theoretic lower bounds for B-bit gradient oracles. tangential.
- 2505.14302 Chen (QAT scaling law): error rises with coarser group size and more tokens; weight vs activation decomposition. tangential (Prediction 3 empirical reference).
- 2501.02423 Sun: block-size term in FP-training scaling law. tangential (Prediction 3).

## 4. Critiques of the idea surfaced by this sweep
1. Prediction 3 as written ("group 16→8 quarters Δ²") is wrong for block-scaled formats: Δ ∝ blockmax/grid_max, and halving the block only mildly reduces the expected block maximum (extreme-value statistics; see idea 3 and Sun 2501.02423 / Chen 2505.14302 empirical granularity laws). Restate as "B ∝ E[Δ_block²]" measured, not ∝ 1/group².
2. Prediction 2 assumes the backward pass is unbiased. Chmiel App. B.2 shows any *bias* in gradient/update noise gives an η-independent floor μ²/(2λ); Quartet II argues SR of micro-scales and "plateauing" tricks introduce bias. So a backward-only switch *can* remove an η-independent component — Chmiel's own 7B QAF (backward→BF16) closed the gap. The idea should predict this conditionally.
3. The switch criterion (Prediction 5) competes with Chmiel's gradient-to-noise ratio ‖∇L‖/σ_q < √3 criterion; a fair paper must compare both.
4. 2604.15167 argues onset tracks FP32 convergence, not LR decay, for INT4 PTQ; Prediction 4 (η-intercept from two WSD final-η runs) must hold convergence state fixed.
5. The (η/4)tr(C) stationary formula assumes SGD; for Adam the floor scales differently (preconditioned). The idea admits this; Catalan-Tatjer's Fig. 26 (higher LR → lower tr(H)) also means tr(H) itself depends on η, so "B independent of η" is only true at fixed H — the decay-phase tr(H) growth is in fact the dominant B-dependence on η.
6. Nemotron 3 Ultra: gap *grew* 0.27→0.33% after switching all tensors to BF16 at the same η (5T, 10T) — the two-floor model predicts monotone recovery; needs an explanation (e.g. transient from Adam-state mismatch).
