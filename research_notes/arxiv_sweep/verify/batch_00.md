# arXiv citation verification — batch_00

Report: `reports/LLM 양자화 연구 아이디어 10선.md`
IDs: 2307.13304, 2507.18553, 2508.04853, 2411.17525
All four cited in one sentence (verification tags in brackets):

> 1. 엄밀한 이론은 거의 전부 "층별 MSE proxy" 위에 있다(QuIP 비간섭성 정리, GPTQ=Babai nearest-plane, 격자/TCQ 율-왜곡). proxy에서 end-to-end 손실로 가는 다리는 1차(HIGGS linearity theorem, KL≈¼‖δ‖²)이거나 느슨한 Lipschitz 상한뿐이다 ([QuIP 2307.13304] [V], [GPTQ=Babai 2507.18553] [S], [Provable PTQ 2508.04853] [S], [HIGGS 2411.17525] [S]).

Overall: the structural claim ("rigorous theory lives on the layer-wise MSE proxy; the bridge to end-to-end loss is a linear/approximate theorem or a Lipschitz-composition bound") is supported by all four papers. Two attribution details need fixing: (a) "KL≈¼‖δ‖²" is **not** in the HIGGS paper (HIGGS states a PPL, not KL, relation with per-layer coefficients α_l and a ½ constant); (b) Provable PTQ does not itself prove a Lipschitz end-to-end bound — it only points to the route and states end-to-end PPL guarantees are open.

---

## 2307.13304 — QuIP: 2-Bit Quantization of Large Language Models With Guarantees

- **Exists:** yes. Chee, Cai, Kuleshov, De Sa. Published 2023-07-25 (v2 2024-01-15).
- **Venue:** NeurIPS 2023 (text line 55: "37th Conference on Neural Information Processing Systems (NeurIPS 2023)").
- **Report claim:** QuIP's incoherence theorem is a result on the layer-wise MSE proxy. Tag [V].
- **Verdict: confirmed.**
- **Evidence (paper text):**
  - Sec. 3, Eq. (1): "we round weights per-layer by minimizing the 'adaptive rounding' proxy objective, as in Nagel et al. [20], ℓ(Ŵ) = E‖(Ŵ − W)x‖² = tr((Ŵ − W)H(Ŵ − W)ᵀ)" with H "the second moment matrix of these vectors, interpreted as a proxy Hessian."
  - Theorem 1: LDLQ is worst- and average-case optimal for the proxy: "m/4 tr(D) = Lworst(LDLQ, H) ≤ Lworst(A, H)", "c = 12 for nearest, c = 6 for stochastic".
  - Definition 1 (μ-incoherence): "|Qij| ≤ μ/√n" for the Hessian eigenvectors; "|Wij| ≤ μ‖W‖_F/√(mn)" for weights.
  - Lemma 2 (the "incoherence theorem"): "Let H be a μ-incoherent PSD symmetric matrix … Then tr(D) ≤ (μ²/n) tr(H^{1/2})²." Paper: "a novel result using incoherence to obtain a bound on tr(D) that depends only on the spectrum of H."
  - Theorem 4: without incoherence, the best spectral bound cannot separate LDLQ from nearest/stochastic rounding ("sup Lworst(LDLQ, H̃) = Lworst(Stoch, H) = m/4 tr(H)").
  - Theorem 6: OPTQ (GPTQ) is in the LDLQ class, so QuIP's proxy theory covers GPTQ too.
  - Theorem 7: with incoherence processing and a convex "fixed" program, w.p. ≥ 1−δ no clipping is needed and "tr((Ŵ − W)H(Ŵ − W)ᵀ) = Õ((1/(n² 4^b)) tr(H^{1/2})² ‖W‖_F²)". Still a proxy-loss statement.
  - Own limitation (App. A.2): "The adaptive rounding [3] proxy objective considers each layer in isolation; it remains to be seen what other computationally tractable proxies could improve quantization."
- **New relevant facts:** every theorem is on the proxy ℓ(Ŵ); no end-to-end statement. Theorem 7's clean bound requires the ADMM "fixed" variant, which the authors say they never run in practice ("we always just use QuIP as described in the previous sections").
- **Correction:** none. [V] tag is appropriate.

---

## 2507.18553 — The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm

- **Exists:** yes. Chen, Shabanzadeh, Crnčević, Hoefler, Alistarh. Published 2025-07-24 (v-latest 2026-05-13).
- **Venue:** ICLR 2026 (arXiv comment: "Published as a conference paper at the Fourteenth International Conference on Learning Representations (ICLR 2026): https://openreview.net/forum?id=NFB4QGGS65"; journal_ref "International Conference on Learning Representations, 2026, pp. 122653-122695").
- **Report claim:** "GPTQ=Babai nearest-plane" is a layer-wise MSE-proxy theory. Tag [S].
- **Verdict: confirmed.** Recommend upgrading [S] → [V].
- **Evidence (paper text):**
  - Contributions (Sec. 1): "(i) the GPTQ optimization problem, i.e., linear-layer quantization with the L2 objective on the output, is equivalent to the closest vector problem (CVP) w.r.t. L2 distance; (ii) the GPTQ algorithm executed from the last to first dimension is the same as Babai's nearest plane algorithm on the basis of the factorized Hessian matrix, without LLL basis reduction … (iii) the worst-case layer-wise error in the no-clipping setting is bound tightly by the trace of the diagonal matrix of the LDL decomposition of the Hessian matrix." The paper itself calls this "a layer-wise global error bound."
  - Theorem 4 (GPTQ and Babai): "GPTQ and Babai's algorithm without basis reduction will have the same results if we align the dimensional order of these two algorithms, e.g., running GPTQ from the last to the first dimension."
  - Theorem 5 (GPTQ Error Bound): "Assume no clipping (Z† = Z) … the (absolute) quantization error has a tight upper bound: ‖X diag(sᵢ) zᵢ − Xwᵢ‖² ≤ ¼ (T⁻¹sᵢ)ᵀ D (T⁻¹sᵢ)"; relative bound 1 ≤ γ ≤ √(c+1) · max d_j'/d_j.
  - Sec. D.2: under a uniform prior inside Babai's hyper-cuboid, "the average layer-wise quantization error is exactly 1/3 of the worst-case guarantee stated in Theorem 5."
  - Sec. 5: "The original GPTQ algorithm clips the overflowed integers at the rounding step, introducing large errors that violate the error bound in Theorem 5."
- **New relevant facts:** the bound holds only with no clipping (unbounded integer grid); real INT4 GPTQ with clipping is outside the theorem. Order matters: min-pivot order reduces tr(D) vs act-order but "downstream accuracy gains are modest." Finding the optimal order is NP-hard (Rose et al., 1976). Concurrent work: Birnick (2025). The equivalence is "tight": composing GPTQ update after Babai is "algebraically redundant" (Sec. C.4).
- **Correction:** none to the sentence. Optionally note "no-clipping assumption" when citing the bound.

---

## 2508.04853 — Provable Post-Training Quantization: Theoretical Analysis of OPTQ and Qronos

- **Exists:** yes. H. Zhang, S. Zhang, Colbert, Saab. Published 2025-08-06 (v2 2026-04-09).
- **Venue:** preprint/unknown (no arXiv comment/journal_ref; WebSearch found no acceptance). Formatting (running heads "H. ZHANG, S. ZHANG, I. COLBERT, R. SAAB") suggests a SIAM-style journal submission, unconfirmed.
- **Report claim:** cited as supporting "proxy → end-to-end bridge is only a loose Lipschitz upper bound." Tag [S].
- **Verdict: partially confirmed.** The paper's theorems are all layer-wise calibration-error bounds (ℓ2 and ℓ∞). It does **not** prove an end-to-end Lipschitz bound; it only says such a bound could be obtained under extra assumptions by citing earlier work, and explicitly states that end-to-end PPL/accuracy guarantees are an open problem. So it supports the report's *diagnosis* but should not be read as *providing* the Lipschitz bound.
- **Evidence (paper text):**
  - Theorem 3.3 (General ℓ2 bound, λ>0): "‖Xw − Xq‖₂² + λ‖w − q‖₂² ≤ (δ²/4)·N·C₂(X, λ)²".
  - Corollary 3.5: "‖Xw − Xq‖₂ ≤ (√N δ/2) · min{√(Tr(XᵀX)/N + λ), ‖X‖op}".
  - Theorem 4.6 (ℓ∞ bound, stochastic rounding): "‖Xw − Xq‖∞ ≤ δ√(2πp log N) · C∞(X, λ)" w.h.p.; deterministic OPTQ can have "‖Xw−Xq‖∞ = ‖Xw−Xq‖₂ = O(√N)" (Sec. D).
  - Theorem 3.4 rigorously justifies ordering columns by decreasing norm.
  - Sec. 5: Qronos bounds (Prop. 5.3, Thm 5.4); "the first term in the Qronos error … is reduced" relative to OPTQ, explaining the gap when OPTQ uses activations X̃ from the partially-quantized net.
  - Conclusions (Sec. 6): "Our analysis focuses on the layer-wise calibration errors … Under added assumptions, for instance Lipschitz nonlinear activation functions, our results can be used to bound the calibration error induced by quantizing the weights of an L-layer MLP (see, e.g., [39, 40]). It should also be possible to extend this framework to transformer layers used in LLMs, and to loss functions such as the cross-entropy loss but we leave this to future work. By contrast, obtaining guarantees directly for performance metrics such as perplexity and zero-shot accuracy on downstream tasks remains an open problem."
  - Motivation for ℓ∞ (Sec. 4): "if ‖Xw − Xq‖∞ does not exceed half the entrywise gap within the sorted entries of Xw, the ordering—and thus the output—remains intact. An ℓ2 bound cannot guarantee this."
- **New relevant facts:** the ℓ∞ result is the paper's only step toward nonlinearities (softmax ranking preservation), and it needs stochastic rounding. The ℓ2 worst case grows as √N·δ and is shown tight by construction (Sec. D). Regularization λ ≈ const·‖X‖_F²/N is heuristically justified (Remark 3.7).
- **Correction (suggested wording):** "Provable PTQ(2508.04853)는 OPTQ/Qronos의 층별 보정오차 ‖Xw−Xq‖에 대한 ℓ2(Thm 3.3/Cor 3.5)·ℓ∞(Thm 4.6, 확률적 반올림) 상한을 주며, 다층 확장은 Lipschitz 활성화 가정 하에 선행연구[39,40]로 가능하다고만 언급하고, PPL/다운스트림 정확도 보장은 '열린 문제'라고 명시한다." Keep [S] or mark [V] with this clarification.

---

## 2411.17525 — Pushing the Limits of Large Language Model Quantization via the Linearity Theorem (HIGGS)

- **Exists:** yes. Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh. Published 2024-11-26 (single version).
- **Venue:** NAACL 2025 Long Papers, pp. 10857–10886, published title "HIGGS: Pushing the Limits of Large Language Model Quantization via the Linearity Theorem" (ACL Anthology 2025.naacl-long.543, DOI 10.18653/v1/2025.naacl-long.543).
- **Report claim:** "proxy → end-to-end bridge is first-order (HIGGS linearity theorem, KL≈¼‖δ‖²)". Tag [S].
- **Verdict: partially confirmed.**
  - Confirmed: the linearity theorem is a bridge from per-layer ℓ2 (relative Frobenius MSE) error to the model-level loss (perplexity), linear in the per-layer MSE.
  - Incorrect: "KL≈¼‖δ‖²" does not appear in this paper. HIGGS states E[PPL(Ŵ)] ≈ PPL(W*) + Σ_l α_l t_l² — a perplexity (not KL) relation, with layer-dependent coefficients α_l (not a universal ¼), in terms of relative error t_l² = E‖Ŵ_l−W*_l‖_F²/‖W*_l‖_F² (not raw ‖δ‖²). The only KL mention is as an alternative calibration metric for the data-free mode.
  - Imprecise: "1차" — the relation is linear in MSE but is derived from a *second-order* Taylor expansion around a local minimizer (first-order term vanishes); it is an approximation (≈) with a third-order remainder, valid only for "small enough" t_l and empirically validated in the 3–8 bit range.
- **Evidence (paper text):**
  - Abstract: "a 'linearity theorem' establishing a direct relationship between the layer-wise ℓ2 reconstruction error and the model perplexity increase due to quantization."
  - Intro: "for reasonable quantization bit-widths, the relationship between per-layer MSE and output PPL is linear, modulo a constant scaling coefficient per layer, which is independent of the quantization approach. We show experimentally that the theorem works remarkably well at predicting quantization error for different schemes in the 3-8 bit range."
  - Assumption 1: "The uncompressed model weights w⋆ are a local minimizer of perplexity ϕ" ("not needed if the compression mechanism … is unbiased"). Assumption 2: "ϕ is three times continuously differentiable in a neighborhood of w⋆." Assumption 3: "D⋆∇²ϕ(R(W⋆))D⋆ ≈ Z" with Z block-diagonal, Z_l = z_l I (validated empirically on OPT-125M, App. E).
  - Theorem 1: "t_l² := E‖Ŵ_l − W_l⋆‖_F² / ‖W_l⋆‖_F² … Then, as long as t₁,…,t_L are small enough, the following linear approximation of the expected perplexity holds: E[PPL(Ŵ)] ≈ PPL(W⋆) + Σ_{l=1}^{L} α_l t_l²".
  - Proof (App. D): "From Taylor's approximation of ϕ around w⋆ … = ϕ(w⋆) + ½ ϕ''(w⋆) t² |w⋆|² ξ²"; multivariate: "ϕ(w⋆) + ½ E[Σ_l t_l² z_l ‖ξ_l‖²₂]" so α_l ∝ ½ z_l.
  - KL mention (Sec. 5, data-free dynamic mode): "we can change the metric we use in calibration … from perplexity on a calibration dataset to KL-divergence between pretrained and quantized models on randomly sampled text tokens. We evaluate KL-divergence on 287k random tokens." No formula given.
  - Sec. 6.1: "One key point in the Theorem is that the result holds for sufficiently small relative per-layer quantization errors t_l."
- **New relevant facts:** the theorem implies that after Hadamard rotation (weights ≈ Gaussian) "MSE-optimized grids should be theoretically-optimal in terms of end-to-end model error, given a fixed bit budget"; α_l are quantizer-independent and estimated by regression (Alg. 3); the per-layer bit allocation becomes a DP problem. Applicability caveat: low-bit (≈2-bit) regimes where t_l is not small are outside the theorem's stated range; the paper notes competing methods are "Pareto-competitive down to 2 bits" but validates its own theorem at 3–8 bits.
- **Correction (suggested wording):** replace "1차(HIGGS linearity theorem, KL≈¼‖δ‖²)" with "2차 Taylor 근사에서 나온, 층별 상대 MSE에 선형인 근사식(HIGGS linearity theorem: E[PPL(Ŵ)] ≈ PPL(W*) + Σ_l α_l t_l², t_l 충분히 작을 때·3–8bit에서만 검증)". Remove or re-source "KL≈¼‖δ‖²" — it is not from 2411.17525 (and no paper in this batch contains it).
