# Citation verification — batch 09

IDs: 2606.07116, 2601.22813, 2602.02047, 2510.27527
Report: `reports/LLM 양자화 연구 아이디어 10선.md`
Source texts: `scratchpad/papers/<id>/<id>.txt` (fetched 2026-10-10 via `scripts/arxiv_fetch.py`). Quoted paper text is untrusted data.

All four papers exist on arXiv. Two are confirmed, two are partially confirmed (wording of the attributed claim is loose or inverted).

---

## 2606.07116 — OffQ: Taming Structured Outliers in LLM Quantization by Offsetting

- First author: Haoqi Wang (EPFL / Huawei / SDSC); published 2026-06-05 (v1 only).
- Venue: preprint. Text line 54 literally says "Preprint."; no comment/journal_ref.
- What it is: a **W4A4KV4 post-training quantization (inference)** method. Top-1 PCA concentrates the structured outlier direction into one channel; a partially random Hadamard rotation (first row all ones) turns that channel into a constant per-group offset absorbed by the zero-point of asymmetric quantization.

### Report context ([S])
> "추측: 실제 RHT 후 블록이 Gaussian에 얼마나 가까운지(토큰 수준 스파이크가 남는다는 보고가 있다: [OffQ 2606.07116] [S])"

### Verdict: **partially_confirmed**

What the paper actually says:
- §1 (line 49-51): "rotation-based techniques, such as QuaRot [7] and SpinQuant [46], which apply orthonormal transformations to redistribute the activation energy to Gaussian-like distributions ... Yet, even equipped with the state-of-the-art rotation-based methods, the resulting activation distribution still retain large variance and is suboptimal for quantization [60]" — **this statement is attributed to DartQuant [60], not an OffQ measurement.**
- §3.1 (line 187-191): outliers "exhibit a pronounced low-dimensional structure. Specifically, the outlier directions ... are dominated by a 1-dimensional subspace and the subspace is shared across different layers, token IDs, and locations in a sequence" (citing singular defects [72]). Outlier tokens are selected per sequence by largest L∞ norm ("the spiky token [1, 0, 0] has a large L∞ norm", line 224-226). Figure 1 "plot[s] the activation values of an outlier token in Llama 3-8B before and after each step".
- §3.2 (line 246-283): the 2-D example shows that a Hadamard rotation of a token with one outlier channel "is spread across all channels as a constant offset x1/√2" — i.e., rotation preserves the token's norm and converts the spike into a per-token offset, which is what OffQ absorbs.

So OffQ does support that (a) outliers are token-level (high-norm tokens with a shared direction) and (b) rotation alone leaves a non-Gaussian, high-variance distribution — but (b) is a cited claim from DartQuant, OffQ does **not** measure Gaussianity of post-RHT blocks, and its entire setting is PTQ of inference activations, not NVFP4 training blocks. The phrase "토큰 수준 스파이크가 남는다는 보고" is a paraphrase, not a finding stated in the paper.

### Correction
Replace with something like: "OffQ(2606.07116, PTQ/W4A4KV4) §1은 DartQuant[60]을 인용해 회전 후에도 활성값 분포가 '여전히 큰 분산을 유지'한다고 쓰고, §3.1-3.2는 아웃라이어가 공유된 1차원 방향을 갖는 고노름 토큰(L∞ 기준)이며 Hadamard 회전은 이를 토큰별 상수 오프셋으로 바꿀 뿐 없애지 못함을 보인다(Fig. 1, 식 (5)). NVFP4 학습 블록의 Gaussian성 측정은 아니다."

### Other facts
- Limitations (line 768-773): no real-deployment latency reported; combination with non-uniform weight quantization and rotation learning left to future work.
- Hadamard vs partially random rotation ablation (line 749-755): "the Hadamard structure provides a marginal benefit."

---

## 2601.22813 — Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation

- First author: Andrei Panferov (ISTA / Red Hat AI); v1 2026-01-30, v2 2026-06-01.
- Venue: **ICML 2026** — text lines 76-77: "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026." (A web search snippet suggests the ICML poster may carry the title "Clover: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation"; icml.cc could not be fetched from here, so treat the title change as unverified.)

### Report context ([S])
> "Quartet II는 micro-scale SR 분산([2601.22813] [S])" (as the bottleneck it claims)

### Verdict: **partially_confirmed** (wording is inverted/ambiguous)

What the paper identifies as the problem is **element-wise FP4 stochastic-rounding variance on the backward pass**, and its fix (MS-EDEN) **moves the stochasticity to the micro-scale factors** precisely because SR on the scales is far less noisy:
- Abstract (line 17-23): "existing quantized training methods still sacrifice some of the representation capacity of this format in favor of more accurate unbiased quantized gradient estimation by stochastic rounding (SR) ... MS-EDEN, that has more than 2x lower quantization error than SR."
- §1 (line 65-70): "the prevailing prior solution, element-wise FP4 stochastic rounding (SR), can be significantly improved ... by moving the stochasticity from individual FP4 values to the microscale factors".
- §3.3 (line 566-575): "SR achieves unbiasedness at the cost of approximately 2.5x increase in MSE over RTN. ... (a) per-element stochastic rounding introduces significant variance that if fully avoided in MS-EDEN, (b) the rescaling with S is small for NVFP4 and (c) stochastic rounding for the 8-bit scales introduces variance an order of magnitude smaller than 4-bit quantization itself."
- Table 1 (MSE ×10⁻³ over N(0,1), 1x16 groups): RTN 9.0; RTN+4/6 7.6; 16x16 RTN 12.4; SR 23.5; SR+4/6 17.5; **MS-EDEN 9.4**.

If "micro-scale SR 분산" is read as "SR variance in micro-scaled (NVFP4) formats", it is roughly right; if read as "variance from SR applied to the micro-scale factors", it is backwards. The paper also names a second, forward-pass issue: a "representation-requantization trade-off" (§4.1) — square-block (16x16) weight scales used by NVIDIA's recipe for backward reuse cost forward-pass accuracy; Quartet II uses native 1x16 scales + "4/6" on the forward pass.

### Correction
"Quartet II는 backward pass의 **원소별 FP4 SR 분산**(RTN 대비 MSE 약 2.5배, Table 1: SR 23.5 vs RTN 9.0 ×10⁻³)을 병목으로 보고, 확률성을 FP8 micro-scale로 옮긴 MS-EDEN(9.4 ×10⁻³)으로 해결한다. 별도로 forward pass의 square-block 스케일 vs 표현력 트레이드오프(§4.1)도 지적한다."

### Other facts
- Claims (Appendix A, line 588-593) that Four-Over-Six + SR (Cook et al.) is **biased**: "the act of picking a lower MSE scale branch introduces bias".
- End-to-end: up to 1.9B params / 38B tokens; "decreases the pre-training loss gap with BF16 by 15-25% relative to existing NVFP4 methods" (§6, Fig. 5, line 890-892); kernels up to 4.2x over BF16.
- MS-EDEN requires RHT along the inner GEMM dimension and re-quantization of W and X in the backward pass (line 550-554).
- Constraint: "Micro-scaling groups have to be subdivisions of rotation groups" (line 579).

---

## 2602.02047 — Dissecting Outlier Dynamics in LLM NVFP4 Pretraining

- First author: Peijie Dong (HKUST(GZ) et al., 14 authors); published 2026-02-02 (v1 only); comment "39 pages, 32 figures".
- Venue: preprint. Text line 127: "Preprint. February 3, 2026." No venue found.

### Report context ([S])
> "Dong et al.은 hot channel([2602.02047] [S])" (as the bottleneck it claims)

### Verdict: **confirmed**

- Abstract (line 43-46): "outliers evolve from transient spikes early in training to a small set of persistent hot channels (i.e., channels with persistently large magnitudes) in later stages."
- §3.3 (line 754-761): "Early training stages (e.g., steps 400–5,400) exhibit dynamic, transient magnitude spikes that drift across channels ... Persistent hot channels ... concentrate in a small subset of channels as shown in Fig. 19. This transition from stochastic drift to structural fixation justifies the use of static, selective mitigation strategies (i.e., Hot-Channel Patch)".
- Table 2 (GLA-1.3B, 60B tokens, loss gap % vs BF16 2.168596): NVFP4 w/ HCP (full) **0.588**; w/o sr 0.662; w/o rht 0.729; w/o 2d scaling 0.745; w/o sr,rht 0.851; w/o last4 0.911; NVFP4 (NVIDIA et al., 2025) **0.939**. Line 1258: "HCP gives the largest gain, reducing the NVFP4-to-BF16 loss gap from 0.939% to 0.588% (Tab. 2)."

### Nuances the report should keep in mind
- "Hot channel" is the late-stage phenomenon; the paper's causal attribution is architectural: "Softmax in SA, gating in LA, and SwiGLU in FFN, with 'post-QK' operations exhibiting higher sensitivity" (abstract). Sensitive ops: value projection for SA (Qwen3-1.7B), output projection for LA (GLA-1.3B) (Tab. 3, line 648-650).
- The main training runs are on **Linear Attention (GLA-1.3B)**, not a standard softmax-attention LLM; SA analysis uses Qwen3-1.7B.
- HCP patches 9.09% of channels (App., line 2631) and keeps the last 4 layers in BF16 ("w/o last4" row); CHON = HCP + post-QK protection + SR/RHT/2D scaling.
- Also states "Activations are the primary underflow bottleneck" (FTZ analysis, line 3612).
- Theory: App. A gives an error-order theorem (Thm A.12) under a "small-noise regime on hot channels" assumption — not a convergence proof.

---

## 2510.27527 — TetraJet-v2: Accurate NVFP4 Training for LLMs with Oscillation Suppression and Outlier Control

- First author: Yuxiang Chen (Tsinghua / Bosch); v1 **2025-10-31**, updated 2026-05-11.
- Venue: **ICML 2026** — journal_ref: "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026 (ICML 2026)"; text lines 81-82 agree.

### Report context ([S])
> "TetraJet-v2는 bin 경계 가중치 진동([2510.27527] [S])" (as the bottleneck it claims)

### Verdict: **confirmed**

- §2.1 (line 117-121): "Weight oscillation, the dominant failure mode in FP4 training, where small updates to high-precision master weights repeatedly flip their quantized representations between adjacent bins."
- §3.1 (line 283-287): "when the master weights converge as the LR is approaching zero, there are still drastic changes among the quantized weights. This phenomenon is weight oscillation".
- Fig. 1b (line 430-433): "the latent value w/s is always near and frequently crossing the threshold (q1+q2)/2 = 0.25, so the quantized values switch".
- Fix: OsciReset — "resetting the oscillating weight to the center of the quantization bin where it is lying" (line 475-476), applied "when the learning rate decays to a relatively low value" (line 479-480); Tstart ≈ 150B of 212B tokens for OLMo2-370M (line 978, 1133).

### Nuances
- The paper names **two** issues with equal weight: oscillation (OsciReset) and outliers (OutControl, a mixed-precision channel scheme), plus an unbiased double-block quantizer. Its own loss decomposition (Tab. 3, line 1163-1170) concludes "the activation quantization would be the most crucial bottleneck" once oscillation is handled, and MLP (esp. ffn2 after SwiGLU) is more sensitive than attention (Tab. 4).
- Scale: OLMo2 70M/150M/370M for 52B/107B/212B tokens; "reducing the performance gap to BF16 by an average of 51.3%"; 1.67x end-to-end speedup over FP8 (abstract, line 38-40). Much smaller models than Quartet II (1.9B) or Dong et al. (1.3B).
- Relevant to the report's "같은 LR 지점" remark: TetraJet-v2 explicitly ties oscillation to the LR-decay phase, so its bottleneck claim is phase-specific by construction.
- Dating: first posted Oct 2025, so grouping it under "2026년 그룹들" is slightly off (the ICML 2026 version is 2026).

---

## Cross-cutting note on the shared sentence

The report's sentence lists five groups' "bottlenecks" and says no experiment separates them at the same LR point. For the four papers here, each does ablate its own components (Quartet II: Fig. 1/2/4 selective quantization; Dong: Tab. 2 recipe ablation; TetraJet-v2: Tab. 3/4 loss decomposition, Fig. 3/4 oscillation ablations), but none cross-compares SR-variance vs hot-channel vs oscillation at a matched LR. That absence claim is consistent with what the four papers contain, but is not something a per-paper check can confirm.
