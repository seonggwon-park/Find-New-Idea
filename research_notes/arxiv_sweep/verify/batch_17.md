# Citation verification — batch 17

IDs: 2609.05764, 2507.11273, 2606.24033, 2509.14391. All four exist on arXiv. Paper text was fetched with `scripts/arxiv_fetch.py`; quotes below are verbatim from the downloaded text (untrusted data, quoted only).

Report context for all four (report_citations.json):

- 2609.05764: "Interface-Aware KV는 Hadamard를 post-RoPE에 적용 ([2609.05764] [S])" — listed as a counter-example to the "pre-RoPE is better" camp.
- 2507.11273 / 2606.24033 / 2509.14391 (one sentence): "KV-Latent는 저번호 RoPE 채널이 노이즈 대부분 ([2507.11273] [S]); Block-GTQ는 2-D 주파수 블록별 에너지 비트 할당 ([2606.24033] [S]); Q-ROAR와 'Rethinking RoPE scaling'은 YaRN × W4가 파괴적으로 상호작용(NTK/linear는 PPL>100)하고 대역별 스케일로 복구한다고 보고 ([2509.14391], [2510.00028] [S])."

---

## 2609.05764 — Interface-Aware KV Cache Quantization for Dense On-Chip NVM in Long-Context LLM Decoding

- First author: Jiahao Zheng (with Yifan Qin, Xiaobo Sharon Hu, Yiyu Shi). Published 2026-09-04. Primary cat cs.AR.
- Venue: **ICCAD 2026** (arXiv comment: "IEEE/ACM International Conference on Computer-Aided Design (ICCAD 2026)"; DOI 10.1145/3831252.3845860).
- Verdict: **confirmed**. Tag can move [S] → [V].
- Claim checked: "applies Hadamard post-RoPE". Section 3.2, verbatim:
  > "Each vector x ∈ R^d ... is transformed by a randomized Hadamard matrix H, a scaled Hadamard matrix combined with a random sign flip on each coordinate."
  > "We apply this rotation after the rotary position embedding (RoPE) rather than before, because the two transformations do not commute; RoPE applies to queries and keys, while values carry no positional encoding and are rotated directly. Placing H after RoPE stores both the query and the key in the same H RoPE(·) frame, so the shared H cancels in the score, (H RoPE(q))^T (H RoPE(k)) = RoPE(q)^T RoPE(k), and the key can be scored in its rotated form with no inverse rotation on any read; the opposite order would force one per key read."
- Important nuance for the report: the post-RoPE choice is justified by **hardware read-path cost** (avoiding an inverse rotation per key read on an NVM/crossbar interface), **not** by an accuracy comparison against pre-RoPE. The paper does not report a pre-RoPE vs post-RoPE accuracy ablation, so it is weak evidence "against" pre-RoPE on accuracy grounds. The abstract itself says "KIVI and KVQuant remain more accurate in software; the advantage of our format lies at the memory interface".
- Other facts: 4-bit KV, one analytic key codebook (post-Hadamard key coordinate ≈ Gaussian, "measured excess kurtosis ≈ −0.16 and KL divergence 5 × 10^−4"), one calibrated value codebook; per-vector metadata is one scalar norm ("about 3%"); "3.1-3.6x lower KV read energy than both [KIVI, KVQuant] mapped to the same NVM, and 8x lower metadata overhead than KIVI"; models Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-14B (d=128), backbone QuaRot W8A8 (Sec. 4.1.1); contexts to 32k.

---

## 2507.11273 — KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding

- First author: Luohe Shi (with Zuchao Li, Lefei Zhang, Guoming Liu, Baoyuan Qi, Hai Zhao). Published 2025-07-15. cs.CL.
- Venue: **ACL 2025** (arXiv comment: "To be published in The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)").
- Verdict: **confirmed**. Tag can move [S] → [V].
- Claim checked: "low-numbered RoPE channels contribute most of the noise". Section 3.3.2 "Pattern Finding", verbatim:
  > "Our findings suggest that the low-numbered channels of the RoPE function contribute the majority of the noise, while the high-numbered channels, despite a slower decay, remain relatively stable."
  And 3.3.3: "lower-numbered channels correspond to high-frequency rotations and higher-numbered channels correspond to low-frequency rotations".
- Nuance the report should keep in mind: the "noise" here is **not quantization noise**. It is the oscillation of the RoPE self-similarity curve RoPE_{θ,d}(x) = 1_d · R_{θ,d}(x) · 1_d^T when the head dimension d is **small** (the paper's setting is dimensional down-sampling of Q/K to a latent space, with d < 32 where "the range of oscillation is comparable with intended attenuation"). The finding is an empirical/analytic observation (Figures 3-6, Formula 12 integral approximation argument), not a theorem about quantization. The report's sentence uses it as evidence about frequency structure, which is fair, but it should not be read as "low-index channels are the most quantization-sensitive".
- Fix proposed in the paper: Formula 11 re-samples frequencies ("densifying the sampling of low-frequency rotations and avoiding high-frequency rotation sampling"). Other facts: extra training "less than 1% of pre-training"; model is "more sensitive to d_vo comparing to d_qk".

---

## 2606.24033 — RoPE-Aware Bit Allocation for KV-Cache Quantization (Block-GTQ)

- First author: Fengfeng Liang (with Yuechen Zhang, Jiaya Jia). Published 2026-06-23. cs.LG.
- Venue: **preprint** (comment: "Preprint. Code available at https://github.com/JIA-Lab-research/blockgtq").
- Verdict: **confirmed**. Tag can move [S] → [V].
- Claim checked: "energy-based bit allocation per 2-D frequency block". Verbatim:
  > Abstract: "a key's contribution to a future attention logit decomposes into a position-dependent sum over two-dimensional frequency blocks ... For each layer and KV head, Block-GTQ computes a label-free energy score for each RoPE block and greedily allocates integer bit widths by marginal gain."
  > Sec. 2.1: "|K_Δ(q,k) − K_Δ(q,k̂)| ≤ Σ_{i=1}^{L} ‖q^(i)‖_2 ‖e_k^(i)‖_2. Each block thus contributes independently to the bound, with no cross-block terms. ... norm-based block statistics are RoPE-invariant — we can therefore compute the energy score in pre-RoPE coordinates."
  > Sec. 3.1: energy score "s_i := ½ E[‖q^(i)‖²_2 + ‖k^(i)‖²_2]. By AM-GM, s*_i ≤ s_i in expectation".
- What is PROVED vs CLAIMED: Lemma 2 (per-block accounting of logit error, Cauchy–Schwarz + rotation orthogonality) and Theorem 1 ("Greedy optimality for the allocation objective": greedy minimizes J(b)=Σ s_i 4^{-b_i} under integer budget) are proved; the 4^{-b} rate law is inherited from TurboQuant-MSE (C_TQ = 3π/2). The allocation only touches K; "Block-GTQ allocates bits only on K, leaving V uniform" (Limitations, Sec. 7). Empirical numbers (abstract/conclusion): per-layer MAE cut 32–80% at 2 and 3 b/dim K-only, 367/367 layer wins vs uniform TQ-MSE; Llama-3.1-8B-Instruct K2V2 NIAH 70.6 → 97.4, LongBench-EN 36.87 → 53.31; DeepSeek-R1-Distill-Qwen-7B K3V2 AIME24/25 51.7/37.5 vs fp16 54.2/37.9, uniform TQ-MSE 0.0/0.0 (no fp16 recent-key buffer); Qwen2.5-3B packed K3V3: 3.24x compression, 1.34x faster than fp16 FA2 at 128K, peak memory 56.31 GB → 19.85 GB.
- Relevant to the report's "no theory in a single complex-coefficient representation" statement: Block-GTQ already analyses the logit as a sum over 2-D (i.e., complex-plane) RoPE blocks with a per-block bound, so the report's novelty claim should be phrased as "no unified analysis of all three phenomena", not "no complex/block analysis at all".

---

## 2509.14391 — Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs

- First author: Ye Qiao (with Sitao Huang), UC Irvine. Published 2025-09-17. cs.LG. The arXiv text is a 2-page short paper (10.5k chars).
- Venue: **AAAI 2026 Student Abstract** (text carries "Copyright © 2026, Association for the Advancement of Artificial Intelligence"; web search: ojs.aaai.org article 42269, Proc. AAAI 40(48), 41359–41361). Not stated in arXiv comment.
- Verdict: **partially confirmed**.
  - Confirmed: Q-ROAR reports a destructive YaRN × W4 interaction and recovers with per-band scales on W_Q, W_K. Verbatim: "Under YaRN interpolation (32K/64K), RTN and AWQ exhibit clear drops in accuracy and higher perplexity, confirming the destructive interaction between position interpolation and quantization." and "We partition RoPE pairs into B log-spaced frequency bands {B_b} and assign a single scale g_b" (Eq. 4; B ∈ {6, 8}).
  - **Not in this paper**: the parenthetical "(NTK/linear는 PPL>100)". 2509.14391 evaluates **YaRN only** (s=8 at 32K, s=16 at 64K); the words "NTK" and "linear interpolation" do not appear outside the abstract's generic mention of "linear and frequency-aware scaling". Its worst reported perplexity is **6.713** (Table 1, RTN W4, GovReport, 32768 ctx, YaRN 64K s=16) vs FP16 6.069 and Q-ROAR W4 5.833. The PPL>100 statement comes from the companion paper **2510.00028** (line 293 of its text: "With more naïve interpolation strategies like NTK or linear scaling, the models fail entirely at long contexts, reaching perplexity values greater than 100.").
  - Magnitude caveat: the "destructive" interaction in 2509.14391 is modest in its own numbers. Table 2 (LLaMA-2-7B, YaRN 32K s=8): 5-shot avg FP16 63.71% vs RTN W4 63.32% vs Q-ROAR W4 63.96%; WikiText2 PPL FP16 6.09, RTN W4 6.41, Q-ROAR W4 6.19. Abstract: "recovers up to 0.7% accuracy on standard tasks and reduces GovReport perplexity by more than 10%" (6.713 → 5.833 at 32768 is −13%).
- Correction: in the report sentence, attribute "(NTK/linear는 PPL>100)" to [2510.00028] only, e.g. "Q-ROAR([2509.14391], AAAI'26 student abstract)는 YaRN × W4에서 GovReport PPL 6.07→6.71 악화를 대역별 W_Q/W_K 스케일로 5.83까지 복구; 'Rethinking RoPE scaling'([2510.00028])은 NTK/linear 보간에서 PPL>100 붕괴를 보고". Method scope: weight-only (W_Q, W_K rescale), LLaMA-2-7B only, RTN/AWQ W4 and W4-A4; "assumes RoPE and targets weight-quantized models (heavy activation/KV quantization may still benefit from modest clip expansion guided by TIR)".
