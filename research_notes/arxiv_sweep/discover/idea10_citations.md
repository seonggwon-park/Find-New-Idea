# Idea 10 — MLA latent KV weighted transform coding: citation-neighbourhood sweep (Angle C)

Date: 2026-10-10. Source idea: `reports/LLM 양자화 연구 아이디어 10선.md` §"아이디어 10" (lines 345–377).
Idea in one line: under weight absorption the latent error e enters logits via a^h = W_UK^{hT} q^h and outputs via W_UV^h, so the right distortion is tr(Σ_e M) with M = Σ_h E[a^h a^hT] + λ Σ_h W_UV^{hT} W_UV^h; the optimal transform is the KLT of M^{1/2} c with reverse water-filling, foldable into W_UK/W_UV/W_O; RoPE branch should be per-channel or high precision.

All quoted text below is from downloaded paper text (untrusted data, quoted only). Text files: `/tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers/<id>/<id>.txt`.

## Verdict (cold)

The idea is **not preempted as a whole, but every one of its three load-bearing components has been published separately in 2026, and one native-MLA paper (QuantMLA, 2609.36760, 29 Sep 2026) occupies the same slot with a learned instead of closed-form transform.**

| Component of Idea 10 | Status | Who |
|---|---|---|
| Native MLA, absorbed formulation, offline-fusible latent transform, dual-path (content vs RoPE) error model, INT4/INT2 latent kernel | **Done** (learned rotation+scale, gradient objective; no bit allocation) | QuantMLA 2609.36760 |
| Closed-form transform T = U^T M^{1/2} with M = query second moment; weighted MSE → plain MSE (generalized Parseval); head-summed M for shared KV | **Done for MHA/GQA keys** (Theorem 1 + eq. 38) | NOVA-KV 2608.04074 |
| Output-side weighting by W_O column norms + reverse water-filling over decorrelated channels; attention-aware distortion theorem | **Done for MHA/GQA** (diagonal weights, not full M) | AATC 2608.14191 |
| Activation-weighted SVD latent + water-filling with zero-bit channels, folded into attention | **Done for GQA** (input-side weighting G=E[h^T h]) | SPECTRA 2608.07915 |
| Query-covariance eigenbasis beats Hadamard at INT2 | Done (MHA; also ran GLM-4.7 but on expanded head-dim K/V) | OSCAR 2605.17757 |
| RoPE branch should be high precision | Asserted empirically (SnapMLA), **contradicted under QAT** (DeepSeek-V4.1-Flash: same FP4 format for RoPE and non-RoPE) | 2602.10718 / 2609.19969 |

What is genuinely still open after this sweep:
1. The **joint** K+V weighting on a *shared* latent: M = M_K + λ M_V with both terms acting on the same cached vector. NOVA-KV treats keys and values separately ("Joint quantization of keys and values is left for future work", l.1255); AATC/SPECTRA weight channels diagonally; QuantMLA learns the transform numerically and never writes M in closed form (its H_b = J_b^T J_b is estimated by JVPs, App. B.3).
2. **Closed-form KLT + reverse water-filling across latent directions** in MLA (variable bits, zero-bit directions). QuantMLA uses uniform INT4/INT2 per path with per-token group-64 asymmetric affine quantizer (l.1130). Nobody has measured the effective rank of M^{1/2} Σ_c M^{1/2} for DeepSeek-V2-Lite / GLM-4.7-Flash / Kimi.
3. **Learned (QuantMLA) vs closed-form (KLT) transform** head-to-head on MLA — a direct, cheap experiment.
4. **Per-token vs per-channel scale on the 64-dim RoPE branch** as the discriminator between SnapMLA (BF16) and FlashMLA-V4.1 (FP8/FP4 incl. RoPE). QuantMLA's RoPE path already uses learned per-frequency-pair tied scales S_P plus per-token group scale (Prop. 1), i.e. a static per-channel-pair rescaling — so "per-channel" is partially realised there, but never isolated as an ablation.

Risk: the OSCAR/NOVA-KV/SPECTRA authors and the QuantMLA (SnapMLA) group are all one step away from the remaining gap. Idea 10 should be reframed as "closed-form, jointly-weighted, rate-allocated version of QuantMLA" and must compare against QuantMLA, not against RTN/QuaRot.

## Key papers (fetched, read)

### 1. QuantMLA — 2609.36760 (29 Sep 2026, "Under review") — partially_preempts, P1
Zunhai Su et al. (Su is also a SnapMLA co-author). Native MLA.
- Absorbed formulation (eq. 1): S_h = τ Q̄_{C,h} C^T + Q_{P,h} K_P^T, Y = Σ_h A_h C B_h with Q̄_{C,h} = Q_{C,h} W_{K,h}^T, B_h = W_{V,h} W_{O,h}. Identical to the idea's a^h / W_UV setup.
- Sec 2.2 (empirical): matched cache NMSE 0.01, "the ratio of model-level mean responses between the RoPE and content paths ranges from 1.941× to 6.864×" over DeepSeek-V2-Lite, V3-Base, R1, Kimi-K2-Instruct, LongCat-Flash-Lite, GLM-4.7-Flash.
- Sec 2.3 (model): G_b(e_b) = e^T H_b e / e^T e with H_b = J_b^T J_b (Jacobian of output w.r.t. cached state), exact decomposition G_b = T_b A_b + O_b. This is the weighted-MSE object of Theorem 10.1, but estimated numerically (App. B.3, JVP vs finite perturbation Spearman 0.986) — **no closed form in terms of a^h and W_UV**.
- Sec 3.1 (transform spaces): content: z_C = u R_C S_C with R_C orthogonal, S_C positive diagonal, fused via RMSNorm rotation-equivariance; consumers W̃ = S_C^{-1} R_C^T Γ W. RoPE: Prop. 1 — R_P = blockdiag(SO(2) per frequency pair), S_P = blockdiag(s_i I_2), fused pre-RoPE. **Transform is orthogonal+diagonal, not a general M^{1/2}-type non-orthogonal KLT.**
- Sec 3.2 (objectives): content minimises attention-output reconstruction (eq. 8) by gradient (Cayley parameterisation, 300 steps, 128 WikiText-2 seqs, Table 7); RoPE minimises positional-QK reconstruction (eq. 10); Prop. 2: ||ΔY_P||_F^2 ≤ (τ²/4) Σ_h ||V̄_h||_2^2 Σ_h ||Ω⊙ΔZ_{P,h}||_F^2.
- Quantizer: "All quantized methods use the same per-token asymmetric affine quantizer with group size 64" (l.1130). No per-channel bit allocation, no water-filling (grep: none). Mixed precision only as sink4/local128 BF16 protection.
- Results (Table 1, DeepSeek-V2-Lite): BF16 GSM8K 36.92; C4R4 RTN 16.98, SmoothQuant† 24.56, QuaRot† 30.71, QuantMLA 35.25. C2R4 QuantMLA GSM8K 37.91 (RTN 3.71). S-NIAH-1/2 avg 99.95/99.60 %, S-NIAH-3 87.20 % at C4R4 (4K–32K). "to our knowledge, the first reported joint INT4 caching of the content and RoPE caches".
- Confirms FlashMLA V3.2 layout = FP8 content / BF16 RoPE (l.317).
- Gap it leaves: closed-form transform; variable bit allocation; learned-vs-KLT comparison; explicit per-token vs per-channel RoPE ablation.

### 2. NOVA-KV "Spend Bits Where Queries Look" — 2608.04074 (4 Aug 2026) — partially_preempts, P1
- Criterion (eq. 5): Σ_j (k_j − k̂_j) M_q (k_j − k̂_j)^T with M_q = Q^T Q; "input-weighted quadratic distortion with constant sensitivity matrix M_q (Linder et al., 1999)"; "Orthogonal transforms are suboptimal for the high-resolution regime unless M_q ∝ I."
- Theorem 1: R*_{K,p} = M_q^{1/2} E_{1:p}, E from eigendecomposition of M_q^{1/2} S̃_k M_q^{1/2}. **This is exactly Idea 10's T = U^T M^{1/2} / eigen(M^{1/2} Σ_c M^{1/2}) for the key side.** Prop. 1: ||r − r̂||_2^2 = ||k − k̂||_{M_q}^2 (generalised Parseval).
- App. (eq. 38) GQA: logit error = Σ_j (k_j−k̂_j) Σ_h M_q^{(h)} (k_j−k̂_j)^T, "same form as (5) with M_q = second moment over the heads sharing each KV head". **So a head-summed M for a shared cache is already published; the idea's claim that head-summation is structurally MLA-specific is wrong — it is the GQA case with H→all heads and k→c.**
- Corollary 1: value transform from score-weighted covariance; Theorem 2: equal-volume grouping for fixed-rate VQ. Keys use VQ (256 centroids, dim 4, 2 bits/coord); values use OSCAR SQ.
- Not MLA; not weight-folded on the key side (value inverse "absorbed into the output projection", l.1923); "Joint quantization of keys and values is left for future work" (l.1255).

### 3. OSCAR — 2605.17757 (18 May 2026) — supports / partially_preempts, P2
- Theorem 1: with frozen diagonal residual covariances, R_k = U_Q (eigvecs of C_Q = (1/N) Σ q_n^T q_n) and R_v = U_S (eigvecs of C_S = (1/N) V^T S^T S V) minimise tr(R^T C R E). Composed with Hadamard + bit-reversal, INT2.
- Supports the premise "query-covariance-weighted rotation beats Hadamard" (Qwen3-8B BF16 gap 1.42 pts vs rotation-only collapse).
- Ran GLM-4.7-FP8 (an MLA model) but "d is the head dimension" (l.241), "quantization is applied to the head dimension with block-size 128, 64 or 32" (l.285) — i.e. expanded per-head K/V, not the latent; MLA never discussed.
- Limitations (l.4177): "OSCAR uses a fixed bit-width ... does not study per-token bit allocation", does not explore NVFP4; theory "not yet a full proof of optimality for the complete autoregressive decoding process".
- Cites CARE [59] (same authors) — see below.

### 4. AATC "KV Cache Compression Through the Lens of Transform Coding" — 2608.14191 (14 Aug 2026) — partially_preempts, P2
- Theorem 1 (white-noise + bounded error): D = D_K + D_V + R, D_K = Σ_{i,c} a_i² ||v_i W_O − o||² q_c² E[(δk_ic)²]; D_V weighted by a_i² and W_O column norms; remainder bound |R| ≤ C||q||²||W_O||_F² σ_K² σ_V² + ....
- Transform: Cholesky-whitened SVD of W_K/W_V (input covariance Σ_X), cache H = XA, decorrelated channels ordered by singular values (= PCA of K, same as PALU/KVTC); then **reverse water-filling** with attention-aware per-channel weights (q_c², W_Oc). "To our knowledge, no prior method exposes W_Oc as an explicit allocation factor" — this is the diagonal version of the idea's M_V = Σ_h W_UV^{hT} W_UV^h.
- Explicitly positions against OSCAR: "query-awareness through the rotation versus through the allocation ... applying both double-counts it. However, one could combine OSCAR's query-aware rotation with channel-wise bit allocation." → Idea 10's full-matrix M does both consistently; that is the open slot.
- Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct; ~5.8× near-lossless. Not MLA.

### 5. SPECTRA — 2608.07915 (8 Aug 2026) — partially_preempts, P2
- Activation-weighted SVD (G = E[h^T h]) → latent cache; "Water-filling bit allocation"; "A channel assigned zero bits is simply dropped, so this single allocation spans both quantization and rank reduction"; transform "folds into attention" (up-projection folds into query, closed-form W_O correction, l.382–387, l.447).
- Weighting is input-side (G), not downstream (query/output). GQA models only; mentions MLA: "SPECTRA targets the same latent-cache idea but as [post-hoc]" (l.1570–1572). Leaves "a fused, end-to-end kernel ... to future work".

### 6. DeepSeek-V4.1-Flash report — 2609.19969 (17 Sep 2026) — contradicts (RoPE premise) / supports (deployment model), P1
- §2.4.4 "FP4 Main KV Cache": E2M1 with one E4M3 scale per 16 channels, NVFP4 minus global scale; "we introduce QAT during post-training. The non-RoPE and RoPE components use the same quantization format. We quantize the cache after RoPE: quantizing before RoPE yields only a marginal accuracy improvement in our experiments"; "We retain FP8 for the SWA KV cache due to its sensitivity to quantization"; latent L2 norm ≤ √512 ≈ 22.6 after RMSNorm (γ≈1), observed max ≈ 10.
- "FP4 reduces storage rather than accelerates matrix multiplication. Dequantizing cached values before attention allows us to use a more accurate format" — same deployment model a folded transform would use.
- Only qualitative accuracy ("only marginal performance degradation", l.393–394); no FP4-KV ablation table found (Table 1 is base-model comparison).
- Consequence for Idea 10: FlashMLA's 288 B/token FP4 (V4.1) is a **QAT'd** format; the PTQ story must target BF16-trained MLA models (DeepSeek-V2-Lite, GLM-4.7-Flash, Kimi-K2, LongCat-Flash-Lite, Moonlight). The "RoPE must be high precision" claim is contradicted at least under QAT.

### 7. SnapMLA — 2602.10718 (11 Feb 2026, v2 28 Apr 2026) — supports, P2
- §3.1.1: RoPE component "spans a significantly wider dynamic range (reaching ±10³) ... while the content component is tightly concentrated around zero (within ±10¹)" (LongCat-Flash-Thinking, Fig 3a); FP8 "leads to an order-of-magnitude increase in MSE for the RoPE component" (Fig 3b) → RoPE kept BF16, content per-token FP8 E4M3.
- Per-token granularity justified by decode "page tail" and framework compatibility, **not** by accuracy theory; per-channel never tested. Table 1 near parity on DeepSeek-V3.1 and LongCat-Flash-Thinking.

### 8. CARE — 2603.17946 (ICLR 2026) — tangential / supports, P3
- Covariance-aware factorisation: SVD on C^{1/2} W then unwhiten (C = input activation covariance); rank allocation "via an energy-driven, water-filling schedule" across layers. Same KLT+water-filling machinery, but for **rank** in GQA→MLA conversion, not for quantisation bits.
- "CARE is orthogonal to quantization ... does not yet co-optimize KV quantization" (l.2824, 2837). Fig 2a: DeepSeek-V2-Lite layer-wise rank sensitivity — useful for the effective-rank conjecture.

### 9. Block-GTQ "RoPE-Aware Bit Allocation" — 2606.24033 (23 Jun 2026) — supports (RoPE branch), P3
- Greedy integer bit allocation over 2-D RoPE frequency blocks by label-free energy score; K2V2 Llama-3.1-8B NIAH 70.6→97.4. Includes DS-V2-Lite but "the latent is outside its scope and the diagnostic uses only the decoupled RoPE-key path" (l.3237). Direct prior for frequency-band allocation on the 64-dim RoPE branch.

### 10. RateQuant — 2605.06675 (22 Apr 2026) — tangential (caveat), P4
- Reverse water-filling across heads, but per-quantizer distortion law D(b)=αβ^{-b} with β 3.6–5.3 ≠ 4 ("distortion model mismatch ... inverts the allocation order"). Caveat for Idea 10's b_i = ½log₂(λ_i/θ): the high-resolution 2^{-2b} law must be calibrated per quantizer.

### 11. KV-COBRA — 2609.24298 (21 Sep 2026) — tangential (competitor warning), P4
- Per-head joint rank/bit allocation; Hadamard + attention-KL-reordered SVD basis (query-aware); "Natural extensions include MLA and MoE architectures" (l.1301).

### 12. Agent Memory Below the Prompt — 2603.04428 (17 Feb 2026) — tangential, P5
- Systems paper (Apple M4, MLX). "Perplexity ... +2.8% for Llama, and +3.0% for DeepSeek[-Coder-V2-Lite]" with Q4 KV. Not a sensitivity study; QuantMLA §2.2 supersedes it as MLA sensitivity evidence.

### 13. KVarN — 2606.03458 (2 Jun 2026) — tangential, P5
- Hadamard + dual-axis variance normalisation, 2-bit, calibration-free. Paper text contains no GLM-4.7-Flash / MLA result (only l.1903 "multi-head latent attention (MLA) ... it is unclear how such attention..."). The idea's "GLM-4.7-Flash k4v2 AIME25 tie" must be sourced from the repo README, not the paper.

### 14. SVDq — 2502.15304 (21 Feb 2025) — tangential / early prior, P4
- SVD latent channels of the K cache + importance-aware mixed precision (1.25-bit equivalent), Theorem 4.1 (Eckart–Young variance decay). Weight-space SVD, not query-weighted; MHA/GQA. Earliest "KLT + variable precision for keys" prior found.

### 15. FlashMLA README (github.com/deepseek-ai/FlashMLA, fetched 2026-10-10) — supports (format facts), P3
- V4.1 FP8: 528 B/token = 512 e4m3 + 16 e8m0 scales (one per 32); "The 64 RoPE dimensions are quantized as well, so there is no bfloat16 part". FP4: 288 B/token = 256 B e2m1 + 32 e4m3 scales (one per 16). "Unquantized (bfloat16) KV cache format is not supported." No accuracy statements; 2026.09.30 release dropped V3.2/V4.0 formats. Confirms the idea's numbers.

## Also seen (not in top 15)
- SAW-INT4 2604.19157: block-diagonal Hadamard INT4, evaluated on GLM-4.7-FP8 (expanded K/V), BDR-128 69.97 vs BF16 70.84.
- STAR-KV 2606.08382: §3.3 weight absorption discussion; low-rank-aware mixed-precision with block-wise Hadamard (MHA/GQA).
- KVTC 2511.01815 (ICLR 2026): PCA + adaptive quantisation + entropy coding for storage; mentions MLA only in related work.
- Codec-Gauge 2607.20538: learned orthogonal gauges; "Some recovery operations can be folded into downstream projections; key-side fusion depends on positional encoding".
- JoLT 2607.12550, RDKV 2605.08317, MixKVQ 2512.19206, Palu 2407.21118, SQuat 2503.24358: MHA bit/rank allocation or query-subspace error shaping; none MLA.
- HeadQ 2605.03562: withdrawn by author.
- VestigeKV 2609.03949: NoPE-MLA (Kimi Linear) 64-dim decoupled branch as salience channel; no quantisation.
- Thin Keys 2603.04427, TransMLA 2502.07864, GQLA 2605.15250, Through-the-Bottleneck 2607.23054: MLA architecture/conversion, no latent quantisation theory.
- NVIDIA TensorRT-LLM blog (web): DeepSeek-R1 GSM8K BF16 MLA+KV 0.9629 vs FP8 MLA+KV 0.9613 — an FP8-latent accuracy datum the idea says does not exist (industrial, not peer-reviewed).

## Suggested revisions to Idea 10
1. Rewrite "문헌상 공백": cite QuantMLA as the direct predecessor; cite NOVA-KV Thm 1 + eq. 38 as the closed-form/head-sum prior; cite AATC/SPECTRA for W_O-weighting and water-filling; drop the claim that latent transforms have no literature.
2. Keep as novelty: (a) joint M_K + λ M_V on a shared latent with closed-form KLT; (b) water-filling across latent directions (zero-bit directions, sub-4-bit); (c) learned (QuantMLA) vs KLT comparison; (d) effective-rank measurement of M^{1/2} Σ_c M^{1/2}.
3. RoPE branch: replace "FlashMLA vs SnapMLA contradiction" with the three-way fact pattern — SnapMLA BF16 (PTQ, per-token), QuantMLA INT4 with pair-tied learned scales (PTQ), V4.1-Flash FP4 same format (QAT) — and make the per-token-vs-per-channel ablation the discriminator.
4. Baselines for the judging experiment must include QuantMLA C4R4/C2R4 numbers above and the FP4 E2M1/g16 format from V4.1-Flash.

## Queries run
arXiv API (scripts/arxiv_search.py):
- abs:"multi-head latent attention" AND abs:(quantization OR quantized OR quantize)
- abs:MLA AND abs:"KV cache" AND abs:(quantization OR compression) AND abs:latent
- abs:DeepSeek AND abs:"KV cache" AND abs:(quantization OR quantized)
- abs:"weight absorption" AND abs:attention  (0 hits)
- abs:latent AND abs:"KV cache" AND abs:(quantization OR quantized) AND abs:(transform OR rotation OR rotate)
- abs:"transform coding" AND abs:"KV cache"
- abs:"weighted MSE" AND abs:quantization AND abs:attention  (0 hits)
- abs:"low-rank" AND abs:"KV cache" AND abs:(quantization OR quantized) AND abs:latent
- abs:"KV cache" AND abs:(PCA OR SVD OR "principal component" OR Karhunen OR KLT) AND abs:(quantization OR quantized) AND abs:(latent OR "low-rank" OR "low-dimensional")
- abs:"bit allocation" AND abs:"KV cache"
- abs:(MLA OR "latent attention") AND abs:(FP8 OR FP4 OR INT4 OR "low-bit" OR "4-bit") AND abs:(cache OR KV)
- abs:"covariance-aware" AND abs:"latent attention"
- abs:"latent attention" AND abs:(covariance OR "query statistics" OR "query distribution")
- abs:("water-filling" OR "water filling" OR "reverse water-filling") AND abs:(KV OR cache OR attention) AND abs:quantization
- abs:("attention-aware" OR "query-aware" OR "attention-preserving") AND abs:"KV cache" AND abs:(quantization OR quantized)
- abs:(MHA OR GQA) AND abs:"latent attention" AND abs:(convert OR conversion OR upcycl OR transform)
- abs:"DeepSeek-V2-Lite" AND abs:(quantization OR quantized OR "KV cache")
- abs:(Kimi OR "GLM-4.7" OR LongCat OR Moonlight) AND abs:"KV cache" AND abs:(quantization OR quantized)
- abs:(MLA OR "latent attention" OR "latent KV") AND abs:(Hadamard OR rotation OR rotated OR orthogonal) AND abs:(quantization OR quantized OR "low-bit")
- abs:("weighted quantization" OR "weighted distortion" OR "sensitivity matrix" OR "Mahalanobis") AND abs:(attention OR "KV cache") AND abs:(LLM OR "language model")
- abs:(FP4 OR NVFP4 OR "4-bit floating") AND abs:"KV cache" AND abs:(DeepSeek OR MLA OR latent)
- abs:"low-rank projection" AND abs:"KV cache" AND abs:(quantization OR quantized) AND abs:(Hadamard OR latent)
- ti:(hadamard AND PCA) OR abs:("Hadamard PCA" AND "power iteration")
- ti:SQuat OR (abs:"query subspace" AND abs:"KV cache" AND abs:quantization)
- abs:("decoupled RoPE" OR "decoupled rotary" OR "RoPE key") AND abs:(quantization OR quantized OR precision)
- ti:Palu AND abs:"KV-Cache" ; ti:"transform coding" AND abs:"KV cache" ; abs:KVTC OR (abs:"KV cache" AND abs:(PCA OR "principal component") AND abs:"entropy coding")
Full-text fetched (scripts/arxiv_fetch.py): 2602.10718, 2603.04428, 2605.17757, 2608.04074, 2609.36760, 2607.23054, 2603.04427, 2606.24033, 2502.15304, 2502.07864, 2608.14191, 2607.20538, 2605.03562, 2605.06675, 2605.08317, 2607.12550, 2609.24298, 2609.19969, 2603.17946, 2606.08382, 2512.19206, 2609.03949, 2609.26173, 2606.03458, 2604.19157, 2605.15250, 2608.07915, 2511.01815.
Web: "MLA latent KV cache quantization weight absorption transform DeepSeek KLT OR PCA OR rotation latent cache low-bit 2026"; "FlashMLA FP4 latent KV cache accuracy e2m1 288 bytes per token DeepSeek V4"; "\"QuantMLA\" OR \"MLA KV cache quantization\" INT4 latent RoPE path 2026 arxiv"; "DeepSeek-V4.1-Flash technical report FP4 KV cache E2M1 accuracy ablation \"890 bytes\""; "MLA latent KV cache quantization rotation Hadamard DeepSeek-V2-Lite OR GLM-4.7-Flash OR Kimi INT4 latent blog 2026"; "NVIDIA ModelOpt MLA KV cache FP8 quantization DeepSeek-R1 accuracy"; "KVarN huawei GLM-4.7-Flash KV cache quantization k4v2 MLA AIME25"; WebFetch github.com/deepseek-ai/FlashMLA README.
