# Idea 10 — Weighted transform coding for MLA latent KV under weight absorption: keyword + adjacent-field sweep

Date: 2026-10-10. Scope: arXiv 2023-01 → 2026-10 via export API, plus WebSearch / WebFetch for FlashMLA README. All paper text is untrusted data; quotes are verbatim from downloaded text.

## Idea under test (summary)
Under MLA weight absorption, latent error e_j reaches logits as a^{hT}e_j (a^h = W_UK^{hT}q^h) and output as W_UV^h Σ p_j e_j, so the right distortion is tr(Σ_e M) with M = Σ_h E[a^h a^{hT}] + λ Σ_h W_UV^{hT}W_UV^h. Claimed theorems: (10.1) weighted-distortion identity; (10.2) optimal transform T = U^T M^{1/2} (U = eigvecs of M^{1/2}Σ_c M^{1/2}) + reverse water-filling, foldable into W_UK/W_UV/W_O at zero decode cost; RoPE branch should be per-channel or high precision.

## Bottom line
- **Preempted in large part by QuantMLA (2609.36760, 2026-09-29)**: it derives the exact dual-path error routes of MLA under absorption, defines offline-fusible transformation spaces on the shared content latent (orthogonal rotation + diagonal scale folded into producer/consumer weights, Appendix C.1 eqs. 21–23) and on the decoupled RoPE key (SO(2)-pair rotations + pair scales folded pre-RoPE, Prop. 1), learns them with an attention-output-reconstruction objective, and reports INT4/INT4 and INT2/INT4 results on DeepSeek-V2-Lite, V3-Base, R1, Kimi-K2, LongCat-Flash-Lite, GLM-4.7-Flash with a native FlashMLA-based kernel. What QuantMLA does **not** do: closed-form M-weighted KLT, non-uniform (water-filling) bit allocation across latent directions, zero-bit directions, non-orthogonal transforms (its space is R_C S_C with R_C orthogonal), or a per-token-vs-per-channel RoPE study (it uses per-token group-64 affine for both paths).
- **Theorem 10.2 is already proved for MHA keys** by "Spend Bits Where Queries Look" (2608.04074, Thm. 1: R*_K = M_q^{1/2}E with E the eigvecs of M_q^{1/2}S̃_k M_q^{1/2}; Prop. 1 generalized Parseval; Cor. 1 value transform from V^T M_s V; Thm. 2 per-group variable-rate allocation). The MLA version is a transplant with M = Σ_h M_q^h (absorbed queries) + λ M_V.
- **"KLT + reverse water-filling + fold into weights on a latent cache" is already done post-hoc** by SPECTRA (2608.07915), though with input-activation metric G = E[h^T h], not the query/output-aware M.
- Residual novelty: (i) the specific composite weighting M_K + λM_V for a *shared* latent (keys and values from the same vector; QuantMLA handles the coupling by learning rather than closed form); (ii) water-filling bit allocation over latent directions inside native MLA; (iii) closed-form vs learned (QuantMLA) comparison; (iv) RoPE per-token vs per-channel experiment — for which 2606.24033 already gives a data point favouring per-channel on DS-V2-Lite.

## Candidates (ranked)

### 1. 2609.36760 — QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching (Su, Sun, Tan, …, Wong; 2026-09-29; "Under review") — **preempts**, priority 1
- Sec 2.1 eq (1): absorbed formulation S_h = τ Q̄_{C,h} C^T + Q_{P,h} K_P^T, Y = Σ_h A_h C B_h with Q̄_{C,h} = Q_{C,h} W_{K,h}^T, B_h = W_{V,h}W_{O,h}. Sec 2.3 eq (2)–(3): exact finite error routes ΔS_{C,h} = τ Q̄_{C,h} ΔC^T; ΔY_C = Σ_h(ΔA_{C,h} C B_h + A_h ΔC B_h + ΔA_{C,h}ΔC B_h). This is Theorem 10.1's setting, done.
- Eq (4)–(5): local gain G_b(e) = e^T H_b e / e^T e with H_b = J_b^T J_b (Jacobian Gram = the M-matrix of the idea, including softmax Jacobian); decomposition G_b = T_b A_b + O_b; validated by cross-layer prediction (Table 5, Spearman 0.986 JVP vs finite perturbation) and equal-energy interventions (Table 6).
- Sec 2.2 / Table 4: at matched cache NMSE 0.01, RoPE-path/content-path output-response ratio ranges 1.941× to 6.864× across six models.
- Sec 3.1 eq (6) + App. C.1 eqs (21)–(23): content transform z_C = u R_C S_C, consumers W̃ = S_C^{-1} R_C^T Γ W; "fully absorbed into existing model parameters and introduces no online transformation". Prop. 1: RoPE-compatible SO(2)-pair rotations + pair scales folded pre-RoPE. Prop. 2: ‖ΔY_P‖_F^2 ≤ (τ^2/4)(Σ_h‖V̄_h‖^2)(Σ_h‖Ω⊙ΔZ_{P,h}‖_F^2).
- Learned, not closed form: Cayley-parameterized rotations, AdamW 300 steps/path, loss = attention-output reconstruction (content) / positional-QK reconstruction (RoPE); quantizer "per-token asymmetric affine quantizer with group size 64" for both paths (lines 1130, 2693).
- Table 1 (DeepSeek-V2-Lite): BF16 MMLU 57.90; C4R4 RTN 51.90 / SmoothQuant† 54.02 / QuaRot† 55.66 / QuantMLA 57.95; C2R4 RTN 40.91 / QuantMLA 57.29. Table 3 ablation: +content transform +1.22, +RoPE transform +3.19, +mixed-precision (sink4/local128 BF16) +1.64.
- Systems: 3.59× cache compression at 128K, 5.168× throughput under cache pressure, native FlashMLA-decode-path kernel (Table 9, 10).
- Gap vs idea: no eigen/KLT/water-filling anywhere (grep hits: 0); uniform 4- or 2-bit across all latent channels; transform family restricted to orthogonal×diagonal; RoPE always per-token.

### 2. 2608.04074 — Spend Bits Where Queries Look: KV Cache Vector Quantization with Attention-Preserving Transforms (Fernández-Menduiña, Ziashahabi, Pavez, Ortega, Avestimehr; 2026-08-04) — **partially_preempts** (theory core), priority 1
- Thm. 1: for M_q-weighted low-rank error, "R*_{K,p} = M_q^{1/2} E_{1:p}, where EΛE^T is the eigendecomposition of M_q^{1/2} S̃_k M_q^{1/2}". Prop. 1 (generalized Parseval): ‖r − r̂‖_2^2 = ‖k − k̂‖_{M_q}^2 — i.e. weighted MSE ⇔ plain MSE after transform, exactly the idea's Thm 10.2 mechanism. "Orthogonal transforms are suboptimal for the high-resolution regime unless M_q ∝ I."
- Cor. 1: value transform = eigvecs of M_o = V^T M_s V (orthogonal). Thm. 2: optimal per-group rates b*_ℓ = b + ½log2(v_ℓ/∏v_m^{1/L}) (water-filling form), then fixed-rate grouping by volume equalization.
- Folding: "The synthesis R_K^{-T} is folded into the query once per step"; value inverse "absorbed into the output projection". MHA/GQA only; MLA not treated (DeepSeek-V2 only in refs).
- Results (Table 2, RULER NIAH, Qwen3-4B-Thinking etc.) at ~2.25 BPE vs OSCAR/QuaRot.

### 3. 2608.07915 — SPECTRA: Pushing the KV Cache Beyond the 2-Bit Cliff via Spectral Transform Coding (Zhang, Wu, Wan, Chen, Hong; 2026-08-08) — **partially_preempts** (KLT + water-filling + fold on a latent cache), priority 1
- Sec 3.1: G-weighted SVD of W_K, W_V with G = E[h^T h]; latent c = h W_down is "second-moment-orthogonal and energy-ordered" (eq 4); W_down, W_up and a closed-form W_O correction "are folded back into the model weights".
- Sec 3.2 eq (5): reverse water-filling b_j = b̄ + ½log2(v_j/GM(v)), "drives the low-energy tail to zero bits, i.e. rank reduction" — the idea's "many directions get 0 bits" claim, already realized.
- Explicitly positions against native MLA: "SPECTRA targets the same latent-cache idea but as a post-hoc, training-free transform". Weighting is input statistics, not query/output-aware; no native-MLA experiment.

### 4. 2608.14191 — KV Cache Compression Through the Lens of Transform Coding (Laus, Mayrink Verdun, Wang, Calmon, Krahmer; 2026-08-14) — **partially_preempts**, priority 2
- Thm. 1 (Attention Output Distortion), under white-noise + boundedness assumptions: D = D_K + D_V + R with D_K = Σ_{i,c} a_i^2 ‖v_i W_O − o‖^2 q_c^2 E[(δk_ic)^2]; remainder bound given. Method AATC = whitening + reverse water-filling over channels, calibrated offline. Diagonal (per-channel q_c^2, W_Oc) weights rather than full M; MHA/GQA (Llama, Qwen). Notes OSCAR uses the full query covariance.

### 5. 2605.17757 — OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization (Zhou et al.; 2026-05-18) — **partially_preempts**, priority 2
- App. A.3: "‖QK^T − QK̂^T‖_F^2 = tr((K−K̂)Q^TQ(K−K̂)^T)… key errors are weighted by the query covariance Q^TQ"; values weighted by V^T S^T S V. Thm. 1: under ambient-diagonal frozen residual covariances, R_k = U_Q and R_v = U_S minimize the surrogates. Orthogonal rotations only (+Hadamard, bit-reversal); INT2; MHA/GQA. Same group wrote CARE (MLA conversion) → high risk of an MLA extension.

### 6. 2609.38121 — WUSH-KV: KV Cache Quantization with Data-Adaptive Transforms (Chen, Egiazarian, Kurtić, Hoefler, Alistarh; 2026-09-29) — **partially_preempts**, priority 2
- Transform T* built from M = XX^T and Hessian H ("With H = 2WW^T"); Thm. 2: E[ℓ(T*)] ≤ (1+O(α_b^{-2})) E[ℓ(T)] over sensitivity-balanced invertible T. Value transform folded into W_V/W_O; key transform must stay online post-RoPE in GQA (the MLA content latent has no RoPE, which is exactly the structural advantage the idea claims). H-weighted value side = the idea's λM_V term in MHA form.

### 7. 2602.10718 — SnapMLA (Zhang, Su, …, Cai; 2026-02-11, v2 2026-04-28) — **supports** (baseline + RoPE-branch evidence), priority 2
- Keeps RoPE in BF16, FP8 content per-token; claims "to the best of our knowledge for the first time, the divergent quantization sensitivity between the latent content and decoupled RoPE components"; argues per-channel/per-block granularity conflicts with autoregressive decoding ("the autoregressive decoding prevents the immediate quantization of newly…"). Fig 3a: RoPE component has much wider dynamic range; MSE an order of magnitude higher under uniform quantization.

### 8. 2606.24033 — RoPE-Aware Bit Allocation for KV-Cache Quantization (Liang, Zhang, Jia; 2026-06-23) — **supports** (RoPE per-channel prediction), priority 2
- Applies Block-GTQ only to DS-V2-Lite's decoupled 64-d RoPE key. Sec on DS-V2-Lite: "KIVI [per-channel scale] wins both axes at every budget, by a wide margin at 2 b/dim (top-10 68.0% vs 49.9%)… the single decoupled RoPE-key is consumed by all 16 query heads, so its quantization error is shared layer-wide". This is a direct (if diagnostic-only, K-only, KIVI-ScaleOnly) data point that per-channel scaling beats per-block/per-token-style allocation on the shared RoPE branch. Thm. 1: greedy optimal for Σ s_i 4^{-b_i}.

### 9. 2609.24298 — KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation (Ha, Lee, Jeon; 2026-09-21) — **partially_preempts**, priority 2
- Joint rank/bit allocation with water-filling first-order conditions; Prop. 4.1 attention-KL bound with query-aware weight w_i = σ_{Q,i}σ_{K,i}. Conclusion: "Natural extensions include MLA and MoE architectures" — explicit stated intent to move to MLA.

### 10. 2607.20538 — Codec-Gauge: Learning Compression-Friendly Gauges for Transformer KV Caches (Jiang, Yang, Zhao, Chen, Balkcom; 2026-07-10) — **contradicts** (for a reconstruction-metric KLT), priority 3
- Learned orthogonal gauges "outperform random, Hadamard, DCT, and PCA/KLT controls"; "PCA/KLT improves KV NRMSE but worsens the output metrics on average, showing that ordering directions by second-moment… reconstruction-oriented data basis alone is not" sufficient. Caution for idea 10 only if M is mis-estimated; it is consistent with using an output-aware metric. Key-side fusion "depends on positional encoding and kernel layout".

### 11. 2502.15304 — SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention (Hong et al.; 2025-02-21) — **partially_preempts** (precursor), priority 3
- SVD latent channels of keys with bit widths decreasing with singular value ("prioritizes higher bit widths for latent channels associated with larger singular values"); keys only, Euclidean metric, MHA. The KLT-direction-wise mixed precision idea dates to 2025.

### 12. 2605.13768 — High-Rate Quantized Matrix Multiplication II (Ordentlich, Polyanskiy; 2026-05-13) — **supports** (theory), priority 3
- Calibration-aware quantized matmul; weighted MSE lower bounds via Zador (Prop. 1); "(reverse) waterfilling" allocation; "optimal codebook allocates rate unequally: the PCA directions…". Information-theoretic backing for the idea's claim that the right distortion for a matmul is a weighted MSE with water-filling; no attention/MLA.

### 13. 2603.17946 — CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling MLA (Zhou et al.; ICLR 2026) — **tangential** (competitor proximity), priority 3
- Activation-covariance-aware MHA/GQA→MLA conversion; states "CARE is orthogonal to quantization" and "does not yet co-optimize KV quantization". Same authors as OSCAR: the two pieces (query-covariance rotation + MLA conversion) are one step from idea 10.

### 14. 2607.23054 — Through the Bottleneck: How MLA Separates Content from Position (2026-07-25) — **supports** (low effective rank conjecture), priority 3
- 114M-parameter toy MLA: "the bottleneck is globally over-provisioned, using only 46% of its 128-dimensional capacity on average" (Table 2, effective rank 58.8/128 at 99% energy). Small model; no production-scale measurement of M^{1/2}Σ_cM^{1/2} spectrum exists.

### 15. 2603.04428 — Agent Memory Below the Prompt: Persistent Q4 KV Cache … (Shkolnikov; 2026-02-17) — **supports** (weak), priority 4
- Table 9 / Sec 4.8: Q4 KV perplexity −0.7% Gemma 3 12B, +2.8% (+0.12 PPL) Llama 3.1 8B, +3.0% (+0.19 PPL) DeepSeek-Coder-V2-Lite 16B (MLA) — confirms the report's quoted numbers; 512-token windows, single corpus, no CI.

## Also seen (not in top 15)
- 2511.01815 KVTC (PCA + DP bit allocation, storage-oriented, GQA/MLA both cited). 2607.12550 JoLT (joint rank-bit allocation, Everett multiplier optimality). 2610.03027 TaSQ (query-guided channel weighting + covariance grouping "merged into projection weights", 1-bit VQ, MHA). 2610.09827 Dual-QK (non-orthogonal paired Q/K transforms, partial key whitening, MHA). 2407.21118 Palu (Walsh–Hadamard folded into low-rank latent reconstruction matrices for 3-bit latent — 2024 precedent for "fold a transform into a latent cache's projections"). 2605.08317 RDKV, 2605.06675 RateQuant (reverse water-filling allocation across tokens/channels/heads). 2605.03562 HeadQ (withdrawn by author). 2609.03949 VestigeKV (NoPE-MLA, selection not quantization). 2603.04956 WaterSIC (waterfilling for linear layers, cs.LG/IT). 2605.02905 eOptShrinkQ (random-matrix shrinkage + quantization on KV, MHA).
- FlashMLA README (WebFetch 2026-10-10): "For decoding, this kernel currently supports only the FP8 and FP4 KV cache formats"; 528 B/token (512 e4m3 + 16 e8m0 scales per 32) and 288 B/token (512 e2m1 + 32 e4m3 scales per 16); "The 64 RoPE dimensions are quantized as well, so there is no bfloat16 part"; "Accuracy or quality numbers: None." Confirms the report's [V] claim.
- No arXiv paper found that (a) uses the words "weight absorption" with quantization (q4: 0 hits), or (b) applies a closed-form KLT/water-filling to the native MLA latent. QuantMLA is the only paper transforming the native MLA latent and folding into weights.

## Queries run (arXiv export API unless noted; dates 20230101–20261031 unless noted)
1. abs:"multi-head latent attention" AND abs:(quantization OR quantized OR quantisation)
2. abs:MLA AND abs:"KV cache" AND abs:(quantization OR compression) AND abs:latent
3. abs:DeepSeek AND abs:"KV cache" AND abs:(quantization OR quantized)
4. abs:"weight absorption" AND abs:attention  → 0 hits
5. abs:latent AND abs:"KV cache" AND abs:(quantization OR quantized) AND abs:(transform OR rotation OR rotated OR basis)
6. abs:"transform coding" AND abs:("KV cache" OR "key-value cache" OR attention)
7. abs:("weighted MSE" OR "weighted mean squared" OR "weighted distortion") AND abs:(quantization OR quantized) AND abs:(attention OR "KV cache" OR LLM)
8. abs:"low-rank" AND abs:"KV cache" AND abs:(quantization OR quantized) AND abs:latent
9. abs:("key cache" OR "KV cache" OR "key-value cache") AND abs:(quantization OR quantized) AND abs:(query OR queries) AND abs:(covariance OR "second moment" OR statistics) AND abs:(rotation OR transform OR basis OR PCA OR SVD)
10. [--sort date --from 20250601] abs:("latent attention" OR MLA) AND abs:(quantization OR quantized OR "low-bit" OR FP8 OR FP4 OR INT4)
11. [--sort date --from 20250601] abs:("KV cache" OR "key-value cache" OR "key cache") AND abs:(quantization OR quantized) AND abs:(transform OR rotation OR Hadamard OR PCA OR KLT OR "Karhunen" OR whitening OR whitened)
12. abs:("latent attention" OR "MLA") AND abs:(Hadamard OR rotation OR orthogonal OR "equivalent transformation" OR reparameteriz*)
13. abs:(absorbed OR absorption OR "absorb") AND abs:("KV cache" OR latent) AND abs:(quantization OR quantized OR "low-bit")
14. abs:("low-rank" OR latent) AND abs:("KV cache" OR "key-value cache") AND abs:("bit allocation" OR "water-filling" OR "waterfilling" OR "mixed-precision")
15. ti:CARE AND abs:"latent attention" AND abs:covariance
16. abs:("low-rank" OR "low rank") AND abs:("KV cache" OR "KV-cache") AND abs:(quantization OR quantized) AND abs:(Hadamard OR Walsh)
17. abs:(MLA OR "latent attention") AND abs:(RoPE OR "rotary" OR "decoupled") AND abs:(quantization OR quantized OR precision OR "bit")  [from 20250101]
18. [--sort date --from 20250601] abs:("latent attention" OR "latent KV" OR "latent cache" OR "compressed KV") AND abs:(INT4 OR INT2 OR "4-bit" OR "2-bit" OR "3-bit" OR FP8 OR FP4)
Adjacent-field (cs.IT / eess.SP / stat.ML / math.OC vocabulary):
A1. abs:("weighted mean squared error" OR "weighted squared error" OR "weighted MSE" OR "weighted distortion") AND abs:("transform coding" OR "Karhunen" OR KLT OR "bit allocation" OR "water-filling")
A2. abs:("water-filling" OR "waterfilling" OR "reverse water") AND abs:(transformer OR attention OR "KV cache" OR LLM OR "language model")
A3. abs:(Mahalanobis OR whitening OR whitened OR "metric-aware" OR "operator norm") AND abs:("KV cache" OR "key cache" OR "key-value cache" OR "value cache") AND abs:(quantization OR quantized OR compression)
A4. abs:("rate-distortion" OR "rate distortion") AND abs:("KV cache" OR "key-value cache" OR "key cache" OR "attention cache")
A5. abs:("noise shaping" OR "sigma-delta" OR "sigma delta" OR dither OR dithered OR "noise feedback") AND abs:("KV cache" OR "key-value" OR attention OR transformer) AND abs:(quantization OR quantized)
A6. abs:(Hessian OR Fisher OR "second-order" OR Jacobian OR "sensitivity-aware" OR "loss-aware") AND abs:("KV cache" OR "key-value cache" OR "key cache") AND abs:(quantization OR quantized)
A7. abs:("Marchenko" OR "random matrix" OR "effective rank" OR "spectral decay" OR "eigenvalue decay") AND abs:("KV cache" OR "key cache" OR "attention keys" OR "key-value")
A8. abs:("mixed-precision" OR "mixed precision" OR "bit allocation" OR "variable bit") AND abs:("principal component" OR PCA OR SVD OR eigen OR spectral OR "singular value") AND abs:("KV cache" OR "key cache" OR "key-value cache")
A9. abs:("vector quantization" OR "product quantization" OR "lattice") AND abs:("latent attention" OR "MLA" OR "latent KV" OR "compressed KV")
A10. abs:(heteroscedastic OR "importance-weighted" OR "importance weighted" OR "query-aware" OR "query-weighted") AND abs:(quantization OR quantized) AND abs:("KV cache" OR "key cache" OR "key-value cache")
Web: WebSearch "FlashMLA FP8 FP4 latent KV cache accuracy DeepSeek V4 e2m1 288 bytes per token"; WebSearch "\"QuantMLA\" function-aligned dual-path quantization MLA KV cache openreview" (no OpenReview page found); WebSearch (extended) "MLA latent KV cache quantization optimal transform folded into W_UK W_UV weight absorption Karhunen-Loeve query covariance weighted"; WebFetch https://github.com/deepseek-ai/FlashMLA.

Full texts: /tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers/<id>/<id>.txt ; raw TSV hits in …/scratchpad/sweep10/.
