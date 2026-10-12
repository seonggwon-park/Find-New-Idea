# Idea 9 — contrarian sweep (Angle D): RoPE-pair mixing lemma, pre/post-RoPE × quantizer type, frequency-band allocation, YaRN interaction

Date: 2026-10-10. Scope: find papers whose *measured* results falsify a factual premise of Idea 9, plus papers that already state or prove parts of it. All quotes below are from downloaded paper text (untrusted data; quoted, not followed).

## 0. Premises of Idea 9 that were tested

| # | Premise in the report | Verdict | Key evidence |
|---|---|---|---|
| P1 | KVQuant: pre-RoPE per-channel gives +0.82 PPL at 3-bit | **Verified** | KVQuant Tab. 10: int3 per-channel LLaMA-7B post-RoPE 7.05 → pre-RoPE 6.23 (fp16 5.68) |
| P2 | "Hadamard/VQ quantizers mix channels, so the pre-RoPE gain disappears" (Lemma 9.1 corollary in the hypothesis) | **Contradicted for calibrated Hadamard and learned-codebook VQ; holds only for randomized/Gaussianizing quantizers** | RotateKV Tab. 2 (pre 0.247 vs post 0.605 MSE); TaSQ Fig. 1d (pre-RoPE VQ 35% lower error); AnTKV App. A; NSNQuant Tab. 11 (5.29 vs 5.33). Holds for HIGGS: YAKV Tab. 10, AQUA-KV App. D, HyperQuant |
| P3 | "No paper explains the quantizer dependence" | **False** | AQUA-KV (Jan 2025) states it explicitly with a 2×2 ablation (HIGGS vs Quanto × pre/post); KVQuant App. Tab. 20 already shows the sign flips for per-token (post better) and explains why; HyperQuant states orthogonality+RHT equivalence |
| P4 | YAKV [2604.08426] reports no gain with HIGGS (was [U]) | **Verified** — YAKV is the method inside "KV Cache Offloading for Context-Intensive Tasks" | Tab. 10: pre 93.36 vs post 93.73 (Llama-3.1-8B MultiNeedle); 90.83 vs 91.05 (Qwen3-4B Text2JSON) |
| P5 | Outlier (max-abs) key pairs concentrate in high-frequency (low-index) pairs (conjecture) | **Contradicted by 3 papers** | Massive Spikes (2606.02288): sink channels at i≈55/64, coherent pairs; Massive Values (ICML'25) main text: low-frequency; Round-and-Round: Gemma pushes mid/high-freq norms to ~0 |
| P6 | KV-Latent: "low-index RoPE channels are most of the noise" | **Misread** — KV-Latent's "noise" is the periodic oscillation of the RoPE score-vs-distance function under dimension reduction, not quantization noise or outlier energy | KV-Latent §3.3.2 |
| P7 | Block-GTQ allocates by *key* energy; Cor. 9.3 adds query weighting | **False** — Block-GTQ's block weight is s*_i = E[‖q^(i)‖‖k^(i)‖] (query-weighted), Lemma 2 + Theorem 1, 4^-b rate law, analysed in pre-RoPE coordinates | Block-GTQ §2.3, App. A |
| P8 | YaRN × W4 interacts destructively (NTK/linear PPL>100) | **Overstated for YaRN** | Rethinking RoPE (2510.00028) Tab. 2: YaRN s=8 @16K FP16 2.438 / RTN-W4 2.517 / AWQ 2.476 ("all quantized baselines stay close to FP16"); only s=32 @131K blows up (FP16 5.386 / RTN 9.958 / AWQ 8.700 / Q-ROAR 7.869); PPL>100 is NTK/linear |
| P9 | No theory tracks mixing + band allocation + YaRN in one representation | **Partially false** | Rethinking RoPE eq. (12)–(14): per-pair axis-aligned quantizer MSE ≈ Δ²/6·η_i(m) ("anisotropy shift", "dynamic-range dilation") combined with logit bound; Block-GTQ 2-D block decomposition; FireQ Thm 3.1 pair-norm bound |

Net: the *lemma* (per-channel max-abs range of a rotated outlier pair as a function of G·θ_f) is still unstated anywhere in closed form, but every surrounding claim in the hypothesis needs rewriting. The real dichotomy the evidence supports is **distribution-adaptive (calibrated scales/codebooks/reorderings) vs. distribution-agnostic (randomized Hadamard + fixed grid)**, not "per-channel vs. Hadamard/VQ"; and for per-token quantizers post-RoPE is *better* (KVQuant Tab. 20, AQUA-KV Quanto), which the current Lemma 9.2 does not predict.

## 1. Candidates (ranked)

### 1. RotateKV — 2501.16383 (Su et al., Jan 2025) — **contradicts**, priority 1
Outlier-aware FWHT rotation + calibrated channel reordering on keys. Table 2 (LLaMA-2-7B, 2-bit key reconstruction MSE): "Rotation" pre-RoPE 0.632 vs post-RoPE 0.670; "Rotation + Reordering" pre-RoPE **0.247** vs post-RoPE **0.605**; text: "the quantization error increases by 145% after applying RoPE." Also "Reordering + Rotation" 0.477 vs 0.568. So a Hadamard-family quantizer with calibrated structure gains ~2.4× from pre-RoPE, falsifying "Hadamard ⇒ no pre-RoPE gain." Plain Hadamard alone shows only a 6% gap, which is the weaker claim the idea should make. Figure 6c shows channel 51 before/after RoPE. Block-GTQ's related-work also notes RotateKV operates before RoPE.

### 2. AQUA-KV / "Cache Me If You Must" — 2501.19392 (Jan 2025) — **partially_preempts**, priority 1
§Positional embeddings: "the uniform quantizer (Quanto) works slightly better post-RoPE, while HIGGS works equally well in both cases due to its use of the Hadamard transform." Appendix D "Predictor Order" rows: Before-RoPE HIGGS 2.16 bits, PPL 7.03, LongBench 44.26; After-RoPE HIGGS 7.05 / 44.13; Before-RoPE Quanto-gs64 2.64 bits, PPL 10.90 / 44.54; After-RoPE Quanto-gs64 **10.33** / 43.64. Also: "the optimal predictor for post-RoPE compression needs to be rotation-equivariant" — the equivariance framing of Lemma 9.1 is already in print. This is the explicit quantizer-dependence statement the idea claims nobody made.

### 3. KVQuant — 2401.18079 (NeurIPS 2024) — **partially_preempts**, priority 1
Tab. 10 verifies P1. But App. Tab. 20 (per-token keys, LLaMA-7B, Wikitext-2): int4 pre 6.02 / post 5.98; int4-gs128 5.76 / 5.77; int3 pre **14.68** / post **10.87**; int3-gs128 6.28 / 6.17, with the mechanism: "part of the magnitude from the outlier channel will be shifted to the smaller channel. This partially mitigates the impact of the outlier channel on skewing the quantization range for some of the tokens." KVQuant therefore already reports that the sign of the pre/post gain is a function of the quantizer axis. Idea 9's Lemma 9.2 only covers the per-channel axis; a per-token corollary (mixing *reduces* per-token range) is missing and must be added for the "reconciliation" claim to hold.

### 4. Block-GTQ / RoPE-Aware Bit Allocation — 2606.24033 (Jun 2026) — **partially_preempts** (Cor. 9.3), priority 1
§2.1–2.3: logit decomposed over 2-D RoPE blocks; "norm-based block statistics are RoPE-invariant — we can therefore compute the energy score in pre-RoPE coordinates"; Lemma 2: E_{n,m} ≤ Σ_i ‖q_n^(i)‖‖e_{k,m}^(i)‖; ideal block weight s*_i := E[‖q^(i)‖‖k^(i)‖] (query-weighted), surrogate s_i = ½E[‖q^(i)‖²+‖k^(i)‖²]; TQ-MSE rate law E‖x−x̂‖² ≤ ‖x‖²·(3π/2)·4^-b; Theorem 1 greedy allocation is optimal. This is Cor. 9.3 up to the choice of bound (Cauchy–Schwarz absolute vs. Gaussian variance/reverse water-filling). Results: LongBench-EN avg 36.87 → 53.31 vs uniform TQ-MSE; NIAH 4K–128K on Llama-3.1-8B-Instruct. The report's description ("key-energy allocation") is wrong.

### 5. Rethinking RoPE Scaling in Quantized LLM — 2510.00028 (Qiao et al., Sep 2025) + Q-ROAR 2509.14391 (AAAI 2026) — **partially_preempts**, priority 1
Eq. (13)–(14): for a rotated 2-D pair with per-channel mid-rise quantizer, E‖ẑ^(i)(m)−z^(i)(m)‖² ≈ Δ_i²/6·η_i(m), η_i ↑ "when φ_i^scaled(m) induces axis mismatch" ("Anisotropy shift: per-channel quantization grids are axis-aligned; RoPE rotates the signal relative to these grids"); combined with eq. (12) ⇒ "position-dependent logit error that is typically largest for high-frequency i and large |m−n|". Q-ROAR lists "(iii) anisotropy when axis-aligned quantizers operate on RoPE-rotated pairs" as a failure mode. This is the mechanism of Lemma 9.2 and the YaRN coupling, stated heuristically (no closed form in G·θ_f, no quantizer-class split). Numbers (Tab. 2, Proof-pile, LLaMA-2-7B): s=8: 2K FP16 2.847 / RTN-W4 2.917 / AWQ 2.891 / Q-ROAR 2.888; 16K 2.438 / 2.517 / 2.476 / 2.458; s=32: 32K 4.261 / 6.249 / 5.460 / 5.054; 131K 5.386 / 9.958 / 8.700 / 7.869. "With moderate YaRN scaling (s=8), all quantized baselines stay close to FP16." The idea's P8 must be restated as "YaRN at s≥32 without finetuning", with NTK/linear being the PPL>100 cases. Their appendix also reports Hadamard rotation gives 3–8% improvement in W+A quantization.

### 6. Massive Spikes in LLMs are Bias Vectors — 2606.02288 (Jun 2026) — **contradicts** P5, priority 1
Fig. 3c: "channels driving the attention sink mechanism (top-2 score) strictly localize in low-frequency bands (green) and form coherent RoPE pairs (red)"; "dominant sink features are disproportionately concentrated in high-index subspaces (i ≈ 55 out of 64)". App. G.3: Δm·θ_{i*} ≈ 0 ⇒ θ_{i*} ≈ 0 forces the bias subspace into high-index (lowest-frequency) channels; G.4 "coherent pairing" (both channels of the pair active so cross terms cancel). Consequence for Lemma 9.2: the largest-magnitude key channels sit where G·θ_f ≪ 1 *and* max₁ ≈ max₂, i.e. exactly where the lemma predicts zero pre/post difference. The KVQuant gain must then come from mid-frequency, non-sink outlier pairs — a testable but currently unsupported claim.

### 7. TaSQ — 2610.03027 (Oct 2026) [+ AnTKV 2506.19505, NSNQuant 2505.18231] — **contradicts** P2 ("VQ ⇒ no gain"), priority 1
TaSQ §3.3: "the relatively compact pre-RoPE key distribution becomes widely dispersed after RoPE … pre-RoPE VQ achieves lower reconstruction error across all 32 layers, with a 35% lower total reconstruction error" (Fig. 1d); Fig. 7: post-RoPE-fitted codebooks (NovaKV) "exhibit a sharp increase in error beyond the calibration range", pre-RoPE VQ (CQ, TaSQ) degrade only gradually. AnTKV App. A: "pre-RoPE K exhibits smaller inter-cluster distances and lower intra-cluster variance." NSNQuant Tab. 11 (Hadamard + VQ, calibration-free): pre-RoPE NSN 5.29 vs post 5.33 (LLaMA2-7B), 4.71 vs 4.73 (13B) — small but nonzero gain even with Hadamard. Learned/adaptive codebooks are not pair-equivariant and benefit from pre-RoPE; the idea's hypothesis sentence lumps "Hadamard/VQ" together and contradicts its own support list.

### 8. KV Cache Offloading for Context-Intensive Tasks (YAKV) — 2604.08426 (Apr 2026, v2 Sep 2026) — **supports**, priority 2
Verifies the [U] citation. App. "Comparison of pre-RoPE and post-RoPE", Tab. 10: pre-RoPE 93.36 vs post 93.73 (Llama-3.1-8B-Instruct, MultiNeedle); 90.83 vs 91.05 (Qwen3-4B, Text2JSON). "pre-RoPE key quantization does not significantly improve model accuracy for YAKV. We attribute this to the randomized Hadamard transformation used in the HIGGS." HyperQuant 2606.23406 §6 makes the same point more sharply: "since RoPE is a per-position orthogonal rotation that commutes with ℓ2 normalization, pre- and post-RoPE quantization are statistically equivalent" (for its RHT + lattice pipeline). Note the explanation (orthogonality + Gaussianization) is already written down — Lemma 9.1 would formalize it, not discover it.

### 9. Round and Round We Go — 2410.06205 (Barbero et al., ICLR 2025) [+ DoPE 2511.09146] — **contradicts** practical weight of P5, priority 2
§4: Gemma 7B "assigned much higher norm on average to the lowest frequencies"; "learning discovers that these medium to high frequency 'chunks' are not useful and hence pushes their norm to 0"; high-frequency use mainly in first/last layers and specific positional heads (Thm 5.3). If E_f ≈ 0 for high-frequency pairs, Σ_f[mix(Gθ_f)−1]E_f is small, contradicting "gain concentrated in high-frequency pairs" being material. DoPE (Lemma 3.1, Thm A.4) further argues low-frequency bands concentrate structured energy / massive activations. EliteKV 2503.01586 finds LLaMA2-7B heads are "predominantly mid-frequency". The band-allocation prediction needs per-model measurement, not assumption.

### 10. Massive Values in Self-Attention Modules — 2502.01563 (Jin et al., ICML 2025) — **contradicts** (with caveat), priority 2
Main text: "massive values to concentrate at the bottom (i.e., low-frequency regions)"; Fig. 7 (Gemma2-9B): "Massive values are concentrated in low-frequency regions of K and Q, absent in V." But App. text for Llama3-8B, LLaVA and Qwen2.5-7B says "concentrated in the high-frequency regions of the Q and K matrices." The paper's own frequency labelling is inconsistent, so it cannot be used as clean support for either sign; the report should not cite it without re-measuring.

### 11. WUSH-KV — 2609.38121 (Sep 2026) [+ FPTQuant 2506.04985, ICML 2026] — **partially_preempts** Lemma 9.1's class, priority 2
App. A.1 eq. (9): "For distinct nondegenerate rotary frequencies, the real invertible transforms that commute with every R_s are T = blockdiag_j(ρ_{h,j} R(θ_{h,j}))"; with learned RMSNorm gains only paired sign flips survive, "which are trivial because they do not redistribute coordinate magnitudes." App. A.2: unrestricted pre-RoPE transforms need position-dependent compensation. FPTQuant Thm 3.1: mergeable pre-RoPE transforms T_k = diag(s)·diag({R_n}), R_n ∈ O(2). The pair-equivariant class is thus characterised twice; Lemma 9.1 reduces to "MSE of such a quantizer is invariant", which is immediate.

### 12. FireQ — 2505.20839 (May 2025) — **partially_preempts** Lemma 9.2 range geometry, priority 2
"As RoPE intermixes paired channel distributions while leaving non-paired channels unaffected"; RPN (Thm 3.1): shared pair scale s_i = s_j = α·max_n ‖(k_i^n, k_j^n)‖₂ bounds post-RoPE coordinates by 1/α — i.e. the post-RoPE per-pair range is the max token pair-norm, which is the "both channels → max₁" statement of Lemma 9.2 without the G·θ_f dependence. FireQ also shows post-RoPE per-token INT4 + two-stage smoothing (8 outlier pairs via CRS) works on Llama-3.1-8B.

### 13. KV-COBRA — 2609.24298 (Sep 2026) [+ Tensor Decomposition of KV caches 2609.28029] — **supports** frequency-ordered mixing, priority 2
App. G: "Pre-RoPE remains uniformly better (Figure A6) because the per-head covariance is position-stationary pre-RoPE and position-dependent post-RoPE"; "a static basis faces the average over positions E_p R(p)ΣR(p)ᵀ, which is much closer to isotropic than Σ itself, with the high-frequency rotation planes flattening first." 2609.28029: "post-RoPE keys lose 41%–64% of their pre-RoPE compressibility." Both are low-rank/covariance statements, so they support the ordering in Lemma 9.2 but for a different quantity (second moments, not max-abs range).

### 14. KV-Latent — 2507.11273 (Jul 2025) — **tangential** (premise misread), priority 2
§3.3.2: "the low-numbered channels of the RoPE function contribute the majority of the noise, while the high-numbered channels, despite a slower decay, remain relatively stable" — this is about the periodic oscillation of the RoPE score vs. relative distance when head dim is reduced (Figs. 3–6), fixed by "densifying the sampling of low-frequency rotations". Nothing about quantization noise or outlier energy. Remove it as support for P5.

### 15. PolarQuant (Wu et al.) — 2502.00527 (NeurIPS 2025) — **partially_preempts** Lemma 9.1 example, priority 2
Quantizes *post-RoPE* keys per RoPE pair as (ρ, θ), claiming to "preserve the benefits of pre-RoPE quantization in reducing approximation errors while eliminating redundant computations"; "Since rotation is a magnitude-preserving transformation, these 2D sub-vectors inherit the magnitude characteristics seen in the pre-RoPE case." But its θ quantizer is group-wise asymmetric min–max (s_θ = (max θ − min θ)/2^t), which is *not* rotation-invariant: pre-RoPE angles are concentrated, post-RoPE angles are spread over (0, 2π). Experiment (c) "polar ⇒ invariant" holds only for a uniform full-circle angle grid; with PolarQuant-Wu's quantizer pre-RoPE could still win. Han et al.'s PolarQuant 2502.02617 uses random rotation + known angle distribution and is the Gaussianizing case.

## 2. Additional notes (not in the top 15)
- **PM-KVQ 2505.18610 (ICLR 2026)**: "For low-frequency channels after RoPE, which have a period of over 32K tokens, calibration using short sequences (e.g., 2K tokens) fails"; lowest-frequency period 54,410 tokens in DeepSeek-R1-Distill-Qwen-7B. For per-channel scales calibrated on a window shorter than deployment length, low-frequency pairs *do* drift post-RoPE — the relevant G in Lemma 9.2 is the calibration/scale-sharing span, not the quantization group. Adds a regime the lemma should state.
- **Dual-QK 2610.09827 (Oct 2026)**: "bucket-relative RoPE" stores keys relative to a bucket anchor (w = 2048 = calibration window) so that stored-key offsets stay in [0, w) — an engineering knob on G·θ_f; ablation: removing it lowers HumanEval.
- **RotateAttention 2607.02584 (video DiT, 3D RoPE)**: "low-frequency channel pairs accumulate larger angular spread over long sequences and thus higher variance, while high-frequency pairs remain comparatively smooth" — opposite ordering to Lemma 9.2 in a 3D-RoPE setting; uses pair-wise 2×2 (H₂) block-diagonal rotations.
- **Interface-Aware KV 2609.05764**: Hadamard after RoPE "because the two transformations do not commute" — verified.
- **Transforms for LLM Quantization: The Great Inversion 2608.25188 (survey)**: synthesises KVQuant/RotateKV being "forced into the pre-rotary basis" and PolarQuant inverting the obstacle — good related-work source, no new measurement.
- **QServe 2405.04532**: SmoothAttention must share the scale within RoPE pairs (i, i+d/2) to commute with RoPE.
- **NovaKV** (cited by TaSQ as Fernández-Menduiña et al., 2026) is not findable on arXiv by title; OSCAR 2605.17757 text contains no RoPE-placement statement.

## 3. Queries run
arXiv API (arxiv_search.py, --sort date):
- abs:RoPE AND abs:quantization AND abs:key
- abs:"rotary" AND abs:"KV cache" AND abs:quantization
- abs:"pre-RoPE" OR abs:"pre-rope" OR abs:"post-RoPE" OR abs:"before RoPE" OR abs:"after RoPE"
- abs:frequency AND abs:RoPE AND abs:quantization
- abs:YaRN AND (abs:quantization OR abs:quantized OR abs:low-bit)
- abs:"position interpolation" AND (abs:quantization OR abs:quantized)
- abs:polar AND abs:quantization AND (abs:RoPE OR abs:"KV cache")
- ti:YAKV OR abs:YAKV OR abs:"yet another KV"
- abs:"rotary position" AND abs:outlier AND abs:channel
- abs:"rotary" AND abs:"massive values"
- abs:HIGGS AND abs:"KV cache"
- abs:Hadamard AND abs:"KV cache" AND (abs:RoPE OR abs:rotary)
- abs:RoPE AND (abs:"low-frequency" OR abs:"low frequency") AND (abs:outlier OR abs:"attention sink" OR abs:massive)
- abs:RoPE AND abs:frequency AND abs:"KV cache"
- abs:RoPE AND abs:"vector quantization"
- abs:YaRN AND abs:"KV cache"
- abs:"context extension" AND (abs:quantization OR abs:quantized OR abs:PTQ)
- abs:NoPE AND (abs:quantization OR abs:quantized)
- abs:RoPE AND abs:dimension AND (abs:inefficien OR abs:wasted OR abs:unused)
- abs:"KV cache" AND abs:quantization AND abs:"per-channel" AND (abs:RoPE OR abs:rotary OR abs:"positional")
- abs:"KV cache" AND abs:quantization AND abs:frequency AND (abs:band OR abs:allocation)
- abs:RoPE AND abs:"attention sink" AND abs:frequency
- abs:"long-context" AND abs:RoPE AND abs:quantization AND abs:scaling
- abs:quantization AND abs:"RoPE" AND abs:"key" AND abs:rotation
- ti:NovaKV OR abs:NovaKV
- abs:"KV cache" AND abs:quantization AND abs:outlier AND (abs:"low-frequency" OR abs:"high-frequency")
- abs:"KV cache" AND abs:codebook AND (abs:RoPE OR abs:rotary)
- abs:RoPE AND abs:quantization AND (abs:"group size" OR abs:"group-wise" OR abs:groupwise)
- ti:RoPE AND ti:quantization
- abs:RoPE AND abs:"attention sink" AND abs:quantization
- abs:"KV cache" AND abs:quantization AND abs:"long-context" AND abs:extrapolation
- abs:"logit error" AND (abs:RoPE OR abs:rotary) AND abs:quantization
- abs:quantization AND abs:RoPE AND abs:"relative position" AND abs:error
- ti:OSCAR AND abs:"KV cache"
- abs:"KV cache" AND abs:quantization AND abs:"channel pruning" AND abs:query
- abs:"KV cache" AND abs:quantization AND (abs:"positional encoding" OR abs:"position encoding") AND abs:transform
- abs:"KV cache" AND abs:quantization AND abs:"per-channel" AND abs:outlier AND abs:key
- abs:RoPE AND abs:"massive activation"
- abs:"KV cache" AND abs:"vector quantization" AND abs:"position-dependent"

WebSearch:
- YAKV KV cache quantization HIGGS pre-RoPE Hadamard arXiv
- "pre-RoPE" key quantization "no improvement" OR "no benefit" Hadamard KV cache 2026
- pre-RoPE vs post-RoPE key cache quantization ablation 2026 "RoPE" frequency channels quantization error high-frequency low-frequency
- NovaKV KV cache vector quantization post-RoPE Fernández-Menduiña 2026
- RoPE key outlier channels "low frequency" OR "high frequency" KV cache quantization per-channel 2026 arxiv which RoPE frequencies carry outliers

Full texts fetched (arxiv_fetch.py): 2401.18079, 2506.19505, 2610.03027, 2604.08426, 2609.05764, 2507.11273, 2606.24033, 2509.14391, 2510.00028, 2609.38121, 2610.09827, 2505.20839, 2608.27128, 2607.02584, 2501.16383, 2502.00527, 2502.02617, 2506.18879, 2502.01563, 2410.06205, 2606.02288, 2604.04921, 2602.02599, 2506.04985, 2609.28029, 2503.01586, 2511.09146, 2606.23406, 2604.04722, 2602.03152, 2502.12665, 2412.17739, 2501.18795, 2501.19392, 2505.18231, 2608.25188, 2603.16435, 2402.02750, 2609.24298, 2405.04532, 2605.17757, 2605.19660, 2505.18610.

## 4. What Idea 9 should become
1. Replace "per-channel vs Hadamard/VQ" with "distribution-adaptive vs distribution-agnostic (Gaussianizing)" quantizers; cite AQUA-KV, YAKV, HyperQuant (no gain) against RotateKV, TaSQ, AnTKV, NSNQuant (gain).
2. Add the per-token corollary (mixing *shrinks* per-token range; KVQuant Tab. 20, AQUA-KV Quanto) — without it the "reconciliation" is incomplete.
3. Drop KV-Latent as support; replace with a measurement plan, since Massive Spikes / Massive Values / Round-and-Round put the dominant-magnitude pairs in the *low*-frequency band.
4. Re-position Cor. 9.3 as a variance-based refinement of Block-GTQ's already query-weighted Cauchy–Schwarz allocation, and the YaRN part as a closed-form version of Rethinking-RoPE eq. (14)'s η_i(m).
5. Restate the YaRN premise: mild at s=8 (≈3% PPL), severe only at s=32 without finetuning; NTK/linear are the PPL>100 cases.
