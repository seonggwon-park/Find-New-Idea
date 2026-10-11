# Idea 9 — RoPE-pair mixing lemma: keyword / adjacent-field prior-art sweep

Date: 2026-10-10. Scope: Angle A (direct keyword search 2023-01..2026-10 + adjacent-field vocabulary). Source texts under
`/tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers/<id>/<id>.txt`.
All quotations below are untrusted paper text, quoted for evidence only.

## The claim being tested

Idea 9 asserts three things: (L9.1) a quantizer whose MSE is invariant under per-pair 2x2 rotations (polar, RoPE-commuting VQ,
Hadamard-style) gains nothing from pre-RoPE key quantization; (L9.2) per-channel max-abs key quantizers over a token group G lose
up to ~2x MSE on asymmetric outlier pairs (max1 >> max2) when G*theta_f >= pi, nothing when G*theta_f << 1, so the pre-RoPE gain
is concentrated in high-frequency (low-index) pairs; (C9.3) reverse water-filling over pairs with query-energy weights, phase
irrelevant; plus a YaRN interaction story (band-wise re-mixing + accumulated W_Q/W_K error) that subsumes Q-ROAR's band scales.

## Verdict in one paragraph

No paper states or proves the G*theta_f mixing factor or the frequency-band concentration of the pre-/post-RoPE gain. However the
*qualitative* resolution that the idea sells as its headline — "the pre-RoPE gain depends on the quantizer; Hadamard removes it" —
was already written down, with numbers, in Jan 2025 by AQUA-KV (2501.19392: "uniform quantizer (Quanto) works slightly better
post-RoPE, while HIGGS works equally well in both cases due to its use of the Hadamard transform"; Appendix D table: HIGGS 7.03 vs
7.05 PPL, Quanto-gs64 10.90 vs 10.33) and in Jan 2024 by KVQuant itself (Appendix H Table 10 per-channel int3 post 7.05 / pre 6.23;
Appendix P Table 20 per-token int3 pre 14.68 / post 10.87, with the explicit sentence that rotating an outlier channel with a
small channel shifts magnitude to the smaller channel). RotateKV Table 2 further shows plain rotation is nearly invariant
(pre 0.632 vs post 0.670 key-MSE) while rotation+calibrated-reordering is not (0.247 vs 0.605, +145%) — so "Hadamard => no gain" is
true only for the equivariant/randomized case and false once a non-equivariant permutation/group structure sits on top. The
characterization of the RoPE-commuting orthogonal family (per-pair planar rotations, and nothing else for distinct frequencies)
is proved as Lemma 2 of 2608.13365, which also reports a *negative* result for pair-aligned rotations vs full-head Hadamard under a
token-wise range quantizer, and discusses exactly the "position-averaged second moment vs tokenwise max-min range" mismatch that
L9.2 is built on. C9.3's structure (q_c^2-weighted per-channel white-noise distortion + reverse water-filling) is Theorem 1 of
AATC (2608.14191), and Block-GTQ's score is already E[||q^(i)|| ||k^(i)||] per RoPE block (not key-energy-only as the report says).
The report's conjecture that outlier pairs sit in high-frequency bands is contradicted for the *largest* spikes: Massive-Spikes
(2606.02288) finds sink-driving channels at i ~ 55/64 (lowest frequencies) and in coherent pairs (both channels active, ~80%),
precisely the regime where L9.2 predicts no mixing loss. The idea survives as "formula + band prediction + unification", not as
the discovery of quantizer dependence.

## Queries run (arXiv API via scripts/arxiv_search.py; dates 20230101–20261031 unless noted)

Seed/keyword formulations:
- Q1 `abs:RoPE AND abs:quantization AND abs:key`
- Q2 `abs:rotary AND abs:"KV cache" AND abs:quantization`
- Q3 `abs:"pre-RoPE" OR abs:"pre-rope" OR abs:"post-RoPE" OR abs:"before RoPE" OR abs:"after RoPE"`
- Q4 `abs:frequency AND abs:RoPE AND abs:quantization`
- Q5 `abs:YaRN AND (abs:quantization OR abs:quantized OR abs:low-bit)` -> 0 hits
- Q6 `abs:"position interpolation" AND (abs:quantization OR abs:quantized)` -> only Q-ROAR / Rethinking-RoPE / PM-KVQ
- Q7 `abs:polar AND abs:quantization AND (abs:RoPE OR abs:rotary)`
- Q8 `abs:"rotary position" AND abs:outlier AND abs:channel`
- Q9 `(abs:rotary OR abs:RoPE) AND abs:outlier AND (abs:quantization OR abs:quantized)`
- Q14 `(abs:rotary OR abs:RoPE) AND abs:"per-channel" AND (abs:quantization OR abs:quantized)`
- Q15 `ti:"Resonating with RoPE" OR (abs:"spectral quantization" AND (abs:"key cache" OR abs:"KV cache")) OR abs:SpectrumQuant` -> not on arXiv
- Q16 `(abs:DCT OR abs:"discrete cosine") AND (abs:"key cache" OR abs:"KV cache") AND (abs:quantization OR abs:compression) AND (abs:RoPE OR abs:rotary)` -> 0
Date-sorted, --from 20250601:
- Q10 `(abs:RoPE OR abs:rotary) AND (abs:quantization OR abs:quantized OR abs:low-bit) AND (abs:key OR abs:"KV cache")`
- Q11 `abs:"KV cache" AND abs:quantization AND abs:key AND (abs:frequency OR abs:band OR abs:pair)`
- Q12 `abs:Hadamard AND abs:"KV cache" AND (abs:quantization OR abs:quantized)`
- Q13 `(abs:RoPE OR abs:rotary) AND (abs:quantization OR abs:quantized)`
Adjacent-field vocabulary:
- A1 `("rotation invariant"|"rotation-invariant"|"rotation equivariant"|"rotation-equivariant"|"rotationally invariant") AND (quantizer|quantization)` in cs.IT/eess.SP/cs.LG/stat.ML (2015–) -> only 2303.03106 (weight RIQ), rest MIMO noise
- A2 `("polar quantization"|"magnitude-phase quantization"|"phase quantization"|"polar quantizer")` cs.IT/eess.SP/cs.LG (2010–) -> RIS/PSK literature only (e.g. 2205.05850 capacity under polar quantization) — no transfer
- A3 `("reverse water-filling"|"water-filling") AND "bit allocation" AND (quantization|transform)` -> RDKV 2605.08317, else OFDM
- A4 `abs:RoPE AND ("high-frequency"|"high frequency"|"low-frequency"|"low frequency") AND abs:attention` (non-quant RoPE band analyses)
- A5 `(RoPE|rotary) AND ("channel pair"|"channel pairs"|"dimension pair"|"dimension pairs"|"paired dimensions"|"paired channels") AND (outlier|massive|quantization)` -> 2606.02288 only
- A6 `"per-channel" AND quantization AND key AND (rotation|rotated|RoPE) AND outlier`
- A7 `(RoPE|rotary) AND (quantization|quantized) AND ("relative position"|"relative distance"|"position-dependent")`
- A8 `"transform coding" AND ("KV cache"|"key-value cache"|attention|LLM)` -> 2608.14191, 2608.04074, 2607.20538
- A9 `(RoPE|rotary) AND complex AND (quantization|compression|"KV cache")` -> nothing on complex-coefficient quantization analysis
- A10 `(angle|angular|phase) AND quantization AND ("KV cache"|"key cache")` -> PolarQuant x2, TurboAngle, Spherical-KV, GSRQ, Hurwitz-quaternion
- A11 `equivariant AND quantization AND (rotation|rotations) AND cat:cs.LG` -> SO(3)-GNN quantization only
- A12 `("attention logit"|"attention logits"|"attention score") AND quantization AND error AND (RoPE|position|positional)`
Web: "YAKV KV cache quantization HIGGS Hadamard pre-RoPE arXiv"; "\"pre-RoPE\" \"post-RoPE\" key quantization Hadamard rotation KV
cache 2026"; "RoPE frequency bands KV cache key quantization high-frequency pairs per-channel scale mixing 2026 paper" (surfaced
2608.13365 and ACL-2026 SpectrumQuant). ACL page fetched via curl (aclanthology.org/2026.acl-long.1732).

## Candidates (evidence with locations)

### P1 — must deep-read

**2401.18079 KVQuant (Hooper et al., NeurIPS 2024) — partially_preempts (qualitative mechanism + quantizer dependence).**
- Sec 3.2: "caching Key vectors after applying this rotation ... leads to mixing pairs of channels by different amounts for
  different positions ... makes it harder to quantize Key activation channels which would typically have consistent
  large-magnitude values." Appendix H Table 10 (LLaMA-7B, Wikitext-2, per-channel K / per-token V): int3 post-RoPE 7.05, pre-RoPE
  6.23 (fp16 5.68) -> the 0.82 figure.
- Appendix P Table 20 (per-token K quantization): int4 pre 6.02 / post 5.98; int4-gs128 5.76 / 5.77; int3 pre 14.68 / post 10.87;
  int3-gs128 6.28 / 6.17. Text: "when rotating an outlier channel with large average magnitude and another channel with smaller
  average magnitude together, at some positions ... part of the magnitude from the outlier channel will be shifted to the smaller
  channel. This partially mitigates the impact of the outlier channel on skewing the quantization range."
- So KVQuant already contains the sign-flip of the pre/post gain with quantizer granularity (per-channel vs per-token) and the
  range-mixing explanation in words. Missing: any formula, any dependence on G*theta_f, any frequency-band statement. L9.2 should be
  framed as formalizing KVQuant App. H + App. P, and the per-token case (post better because max over channels of a rotated
  asymmetric pair is <= max1) is a free corollary the idea currently omits.

**2501.19392 Cache Me If You Must / AQUA-KV (Shutova et al., Jan 2025) — partially_preempts (explicit quantizer-dependence claim).**
- Sec. "Positional embeddings": "linear predictors ... offer better accuracy in pre-RoPE compression. We attribute this to the fact
  that the optimal predictor for post-RoPE compression needs to be rotation-equivariant, and the simple linear models we use are
  not. As for the backbone quantization, we found that the uniform quantizer (Quanto) works slightly better post-RoPE, while HIGGS
  works equally well in both cases due to its use of the Hadamard transform."
- Appendix D table (Llama-3.x base, WikiText-2 PPL / LongBench): Before-RoPE HIGGS 7.03 / After-RoPE HIGGS 7.05; Before-RoPE Quanto
  gs64 10.90 / After-RoPE Quanto gs64 10.33 (Quanto is per-token here, Sec. "Per-token and per-channel compression").
- This is the earliest explicit statement of "Hadamard-based quantizer => pre/post RoPE irrelevant; uniform => depends", i.e. the
  idea's reconciliation of KVQuant vs YAKV is not new as a *claim*. It is one sentence with no derivation.

**2501.16383 RotateKV (Su et al., 2025) — contradicts the blanket "Hadamard => no gain", supports it for plain rotation.**
- Table 2 (LLaMA-2-7B, 2-bit key reconstruction MSE, pre vs post RoPE): Rotation 0.632 vs 0.670; Rotation+Reordering 0.247 vs 0.605;
  Reordering+Rotation 0.477 vs 0.568. Text: "the quantization error increases by 145% after applying RoPE" (for the
  rotation+reordering pipeline). Table 7 (LLaMA-2-13B, 2-bit, WikiText-2): Original (post-RoPE per-head) rotations 6.96 -> + Pre-RoPE
  grouped-head key rotation 5.67 (-1.29).
- Reading: a bare FWHT is ~invariant (+6%) as L9.1 says, but FWHT composed with a calibrated channel permutation and per-group
  ranges (and cross-head grouping, only possible pre-RoPE) is strongly non-invariant. L9.1 must be stated for quantizers whose
  *scale-setting statistic* is pair-rotation invariant; "Hadamard" alone is not a sufficient descriptor.

**2608.13365 "When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations for Dynamic 4-Bit Quantisation" (Wang et
al., Aug 2026) — partially_preempts L9.1's class characterization and anticipates the range-vs-moment argument.**
- Lemma 2 (single-head RoPE centraliser): "For pairwise-distinct RoPE angles theta_k in (0,pi), the centraliser of {R_m} in O(d_h) is
  B_{d_h}" = products of per-pair SO(2) rotations (Definition 1). Credits FPTQuant (2506.04985) for the family, proves the converse.
  Theorem 3: closed-form per-pair angle equalising the two diagonal entries of a position-averaged pooled covariance.
  Proposition 4: attention preservation requires shared Q/K band angles.
- Negative result: head-shared pairwise rotations *increase* PPL vs full-head Hadamard under dynamic W4A4KV4 on four checkpoints;
  composing with Hadamard is within +-0.05 PPL. Sec. "Objective mismatch": "Dynamic INT4 instead uses a tokenwise max-min range over
  groups that span multiple frequency pairs ... A quadratic moment averaged over RoPE phases can decrease while Delta_{t,g} remains
  unchanged because the two statistics apply different operations over different supports." Along a block-Hadamard support
  interpolation (b in {2,...,128}) "K range, relative quantisation error, and perplexity degradation decrease with support size".
- Overlap: the idea's L9.2 is exactly a statement about the max-abs scale statistic under pair rotation; this paper frames the same
  dichotomy (second moment vs range) but for the *transform design* question, not for pre- vs post-RoPE storage, and it never writes
  the G*theta_f dependence. The idea should cite Lemma 2 instead of re-proving the equivariant class.

**2606.24033 Block-GTQ "RoPE-Aware Bit Allocation for KV-Cache Quantization" (Liang, Zhang, Jia, Jun 2026) — partially_preempts C9.3.**
- Sec 2.1 / Lemma 2 (App. A.2): |K_Delta(q,k) - K_Delta(q,k_hat)| <= sum_i ||q^(i)|| ||e_k^(i)||, "no cross-block terms", "norm-based
  block statistics are RoPE-invariant — we can therefore compute the energy score in pre-RoPE coordinates". Block weight
  s*_i = E[||q^(i)||_2 ||k^(i)||_2] (already query-weighted — the report's "Block-GTQ key-energy + query weighting from OSCAR/Spend-Bits"
  description is wrong). Theorem 1: greedy integer allocation is exactly optimal for the separable objective with the TQ-MSE 4^-b law.
- Numbers: K-only logit MAE -32..80% over ten models; Llama-3.1-8B-Instruct K2V2 NIAH 70.6 -> 97.4, LongBench-EN 36.87 -> 53.31 vs
  uniform TQ-MSE; AIME24/25 K3V2 51.7/37.5 vs fp16 54.2/37.9. App. B.4: on MLA models (only 32 of 64 blocks rotary) "Block-GTQ's
  RoPE-aware advantage shrinks and per-channel KIVI-ScaleOnly becomes competitive".
- Does *not* treat per-channel range mixing, pre/post storage, or band structure of the gain; uses a rotation-based (TQ-MSE) local
  quantizer. C9.3's "reverse water-filling with |q_f|^2 sigma_f^2" is the continuous-rate version of their greedy integer allocation.

**2510.00028 "Rethinking RoPE Scaling in Quantized LLM: Theory, Outlier, and Channel-Band Analysis with Weight Rescaling" (Qiao et
al., Sep 2025; long version of Q-ROAR 2509.14391) — partially_preempts the YaRN-interaction half and the "axis-aligned grid vs
rotation" mechanism, heuristically.**
- Sec 3.2 "Interaction with quantization": Eq. 12 logit perturbation bound; Eq. 13-14 for a per-channel mid-rise quantizer on a rotated
  2D pair, "E||z_hat - z||^2 ~ Delta_i^2/6 * eta_i(m), eta_i(m) up when phi_i^scaled(m) induces axis mismatch"; two named effects
  "Dynamic-range dilation" and "Anisotropy shift: Per-channel quantization grids are axis-aligned; RoPE rotates the signal relative
  to these grids". Conclusion: "position-dependent logit error that is typically largest for high-frequency i and large |m-n|".
  Eq. 15 tail-inflation rho^A_i(m) = quantile of ||R(phi_scaled) u_i||_inf over ||R(phi) u_i||_inf (axis-aligned amplitude of the pair).
- Numbers: Table 1 GovReport LLaMA-2-7B 32K: Q-ROAR W4 5.833 vs FP16 6.069, AWQ-W4 6.302, RTN-W4 6.783; NTK/linear PI on quantized
  models "reaching perplexity values greater than 100"; YaRN s=8 keeps all W4 baselines close to FP16 (Table 2 Proof-pile).
- eta_i(m) is never derived; no mixing factor, no pre/post comparison, weights-only focus. But the "high-frequency pairs carry the
  position-dependent error" sentence and the axis-mismatch framing pre-date the idea.

**2610.03027 TaSQ "Tailoring the Quantization Space for 1-Bit KV Cache Compression" (Cheong, Son, Yoo, Oct 2026) — contradicts the
loose form of L9.1 ("pair-preserving VQ is invariant") and partially preempts C9.3's query weighting under RoPE.**
- Sec 3.3: "pre-RoPE VQ achieves lower reconstruction error across all 32 layers, with a 35% lower total reconstruction error"
  (Fig. 1d) vs post-RoPE VQ with a shared codebook; Fig 7: post-RoPE-fitted NovaKV error rises sharply beyond calibration length,
  pre-RoPE CQ/TaSQ only gradually. A learned pair-grouped codebook is *not* rotation-equivariant; L9.1 holds only for codebooks that
  are themselves SO(2)-invariant per pair (CommVQ) or polar.
- Sec 4: squared logit error (q_bar^T R_p delta)^2 = delta^T A_{t,p} delta, A = R_p^T q q^T R_p, averaged over calibration (t,p) and
  diagonalised -> channel weights W_h = diag(A_tilde_h). This is the position-averaged query-energy weighting of C9.3 in channel form.

### P2 — read

**2604.08426 "KV Cache Offloading for Context-Intensive Tasks" (YAKV; Bocharnikov et al., Apr 2026, v2 Sep 2026) — supports.**
Confirms the report's citation id (title differs from "YAKV"). App. L Table 10: Llama-3.1-8B-Instruct MultiNeedle pre-RoPE 93.36 /
post-RoPE 93.73; Qwen3-4B-Instruct-2507 Text2JSON 90.83 / 91.05; "We attribute this to the randomized Hadamard transformation used in
the HIGGS quantizer". Two task numbers only, no PPL, no MSE.

**2502.00527 PolarQuant (Wu et al., Feb 2025, v3 Jun 2026) — supports L9.1 as an empirical instance, no theorem.**
Fig. 1: "outliers generally appear in only one of the two dimensions rotated together by RoPE"; "Since rotation is a magnitude-preserving
transformation, these 2D sub-vectors inherit the magnitude characteristics seen in the pre-RoPE case"; aims "to preserve the benefits
of pre-RoPE quantization in reducing approximation errors while eliminating redundant computations". Sec 5: robust to RoPE base
{1e4, 5e5, 1e6} and to NTK scaling. (Note the asymmetric-pair observation is the premise of L9.2.)

**2505.20839 FireQ (Baek et al., May 2025) — partially_preempts the pair-level engineering of the mixing effect.**
"As RoPE intermixes paired channel distributions while leaving non-paired channels unaffected, we separate key matrix channels into
(i) outlier channels with their paired counterparts, and (ii) regular channel pairs"; CRS post-RoPE scales for group (i), RPN
pre-RoPE pair-norm normalisation for group (ii) with Theorem 3.1 (||(k_i^n/s_i, k_j^n/s_j)||_2 <= 1/alpha for all n with
s_i = s_j = alpha * max_n ||(k_i^n,k_j^n)||_2). Per-token quantization only; no frequency dependence.

**2608.14191 "KV Cache Compression Through the Lens of Transform Coding" (Laus et al., Aug 2026) — partially_preempts C9.3.**
Theorem 1 (white-noise + non-overloading max-based token scaling): D = D_K + D_V + R with
D_K = sum_{i,c} a_i^2 ||v_i W_O - o||^2 q_c^2 E[(delta k_ic)^2]; bits allocated per channel by reverse water-filling on calibration
data (AATC); ~5.8x compression near-lossless on Llama-3.1-8B / Qwen-2.5-7B. No RoPE; but a RoPE pair is two channels, so C9.3 adds
only the remark that phase does not enter the variance.

**ACL 2026 long 1732 "Resonating with RoPE: Spectral Quantization for High-Fidelity Key Cache Compression" (SpectrumQuant;
aclanthology.org/2026.acl-long.1732, not on arXiv) — partially_preempts the band-allocation vocabulary.**
Abstract: DCT "to convert these [RoPE-induced] oscillations into sparse spectral representations", "dominant frequency extraction,
hybrid bit-width allocation, and high-frequency pre-emphasis", fused Triton kernels. Mechanism appears to be a DCT along the token
axis of post-RoPE keys, not per-pair range analysis, but it claims bit allocation by RoPE frequency; must be read before claiming
"no one allocates bits by RoPE band".

**2606.02288 "Massive Spikes in LLMs are Bias Vectors" (Chen et al., Jun 2026) — contradicts the report's conjecture on where
outlier pairs live; suggests a sharper hypothesis.**
Fig. 3c: "the dominant sink features are disproportionately concentrated in high-index subspaces (i ~ 55 out of 64) ... the model
targets the lowest-frequency band"; "in ~80% of cases, the top-2 channels form an exact RoPE 2D pair (i, i+d/2) ... By activating
both elements, the model preserves magnitude robustness even as energy rotates". Under L9.2 these spikes (low frequency, symmetric
pair) produce *no* pre/post difference, so the KVQuant gain must come from mid/high-frequency *asymmetric* pairs — a testable
refinement, and a reason the gain shrinks with larger RoPE base (theta_min falls). Massive-Values (2502.01563, Sec 4.1) likewise
places concentrated massive values in "low-frequency regions" of Q/K.

### P3 — supporting / adjacent

- **2609.28029 Tensor decomposition of KV caches (Krishnan & Schulz, Sep 2026) — supports.** Table 8: post-RoPE keys lose 41-64% of
  pre-RoPE compressibility (Tucker/CP/TT/t-SVD) on Mistral-7B-v0.3 and LLaMA-2-13B; a frozen-allocation control widens the gap.
  Low-rank, not quantization.
- **2603.18017 Frayed RoPE (Wertheimer et al., ICLR 2026) — tangential, adjacent math.** Lemma 1/Theorem 1: for rank-1 X = u v^T,
  RoPE shrinks the top singular value by between sqrt(2) and sqrt(d) and raises stable rank by a factor between 2 and d as n -> inf,
  "the precise increase factor depends on the energy distribution of v across the rotation planes". Same ingredients (per-plane
  energy, sequence length) as L9.2, different statistic (spectral norm vs max-abs range).
- 2506.18879 CommVQ — Property 1 commutativity of 2x2 codebook blocks with R_m; the canonical equivariant codebook for L9.1.
- 2506.04985 FPTQuant — "pre-RoPE transforms ... commute with RoPE's R so that they can be applied before RoPE and merged into W_q,W_k";
  2609.38121 WUSH-KV App. A: folding a key transform before RoPE with RMSNorm leaves "only paired sign-flip transforms"; full
  pre-RoPE transforms need position-dependent compensation. 2609.36760 QuantMLA repeats the commutation constraint for MLA.
- 2506.19505 AnTKV App.: pre-RoPE K "exhibits smaller inter-cluster distances and lower intra-cluster variance" (Fig. 8) — VQ pre-RoPE
  preference, same caveat as TaSQ for L9.1.
- 2609.05764 Interface-Aware KV: Hadamard placed after RoPE "because the two transformations do not commute"; cancels in the logit.
- 2507.11273 KV-Latent: "low-numbered channels of the RoPE function contribute the majority of the noise" — but in the
  reduced-dimension-RoPE setting, not quantization; the report's use of it is a loose analogy.
- 2503.01586 EliteKV: per-head RoPE-frequency preference (RoPElite) — frequency-wise importance heterogeneity, not quantization.
- 2608.25188 "Transforms for LLM Quantization: The Great Inversion" (survey) summarises KVQuant/RotateKV pre-RoPE choices in the same
  words ("rotary embedding mixes channel pairs by a position-dependent angle"); no analysis.
- 2608.04074 Spend-Bits, 2605.17757 OSCAR, 2605.06675 RateQuant, 2605.08317 RDKV: query-covariance / rate-distortion allocation at
  head/layer/token granularity; none per RoPE pair; none discuss pre/post RoPE.
- 2603.27467 TurboAngle, 2502.02617 PolarQuant (Han et al.): angle quantization of *random-rotated* pairs (uniform on the circle) —
  rotation-invariant quantizers of a different kind; Theorem 1 of 2502.02617 is a Gaussian error bound, no RoPE.
- 2303.03106 "Rotation Invariant Quantization" (weights, cs.LG) and the cs.IT polar/phase-quantization literature (e.g. 2205.05850)
  do not transfer: nothing on axis-aligned range quantizers under known rotations.

## What remains unclaimed (as far as this sweep can tell)

1. A closed-form mixing factor m(G*theta_f, max1/max2) for per-channel/per-group max-abs ranges, and the resulting prediction that
   the pre-RoPE gain is a monotone function of G*theta_f concentrated in high-frequency pairs, growing toward low frequencies with G.
2. The pair-asymmetry condition (max1 >> max2) as the *other* necessary ingredient, and the consequence (via 2606.02288) that the
   biggest spikes contribute nothing — i.e. a prediction about *which* pairs drive KVQuant's 0.82.
3. A single accounting that puts KVQuant App. H (per-channel: pre wins), KVQuant App. P (per-token: post wins), AQUA-KV/YAKV
   (RHT: tie), RotateKV Table 2 (RHT+permutation: pre wins) on one axis (invariance of the scale-setting statistic).
4. The YaRN re-mixing effect (band-wise theta_f shrinkage changes which pairs mix inside a group). 2510.00028 has the
   weight-error-through-RoPE half, not the storage-quantizer half.

## Corrections for the report text

- YAKV's arXiv title is "KV Cache Offloading for Context-Intensive Tasks" (2604.08426, App. L Table 10); numbers are task accuracies,
  not PPL.
- Block-GTQ already weights by query norm (s*_i = E[||q^(i)|| ||k^(i)||]); the report's framing of C9.3 as "Block-GTQ + query
  weighting" overstates the delta. The honest delta is continuous-rate reverse water-filling + the |q_f|^2 sigma_f^2 form + the
  phase-irrelevance remark, all of which are also implicit in AATC Theorem 1.
- "Pair-preserving VQ" is not equivariant in general (TaSQ 35% figure); restrict L9.1 to SO(2)-invariant codebooks (CommVQ, polar).
- Cite 2608.13365 Lemma 2 for the equivariant class; cite AQUA-KV for the earliest explicit quantizer-dependence statement.
