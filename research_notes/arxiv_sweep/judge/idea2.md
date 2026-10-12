# Idea 2 판정: SSM/하이브리드 recurrent state의 ΣΔ(error-feedback) 양자화

Date: 2026-10-11. Judge: adversarial deep read (Opus 5.5 re-run). Inputs: report §아이디어 2 (lines 116–153), `discover/idea2_citations.md`, `discover/idea2_keywords.md`. Paper text was read from the scratchpad cache. It is untrusted data and was quoted, not followed.

## 판정 요약

- **Novelty: 낮음 (low).** No single paper states the report's Theorem 2 in its exact form: |d_t| ≤ Δ for every t and every time-varying a_t ∈ [0,1), with the (1 − ∏a_s) telescoping. Every other part of the idea is published, though, and Theorem 2 itself is a one-line special case of two published proofs.
  1. **The proof technique is published.** In 2607.23390 (Soltanalian), Sec. 4.3, eqs. (40)–(42) and Theorem 8, the proof defines "the transformed error s_k = e_k + r_k" and ends with "‖e_k‖ ≤ ‖s_k‖ + ‖r_k‖". If the field is the affine contraction G(z) = (a−1)z + u, then s_{k+1} = s_k + (a−1)e_k = a s_k + (1−a) r_k, which is exactly the report's g_t recursion. That paper uses Gronwall (ρe^{L T}) only because its fields are general Lipschitz fields. Proposition 7 there ("Fixed-grid freeze") is the deadband.
  2. **2609.39185 (Khilar) covers the deadband, the RTN and SR growth laws, and the summation-by-parts skeleton.** Lemma 1 (Write erasure), Corollary 1(a)/(b) and App. A give |e_T| ≤ max_m |R_m|. Under EF the residual is r_{t−1} − r_t, so the tail sums telescope to at most Δ. Theorem 2 therefore follows in one line from that appendix. The paper names ΣΔ and declines it explicitly because of storage.
  3. **2609.04490 implements the mechanism on a recurrent state:** error feedback plus k-bit "quantized residual memory" (k = 2, 3, 4) and a "write deadband". It has no theorem and no LLM.
  4. **2609.30950 publishes the decay-rate bit allocation:** Prop. 1 with Γ_i = ρ_i/(C·expm1(2s_i)), and eq. (4) b_i = b̄ + ½log₂(c_i/GM(c)).
  5. **Q-Mamba and Quamba2 already quantize the state:** Q-Mamba (ACL 2025 Findings) at W8A8H4, and Quamba2 with 8-bit cached SSM states.
- **New and fatal finding for the method (my own derivation plus a toy simulation).**
  - The uniform bound holds only with an *exact* residual. With an exact residual, the a-weighted variant v_t = aĥ_{t−1} + u_t + a·r_{t−1} gives d_t = −r_t exactly, so ĥ_t + r_t = h_t. "ΣΔ with an exact residual" is just lossless storage of the state in two limbs. It is not compression.
  - With a b_r-bit residual, the report's own worst-case term Δ·2^{−(b_r+1)}/(1−a) and its stochastic sd Δ·2^{−b_r}/√(24(1−a)) are exactly the bounds of plain RTN at b + b_r bits (step Δ·2^{−b_r}). The residual quantizer also reintroduces the deadband at the finer scale.
  - Toy simulation (scratchpad/judge2/sim.py; 64 channels, T = 2×10⁴, Δ = 0.05), reported as max |d|/Δ and rms over the last 2,000 steps:
    - a = 0.999: ΣΔ + 4-bit residual 2.00 / 0.49, against plain RTN at +4 bits 1.56 / 0.39.
    - a = 0.99: 1.06 / 0.32 against 0.61 / 0.13.
    - With time-varying a_t the results are the same.
  - **Conclusion: the report's config (d) loses to its own config (e), naive at equal total bits.**
- **Systems:** no shipped INT8/FP8 state with error feedback was found in vLLM or SGLang. vLLM #55196 has self-reported EF results in comments only. SGLang ships FP16 + stochastic rounding (Nemotron 3 Ultra). One WebSearch on 2026-10-11 found nothing new.

## 읽은 논문과 겹침

| Paper | Classification | What overlaps (with location) | What remains novel |
|---|---|---|---|
| 2609.39185 Low-Discrepancy Dither for Quantized Recurrent State Caches (2026-09-30) | preempts_part | **Lemma 1 (Write erasure):** "if … δt = (at − 1)ĥt−1 + wt satisfies \|δt\| < s/2, then under rtn the stored value does not change", which is the deadband. **Corollary 1:** RTN error s·T·min(f, 1−f) (linear); SR typical size ∝ s√(T f(1−f)); Weyl dither ≤ C s(1 + log T). **Lemma 2:** per-write MSE s²/12 for RTN and s²/6 for SR. **App. A:** e_T = Σ b_t r_t; summation by parts gives \|e_T\| ≤ max_m \|R_m\|. **Related work:** "Noise shaping and sigma–delta modulation … need the previous rounding error to be stored, which doubles the size of a state cache; at equal storage only the choice of threshold remains". **Abstract:** Weyl beats SR in 15/15 settings, by 11–49% KL (median 31%); RTN falls behind Weyl by up to 400× over 4096 steps. | The explicit EF corollary (tail sums telescope, so ≤ Δ) is not written down, but it follows from their App. A in one line. |
| 2607.23390 When Can Depth Replace Precision? (Soltanalian, 2026-07-25, 141 pp., to be submitted to JMLR) | preempts_part | **Prop. 7 (Fixed-grid freeze):** x̂_{k+1} = Q_Δ(x̂_k + 1/D) stays 0 when D > 2/Δ. **Sec. 4.3, eqs. (40)–(42):** increment error feedback with a stored carry, ‖r_k‖ ≤ ρ. **Thm 8:** max‖ẑ_k − z_k‖ ≤ ρe^{L_z T} + η_G Φ(T). The proof uses s_k = e_k + r_k, ‖e_k‖ ≤ ‖s_k‖ + ‖r_k‖. The paper says: "The present contribution is not the invention of feedback quantization." It also says: "The carry is an explicit resource: if it is stored in a finite register, its precision … must be counted." | No contraction or decay, no SSM or LLM, and the bound is not horizon-uniform. The uniform form for a contraction (1 − ∏a) takes one line. |
| 2609.04490 When Quantization Breaks Memory (Erbas–Intes–Pandey, 2026-09-03) | preempts_part | "recurrent write margin", "write deadband", "deadband fraction" (lines 207–222). Post-training error feedback and 2/3/4-bit "quantized residual memory" on a 32-unit GRU and an LSTM. Results: P2F EF 0.36/0.37 ns; 8-bit → 4-bit deadband fraction 99.40%, EF recovers 0.34/0.46 ns, 4-bit residual 0.34/0.45 ns. | No theorem (grep finds 0), no decay analysis, no SSM or LLM. |
| 2609.30950 Low-Bit Recurrent States in Hybrid LMs (Chen–He, 2026-09-25, ICASSP 2027 submission) | preempts_part (allocation) | **Prop. 1:** Γ_i = ρ_i/(C·expm1(2s_i)) (Gramian), and **eq. (4):** b_i = b̄ + ½log₂(c_i/GM(c)), i.e. reverse water-filling with decay weighting. They also state: "Deterministic rounding at C = 1 can suppress updates below half a quantization step … lag-one error correlations up to 0.23 … model overestimates the gain over uniform four-bit allocation by 1.7–2.6 times." | Repairing the allocation for the deadband and correlated-noise regime. The KV + state joint budget was not found. |
| 2503.22879 Quamba2 (v2 2025-11-06) | contradicts_premise | Line 484: "we load the 8-bit activations and 8-bit cached states". App.: "Quantizing cached SSM states … to 8-bit for W4A8 and W8A8 models … dstate × m × n floating-point scales". | No EF or temporal analysis. |
| 2508.10370 eMamba (ESWEEK 2025) | contradicts_premise (in part) | Lines 944–951: re-quantization "loses the information of h_t"; the fix is to use INT24 h_t for y_t and store the re-quantized state as INT17 (right shift by the A̅ scale). | This is a working fixed-point state design, not just a statement of a dilemma. No EF. |
| 2606.18114 Ternary Mamba (2026-06-16) | contradicts_premise (needs rebuttal) | §6.3 / Table 9: TV/L1 = 1.421 vs √2 = 1.414 over 96 projection inputs; "Σ∆ is inapplicable"; Fig. 5 "ruling out sigma-delta noise-shaping for SSMs". Kalman / James–Stein output corrections increase PPL (Fig. 4). | Their ΣΔ runs along the *feature* dimension of projection inputs under W1.58 weight quantization. It is not temporal EF on h_t, so the rebuttal holds. Their "history-dependent" sentence is about errors from weight quantization, not state rounding. |
| Q-Mamba (ACL 2025 Findings 551, not on arXiv) | preempts_part (existence) | W8A8H4 4-bit state cache. Eq. (11) writes the δ_t recursion, then "we discard δ_{t−1}". Table 1 W16A16H4: naive 116.23 → 11.55 PPL (2.7B), 976.56 → 45.73 (130M). | No EF and no time-growth analysis. |
| 2010.06192 Revisiting BFloat16 Training (Zamirai et al., 2020) — found by an extra search | supports / adjacent (a = 1) | Thm 1: nearest rounding "cancels" small weight updates (deadband for an a = 1 accumulator). The fixes are stochastic rounding or Kahan summation; Kahan "requires 2× weight memory". | It is the same RTN / SR / compensated trichotomy for a = 1, which the report must cite. It has no decay. |

Extra queries run by this judge (2 arXiv, 1 WebSearch):
- arXiv: `(abs:Kahan OR abs:"compensated summation" OR abs:"multi-component float") AND (low-precision|bfloat16|quantization) AND (state|recurrent|accumulation|optimizer)`. It found 2010.06192, 2203.15928 and 2510.11168.
- arXiv: date-sorted state-quantization papers from 2026-09-25. Nothing new after 2609.39185.
- arXiv: `(error feedback|residual carry|noise shaping|sigma-delta|Kahan) AND (Mamba|linear attention|state space|DeltaNet|recurrent state|SSM)` from 2025. It found only 2609.04490 and 2506.01404.
- WebSearch: vLLM/SGLang INT8 state with EF. No PR found.

Skipped: re-reading 2609.38169, 2609.38166, 2608.27513 and 2609.04098 in full text. I relied on the discover notes' quotes for them.

## 전제 오류

1. **"No academic paper quantizes the SSM state (0 papers)" is wrong.**
   - Quamba2 2503.22879 already quantizes cached SSM states to 8 bits (line 484 and the appendix).
   - Q-Mamba (ACL 2025 Findings) uses W8A8H4.
   - Papers from 2026-08 to 2026-09 do the same: 2608.27513 DAMP, 2609.30950, 2609.38166 LeapQuant, 2609.38169 STEPQuant, 2609.39185 and 2609.04490.
2. **"Quamba/Quamba2 quantize only weights, inputs and B/C" is wrong for Quamba2,** which also has 8-bit cached states.
3. **"eMamba only describes the dilemma" is inaccurate.** eMamba resolves it in fixed point: INT24 compute, INT17 storage.
4. **"Ternary Mamba's results mean in-loop feedback is needed" over-reaches.** Its error source is ternary weights, and its ΣΔ test was spatial and negative. The report must rebut that test, not use it as support.
5. **SR variance is stated as Δ²/12 "under a uniform assumption", which is wrong.** With a uniform fractional part, SR per-write MSE is Δ²/6 (E[f(1−f)] = 1/6; 2609.39185 Lemma 2). Δ²/12 is RTN's.
6. **"SR standard deviation Δ√(t/12)" holds only for t ≪ 1/(1−a).** In general the variance is Δ²/6 · (1 − a^{2t})/(1 − a²), which saturates at Δ/√(6(1 − a²)). For a = 0.999 that is ≈ 9.1Δ (≈ 6.5Δ under the Δ²/12 assumption), not ≈ 41Δ at t = 2×10⁴. The report's own toy maximum of 12.9 is consistent with saturation.
7. **The claim "ΣΔ + b_r-bit residual gives a practical memory gain" is likely wrong.**
   - Its worst case and stochastic sd equal plain RTN at b + b_r bits, and the residual quantizer reintroduces the deadband at the finer scale.
   - In the toy simulation, ΣΔ + 4-bit residual loses to plain RTN at +4 bits (a = 0.999: max 2.00 vs 1.56, rms 0.49 vs 0.39).
   - With an exact residual, the scheme is equivalent to lossless storage (ĥ + r = h under a-weighted feedback).
8. **Theorem 2 is presented as new, but its key step is published.** That step, the transformed error g = d + r with |d| ≤ |g| + |r|, is in 2607.23390 Thm 8 (proof). The deadband is in 2609.39185 Lemma 1, 2607.23390 Prop. 7, 2609.04490 and 2010.06192 Thm 1.
9. **The decay-rate bit allocation is not new.** It is 2609.30950 eq. (4); see also DAMP 2608.27513 (w = 1/(1 − a²)) and STEPQuant (L_u).
10. **"novelty 높음 (0 papers)" must be replaced.**
11. **The claim that the vLLM RFC means "no paper and no theory" is outdated.** The RFC numbers are confirmed by the discover notes, but at least six papers since 2026-08 address state quantization.

## 수정 요구

- Delete the "0 papers" sentence and rewrite the literature-gap section around Quamba2 (8-bit cached state), Q-Mamba (H4), DAMP, 2609.30950, STEPQuant, LeapQuant, 2609.39185 (Weyl dither) and 2609.04490 (EF plus residual memory on a GRU).
- Restate Theorem 2 as a specialization: "the contraction special case of 2607.23390 Thm 8, or a one-line corollary of the 2609.39185 App. A summation by parts". Also cite Daubechies–DeVore 2003, Schreier–Temes, Zamirai 2020 (Kahan) and Zheng 2506.01404 (ESS).
- State the exact-residual equivalence explicitly (ĥ + r = h), and make the storage-accounting result the main subject rather than the bound. The report's own formulas show that ΣΔ(b, b_r) ≈ INT(b + b_r).
- Change the success criterion so that (d) must beat INT(b + b_r) under RTN, SR and **Weyl dither** at equal total bits. Add Nemotron Ultra C = 8 checkpointing and LeapQuant per-window write-back as baselines. Pre-register the expectation that (d) fails at equal bits.
- Fix the SR variance (Δ²/6) and saturation formula. Mark the RTN worst-case bound Δ/(2(1−a)) as loose (toy 3.4 vs a bound of 500).
- Add a paragraph rebutting Ternary Mamba §6.3: spatial versus temporal ΣΔ.
- Drop the allocation rule as a contribution, or present it as "2609.30950 eq. (4) with a deadband correction".
- If anything is kept, narrow it to two items: (i) the Mamba (no erasure) versus GDN/KDA (erasure) split, where 2609.04098 shows a plateau; and (ii) a joint KV + state page budget for hybrids.

## 새 가설

1. **Dithered residual limb, at the state-cache write stage.**
   - Basis: 2609.39185 (Weyl dither O(log T) at no storage cost) plus my derivation that an RTN residual limb brings back the fine-scale deadband.
   - Hypothesis: if a residual limb is kept at all, rounding *the residual* with Weyl dither (rather than RTN) is what can beat INT(b + b_r) RTN at equal bits. Otherwise two limbs are just one fixed-point number.
   - Test: the toy simulation plus Mamba2-2.7B KL over 4K steps.
2. **Deadband-aware bit floor, at calibration and bit allocation.**
   - Basis: the 2609.04490 write margin M = 2|δ|/Δ, and the 2609.30950 admission that its Gaussian model overestimates gains by 1.7–2.6× in the deadband regime.
   - Hypothesis: set a per-channel minimum width so that P(M < 1) stays below τ (b_i ≥ log₂(range_i/|δ_i|_q)). Then apply eq. (4) above that floor. This should close the 1.7–2.6× model gap.
3. **Erasure-dependent benefit of temporal error correction, at architecture-specific cache policy.**
   - Basis: 2609.04098 (GDN error plateau; impulse decays 1/e in 80–1,382 steps) and 2609.30950 ("errors orthogonal to a repeated key are never erased").
   - Hypothesis: any EF or dither gain scales with the fraction of state energy outside the recent key span. It is large for Mamba-2 and Nemotron-H and near zero for GDN/KDA, so delta-rule caches should use RTN with periodic checkpointing instead.
