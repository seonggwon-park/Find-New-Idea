# P-LOOP: prior-art check (merged with P02 damping-as-stabilizer and P09 overtraining-raises-loop-gain)

**Verdict: survives_narrowed. Novelty 2/5.**
Two parts are already published. The loop-gain/overload mathematics for stochastic rounding is in QuIP Thm 7 and in the Saab-group OPTQ papers. Feedback-aware block scales are in reference GPTQ (dynamic groups), OSFP4 and Schur Replay. What is left is mostly empirical: (i) how much overload occurs in unrotated INT2/3 GPTQ under static scales; (ii) a bit-width-dependent damping rule; (iii) a clipped Babai bound that is modest as theory.

## Closest work (read in full text)

| id | date | what it already contains |
|---|---|---|
| 2307.13304 QuIP (NeurIPS'23) | 2023-07 | Sec. 5.2 "A Bound for Rounding to a Finite Grid": clamped LDLQ can be "asymptotically worse" (Fig. 4, coherent H counterexample). Algorithm 5 constrains the feedback matrix with `e_i^T R^T R e_i <= 1 + c` for all i (a bound on the column norms of the feedback matrix, i.e. the loop gain `g_i^2 <= c`) and uses stochastic rounding plus a scale rho (headroom). **Thm 7**: "with probability at least 1-delta, all the quantized weights will be in range (no overflow or need for clipping)" with error O~(tr(H^{1/2})^2 ||W||_F^2 / (n^2 4^b)). l.560: "This clamping issue does not seem to arise in practice"; l.3984: "especially as model size increases". |
| 2508.04853 Zhang/Zhang/Colbert/Saab (v2 2026-04-09) | 2025-08 | Remark 3.7: "additional dynamic range is needed to absorb errors that arise from projection-based cancellation", i.e. the headroom. Deterministic worst case is ||w||_inf + O(sqrt N) delta and sharp (D.2). **Cor 4.7** (SR): max abs(W-Q)_ij <= delta sqrt(2 pi (p log N + p' log N')) * sqrt(max_j ||X_j||^2/lambda + 1). This makes headroom an explicit function of the damping lambda. **Remark 4.8**: 2^{b-1} delta >= ||w||_inf + O(sqrt(log N)) delta suffices; "the relative gap ... is more significant in the low-bit regime." |
| 2609.31560 George/Saab | 2026-09 | Besides Eq. 35 (lambda from generalization), l.494-495 gives an l_inf bound on q in terms of lambda, X, w, so that b can be chosen "so that with high probability no clipping will need to be performed." |
| 2610.08231 OSFP4 (Ben David, Meir, Ordentlich) | 2026-10-06 | NVFP4 block scales optimized with the SIC (GPTQ) feedback in the loop, analyzed through a dithered quantizer. l.1414-1418: "The main challenge in choosing gamma is that we do not know what will be the input to the quantizer before the feedback quantization procedure was executed." Algorithm 1 chooses each block scale on the feedback-updated input and estimates the intra-block feedback. |
| 2609.36654 Schur Replay | 2026-09-29 | NVFP4 block-scale selection that "reproduces the GPTQ updates caused by each block scale", "capturing feedback within the 16-value block". |
| 2507.18553 GPTQ = Babai | 2025-07 (v 2026-05) | Thm 5 assumes no clipping. l.720: clipping violates the bound. **l.723-724: "enforcing no-clipping by simply increasing scales is counterproductive: larger scales enlarge the bound, and the resulting errors can exceed those of a clipped scheme such as MSE."** This is prior negative evidence on headroom scales. l.876: clipped grids are "immediate next steps". |
| 2602.02126 Two-Stage Grid Opt. | 2026-02 | Refits group scales after GPTQ, with the integers frozen. Closed-form coordinate descent. |
| 2510.06213 Training dynamics vs PTQ | 2025-10 (v 2026-01) | Degradation is driven by LR decay, "largely independent of training data scale". The trend replicates with AWQ and LLM.int8() (App., l.1562-1566), so it is not specific to GPTQ. |
| Code: IST-DASLab/gptq gptq.py l.120-123 | - | With the default static_groups=False, `find_params(W[:, i1+i : i1+i+groupsize])` is called at each group start on the **feedback-updated** W. Group scales therefore already include all inter-group feedback. |
| Code: llm-compressor modifiers/gptq/gptq_quantize.py | main | Scale and zero_point are precomputed by the observer and passed in fixed, so these are static scales from the original W. |

## Premise check
- 2507.18553 Sec. 6 ("immediate next steps", MXFP4/NVFP4 "essentially no-clipping") and Sec. 5 (clipping violates Thm 5): VERIFIED (l.876-881, l.720-721). The candidate omits l.723-724, where enlarging scales to avoid clipping is reported as counterproductive.
- 2605.13768 high-resolution overload assumption (l.717-727) and low rates left open (l.1625-1630): VERIFIED.
- 2508.04853 "cannot be made fully rigorous": VERIFIED, but it refers only to the deterministic case. The same paper proves the SR finite-alphabet case (Cor 4.7, Rem 4.8), and that bound depends on lambda. The gap claim understates the paper.
- 2608.25188: 123.9 vs 6.33 (Llama-2-70B, 2 bits) VERIFIED (l.4256-4257). The "permutation alone worth 74 perplexity" figure is for **OPT-125m**, not Llama-2-70B (l.4262).
- QTIP 2406.11235: the phrase "worse than expected on Llama 3" is NOT FOUND in the fetched version (updated 2025-06-18). That version reports only a catastrophic collapse when layer-0 v_proj of Llama 3 70B is quantized, and says the paper is "on what to round with and not how to round". Earlier versions were not checked. Treat this premise as unverified.
- 2609.31560 Eq. 35 does not depend on bit-width and is validated on synthetic data: VERIFIED (l.744-760). The same paper also has an l_inf/no-clipping bound in lambda.
- "Block scales computed from the original W": **partly false**. It is false for reference GPTQ and GPTQModel-style dynamic groups, where only intra-group feedback is unseen. It is true for static_groups=True and for llm-compressor (the vLLM production path, including NVFP4/MXFP4).
- The candidate's 15.7% overload toy uses damping 1e-4, about 100x below the default 1%. At the default 1% damping its own proxy is 0.031 rather than 0.152. QuIP reports clamping as rarely relevant in practice, so the effect size at default settings is unknown.

## What remains
1. A measured decomposition, on real LLMs at INT2/INT3 with static scales vs dynamic scales vs RHT, of how much of the GPTQ/LDLQ excess error is overload. None of the papers above reports this. QuIP asserts that it is negligible.
2. A bit-width-dependent damping rule lambda*(b, H) validated on real LLMs. The theoretical link between lambda and headroom exists (2508.04853 Cor 4.7). The trade-off between tr(D) and headroom, the comparative statics dlambda*/dDelta > 0, and the experiments do not.
3. A clipped Babai bound (tr(D) Delta^2/12 + overload tail via the per-coordinate g_j^2). This is a refinement of QuIP Thm 7 and 2508.04853 Cor 4.7 from a uniform bound to a per-coordinate one, and from a no-overflow guarantee to an explicit penalty. It is incremental.
4. P09: whether the RTN/GPTQ damage ratio tracks the loop gain across checkpoints. 2510.06213 weakens the premise: the effect is LR-driven and appears with AWQ and int8 too.

## Significance
For a main-track reviewer, low. The theory reads as a reparameterization of QuIP Thm 7 (column-norm constraint on the feedback matrix) and Saab et al. Cor 4.7. The headroom-scale method is preempted for NVFP4 (OSFP4, Schur Replay), already partly built into default GPTQ for INT-g128, and reported counterproductive by 2507.18553. Unrotated 2-bit GPTQ is not a competitive regime. A short workshop note on bit-dependent damping plus an overload measurement is the realistic ceiling.

## Queries run
See the structured result.
