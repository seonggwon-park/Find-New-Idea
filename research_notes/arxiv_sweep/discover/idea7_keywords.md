# Idea 7 — keyword/adjacent-field sweep (prior art and evidence)

Idea under test: "quantization-cell theorem" — for a group-uniform quantizer with step Δ and a base weight whose in-cell offset u ~ U[−½,½), a post-training delta D with |d| ≪ Δ is requantized to D̃ = Q(W+D) − Q(W) with E[D̃] = D, Var = Δ|d| − d², SNR ≈ ‖D‖_F²/(Δ‖D‖_1) ≈ mean|d|/Δ; this one scalar is claimed to unify LoRA-merge loss, unlearning reversal under 4-bit, and quantization-conditioned attacks; Design 1 = cell-aligned sparse deltas D ∈ {−Δ,0,+Δ}; Design 2 = secret-seed subtractive dither as a defense.

Sweep date: 2026-10-10. Sources: arXiv API (export.arxiv.org), full texts via `scripts/arxiv_fetch.py`, WebSearch for 2026 venue info. Downloaded text is quoted, not trusted.

## Bottom line

- The probabilistic core of Theorem 7.1 (uniform in-cell phase ⇒ P(cross) = |d|/Δ, E[jump] = d) is already written down, in a QAT framing, by **2609.21834** (math.OC, Sept 2026): "conditional phase θ_i|q ≈ Unif[−½,½]", crossing probability "p_i = η|u_i|/s_i", and Corollary 6 "E[q_i⁺ − q_i | q] = −s_i sign(u_i) p_i". They do not state the variance Δ|d| − d², the SNR scalar, or any link to merge/unlearning/attacks. **Partially preempts the math.**
- The "delta below the bin is erased; make it cross the bin" story is now a 2026 cottage industry in unlearning: **2605.15138 (MANSU)** has Lemma 1 (quantization survival), Proposition 1 (sparsity–permanence tradeoff: per-parameter update ≤ sqrt(2ε_r/(|C|F̄_C))), Proposition 2 (NF4 amplifies floor-crossing updates) and a *magnitude floor that rescales every update to clear the NF4 bin* — this is Design 1 (cell-aligned sparse deltas) for unlearning, minus the exact {−Δ,0,+Δ} integer-code merge and minus any SNR predictor. **2601.15538 (QUAIL)** and **2602.13151** make the same Δ/2 crossing argument heuristically.
- For the **merge** sub-claim, **2609.04526 (Scale-QLoRA)** explicitly rejects the "noise swamping a small update" reading when the base is *already on the grid* (QLoRA / native NVFP4 / MXFP4): an on-grid value is a fixed point of nearest rounding, so a sub-step delta is returned to W exactly (not unbiasedly), and they measure cos(Q(W+∆) − W, ∆) = 0.017. This is a direct scope restriction: Theorem 7.1's uniform-offset assumption fails when u ≡ 0, which is precisely the QLoRA-merge regime the report's experiment 1 proposes (Llama-3-8B QLoRA adapters). **Contradicts in that regime.**
- For **Design 2 (dither defense)**: no paper tests dithered/stochastic rounding against quantization-conditioned attacks (confirmed by arXiv and web search). But (i) the closest randomized defense — Gaussian pre-noise — is reported broken by outlier injection in **2605.15152**: "utility begins to drop clearly from σ = 10⁻³, while the attack remains effective"; (ii) deterministic rounding-flip defenses already exist (**EFRAP 2405.12725**, CVPR 2024; **QuantGuard 2606.29239**, with a defense-aware adaptive attacker); (iii) seeded uniform dither of DNN weights with dither cancellation at decode is textbook in **1802.02271** (universal DNN compression) — the mechanics are not new, only the security use.
- Two 2026 papers offer *competing theories* of unlearning reversal that experiment 2 must control for: **2605.02196 (DurableUn)** Proposition 1 (gradient-norm / sharpness bound: Q-INT4 ≥ FA + (M/ρ − Lρ)δ − Lδ²/2) and **2609.27355** Theorem 1 (retain/forget curvature ratio F_f/F_r along the minima-connecting line); **2510.00761** (ICLR 2026) attributes robustness to optimizer "grade" (ZO/sign updates ⇒ flatter basins) — note that sign-type updates also have larger mean|d| per coordinate, so the SNR scalar and the basin story are confounded.

## Queries run (arXiv API, `scripts/arxiv_search.py`)

Seed / keyword (2023-01-01 → 2026-10-31, relevance):
1. `abs:quantization AND abs:unlearning`
2. `abs:quantization AND abs:attack AND abs:(backdoor OR malicious) AND abs:(LLM OR "language model")`
3. `abs:LoRA AND abs:merge AND abs:quantized`
4. `abs:LoRA AND abs:(merging OR merged) AND abs:quantization AND abs:(QLoRA OR "4-bit" OR "low-bit")`
5. `abs:dither AND abs:quantization AND abs:("neural network" OR LLM OR weights OR "language model")`
6. `abs:"stochastic rounding" AND abs:(defense OR defence OR security OR attack OR backdoor) AND abs:quantization` (0 hits)
7. `abs:"quantization-aware" AND abs:(unlearning OR "safety alignment") AND abs:robust`
8. `abs:delta AND abs:compression AND abs:quantized AND abs:"fine-tuned"`
9. `abs:preimage AND abs:quantization` (only math.SG noise)
10. `abs:"task vector" AND abs:(quantization OR quantized)`
11. `abs:(quantization OR quantized) AND abs:("safety alignment" OR "safety fine-tuning" OR "safety-aligned") AND abs:(degrade OR removes OR undo OR revert OR fragile)`
12. `abs:("model editing" OR "knowledge editing") AND abs:(quantization OR quantized OR "4-bit")`
13. `abs:(quantization OR quantized) AND abs:("secret" OR "random seed" OR "unpredictable" OR "randomized") AND abs:(defense OR defence OR attack OR backdoor) AND abs:(LLM OR "language model" OR weights)`
14. `abs:("quantization interval" OR "rounding interval" OR "quantization cell" OR "rounding boundary" OR "rounding boundaries") AND abs:(LLM OR "language model" OR "fine-tun")`
15. `abs:(QLoRA OR "quantized base") AND abs:(adapter OR LoRA) AND abs:(merge OR merging OR "merged into") AND abs:(degradation OR loss OR mismatch OR "performance drop")`
16. `abs:("integer" OR "lossless") AND abs:(LoRA OR adapter OR "low-rank") AND abs:(merge OR merging) AND abs:(quantized OR quantization)`
17. `abs:(unlearning OR "safety") AND abs:(quantization OR "4-bit" OR compression) AND abs:(recover OR recovered OR recovery OR relearn OR resurface OR revive)`
18. `abs:(QA-LoRA OR "quantization-aware low-rank" OR "quantization-aware LoRA" OR "integer adapter" OR "integer LoRA") AND abs:(merge OR merging OR lossless)`
19. `abs:(unlearning OR forget OR forgetting) AND abs:(quantization OR quantized OR "4-bit") AND abs:(sign OR sparse OR "large-magnitude" OR "magnitude floor" OR "bin boundary" OR "cross")`
20. `abs:("quantization-conditioned" OR "quantization conditioned" OR "quantization-triggered" OR "quantization triggered" OR "quantization-activated") AND abs:(defense OR defence OR detection OR mitigation OR repair)`
21. `abs:(quantization OR quantized) AND abs:("fine-tuning attack" OR "harmful fine-tuning" OR "jailbreak fine-tuning") AND abs:(delta OR "weight difference" OR "task vector" OR compress)` (0 hits)

Date-sorted, `--from 20250601 --sort date`:
22. `abs:quantization AND abs:unlearning`
23. `abs:quantization AND abs:(backdoor OR "quantization-conditioned" OR "quantization-triggered")`
24. `abs:(LoRA OR adapter) AND abs:(merge OR merging) AND abs:(quantized OR quantization OR "4-bit")`
25. `abs:(delta OR "fine-tuning update" OR "task vector") AND abs:(quantization OR requantization OR requantized) AND abs:(lost OR erased OR survive OR vanish OR "rounding")`

Adjacent-field vocabulary (cs.LG / cs.IT / eess.SP / stat.ML / math.OC / cs.CR; 2018→2026 where noted):
26. `abs:("subtractive dither" OR "subtractively dithered" OR "dithered quantization" OR "dithered quantizer") AND (cat:cs.LG OR cat:cs.IT OR cat:eess.SP OR cat:stat.ML OR cat:cs.CR)`
27. `abs:("randomized rounding" OR "random rounding" OR "stochastic rounding" OR "randomized quantization") AND abs:(adversarial OR robustness OR backdoor OR defense OR privacy) AND abs:(weights OR "neural network" OR model)`
28. `abs:("quantization noise" OR "rounding error") AND abs:("input-independent" OR "independent of the input" OR "uniformly distributed") AND abs:(weights OR "neural network")`
29. `abs:("high-resolution" OR "high resolution") AND abs:(quantization OR quantizer) AND abs:(bennett OR "uniform error" OR "error distribution") AND (cat:cs.LG OR cat:cs.IT OR cat:eess.SP OR cat:stat.ML)`
30. `abs:(ternary OR "1.58-bit" OR "sign of") AND abs:(delta OR "weight update" OR "fine-tuning") AND abs:(merge OR merging OR "quantized base" OR "integer")`
31. `abs:("weight perturbation" OR "small perturbation" OR "small changes") AND abs:(quantization OR rounding) AND abs:("fine-tuning" OR "fine-tuned") AND abs:(absorbed OR swallowed OR erased OR "below the")` (0 hits)
32. `abs:(dither OR dithering OR dithered) AND abs:(LoRA OR "fine-tuning" OR "fine-tuned" OR backdoor OR unlearning OR "post-training")`
33. `abs:("quantization noise" OR "rounding noise" OR "quantization error") AND abs:(backdoor OR "quantization-conditioned" OR "malicious") AND abs:(random OR randomized OR stochastic)`
34. `abs:(quantization OR quantized OR "low-bit") AND abs:(unlearning OR "safety" OR alignment OR "fine-tuning") AND abs:("weight update" OR "weight change" OR "parameter change" OR "parameter update") AND abs:(magnitude OR "step size" OR "bin" OR "boundary")`
35. `abs:("requantization" OR "re-quantization" OR "re-quantize" OR "requantize" OR "requantized") AND abs:(LLM OR "language model" OR transformer)`
36. `abs:(merge OR merging OR merged) AND abs:(LoRA OR adapter) AND abs:(NVFP4 OR MXFP4 OR "native 4-bit" OR "quantized checkpoint" OR GGUF)`
37. `abs:("stochastic rounding" OR "random rounding") AND abs:(merge OR merging OR "fine-tuning" OR adapter OR unlearning) AND abs:(LLM OR "language model")`
38. `abs:(dither OR dithering OR dithered) AND abs:(quantization OR quantized) AND abs:("deep learning" OR "neural network" OR "neural networks" OR DNN OR "deep neural")` (from 2018)
39. `abs:("defensive quantization" OR "randomized quantization" OR "stochastic quantization" OR "random quantization") AND abs:(adversarial OR backdoor OR attack OR defense)` (from 2018)
40. `abs:(backdoor OR "weight attack" OR "bit-flip") AND abs:(quantization OR quantized) AND abs:(randomization OR randomized OR "random perturbation" OR "random noise") AND abs:(defense OR defence OR mitigation)` (from 2018)
41. `abs:("zeroth-order" OR "zeroth order") AND abs:(unlearning OR "safety") AND abs:(quantization OR robust OR robustness)`
42. `abs:(dither OR dithered OR dithering) AND abs:(LLM OR "large language" OR "language model" OR transformer)` (from 2018)
43. `abs:("universal quantization" OR "lattice quantization" OR "nested lattice") AND abs:(LLM OR "neural network" OR "language model") AND abs:(random OR dither)` (from 2018)

Web searches: `dither "quantization-conditioned" backdoor defense secret seed 2026`; `"stochastic rounding" defense quantization attack LLM`; `fine-tuning delta "quantization step" unbiased "crossing probability" LoRA merge requantization`. None surfaced a dither/stochastic-rounding defense paper.

## Candidates (ranked)

### 1. 2609.21834 — Beyond Shadow Weights: QAT as Quantized-Endpoint Descent (Xu, Li, Ma, Cui; math.OC; 2026-09-18) — **partially_preempts**, priority 1
What it proves (their words, Sec. 3–4 and App. D.3): residual phase θ_i = (x_i − q_i)/s_i; "If the conditional marginal density of x_i varies little across that cell, then the phase ... is approximately uniform on [−½,½). Since the update-selected boundary window has normalized width ρ_i = η|u_i|/s_i, the uniform phase calculation gives crossing probability ρ_i" (Model 2: p_i = η|u_i|/s_i). Corollary 6: on crossing, q_i⁺ − q_i = −s_i sign(u_i); "E[q_i⁺ − q_i | q] = −s_i sign(u_i) p_i" — i.e. the endpoint moves, in expectation, by exactly the shadow update (unbiasedness). Lemma 1 (descent alignment), Proposition 2 (realized endpoint loss bound ≤ −Σ s_i sign(u_i) g_i + ½Σ L_i s_i²), Theorem 3 (expected endpoint descent: −η⟨g,u⟩ + η Σ L_i s_i |u_i|). Empirical: pooled residual-phase histograms at 3/4/8-bit "much closer to a uniform cell-phase density", 2-bit center-concentrated (Fig. 2). Uses this to build QAR (direct code updates). Not stated: Var = Δ|d| − d², SNR, merge/unlearning/attack link, dither. Note their second-order penalty term L_i s_i |u_i| is the same Δ|d| object as the idea's variance.

### 2. 2605.15138 — Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution (MANSU) (2026-05-14) — **partially_preempts** (Design 1 + durability corollary), priority 1
Quotes: "per-parameter updates lie 47–828× below the NF4 quantization bin width"; "even a large-norm gradient induces per-parameter changes of order 10⁻⁶, far below the NF4 quantization bin width of ≈ 8.4×10⁻⁴ ... these changes round to zero"; Proposition 1 (sparsity–permanence tradeoff) bounds ‖Δθ_i‖ ≤ sqrt(2ε_r/(|C| F̄_C)) under a retain constraint; Lemma 1 (quantization survival): any |Δθ_i| ≥ w_i (bin width) changes Q4; Phase 3 "a per-parameter magnitude floor δ_i rescales each update to clear the nearest NF4 bin boundary by construction"; Proposition 2: NF4 amplifies floor-crossing updates (negative PTQ gap). Result: Llama-3.1-8B-Instruct/WMDP-bio PTQ gap −0.040 with MMLU within 0.030 of zero-shot. Also notes "diffuse global updates contain mixed-sign components that cancel below the bin floor under NF4". Gaps left: no unbiased/variance/SNR statement; floor is a rescaling, not an exact {−Δ,0,+Δ} integer-code delta; no cross-domain (merge/attack) unification.

### 3. 2609.04526 — Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs (2026-09-03) — **contradicts** (for on-grid base), priority 1
Sec. 5.3 "Why the naive merge collapses": "The collapse in Table 16 reads like quantization noise swamping a small update. It is not." Claim: base on the grid + sub-step update (|∆_i| below the distance to the rounding midpoint) + same block scale ⇒ "an on-grid value is a fixed point of nearest-rounding, so re-quantizing W + ∆ returns W"; block-level Voronoi condition ⟨∆,D⟩ ≤ ½‖D‖² for all competing representable blocks; measured on 140,000 blocks "the un-adapted block is the global reconstruction optimum on 99.4% of them". Measured "cos(Q(W+∆) − W, ∆) is 0.017 aggregated and 0.141 at the per-matrix median". Naive merge costs "37.4 pp on NVFP4 Llama-8B Banking77 (93.0 → 55.6)", losses up to 39 pp (Table 16); bumped 30.2% of blocks carry the whole merge effect (83.3% vs 0.6% of gain retained). App. B.1: stochastic rounding as a deploy rule "collapses [QAT-LoRA] on two of three (AGNews retains 57%)". Implication for Idea 7: Theorem 7.1's u ~ U[−½,½) premise is false when W is already a quantized checkpoint (u ≡ 0); there the delta is erased deterministically (bias, not noise). The idea must restrict the merge claim to FP bases (and QLoRA adapters trained on a dequantized base are then merged into an on-grid base — the report's experiment 1 as written lands in the wrong regime). Also provides the "66–98% erased" style numbers the report attributes to [2609.18063].

### 4. 2605.15152 — Widening the Gap: Exploiting LLM Quantization via Outlier Injection (ICML 2026) — **contradicts** (Design 2 premise), priority 1
Gaussian-noise defense (Fig. 2, Qwen2.5-7B-Instruct jailbreak): "utility begins to drop clearly from σ = 10⁻³, while the attack remains effective. Further increasing σ reduces ASR only together with a complete utility collapse (σ = 5·10⁻³)"; "prior attacks rely on tightly constrained weight intervals, whereas ours is driven by large inserted outliers that are unaffected by small noise." The outlier-injection attack can be caught by a statistical test on the weight distribution (§C.2). Dither is also "small noise" of size ≤ Δ/2 per element; whether it breaks outlier-induced zero-collapse (a weight |w| ≪ Δ now rounds to ±Δ with probability |w|/Δ instead of deterministically to 0) is untested and is the crux for Design 2.

### 5. 2605.02196 — DurableUn: Quantization-Induced Recovery Attacks in Machine Unlearning (2026-05-04) — **new_hypothesis** (competing theory), priority 2
"INT4 induces recovery of up to 22×"; FA–RA–Q-INT4 trilemma; Proposition 1 (RA-preservation implies quantization fragility): under L-smooth L_f, ‖θ*−θ₀‖ ≤ ρ and L_f(θ*) − L_f(θ₀) ≥ M, then ‖∇L_f(θ*)‖ ≥ M/ρ − Lρ and Q-INT4(θ*) ≥ FA(θ*) + (M/ρ − Lρ)δ − (L/2)δ². Their fix (SAF) is STE-through-INT4 sharpness-aware forgetting; durability certificate Q-INT4 = 0.043 ± 0.002. This is a gradient-norm/sharpness explanation of reversal, orthogonal to the cell-SNR explanation; experiment 2 must separate them (e.g. hold mean|d|/Δ fixed and vary sharpness).

### 6. 2609.27355 — Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction (COLM 2026) — **new_hypothesis**, priority 2
Reports MUSE: models "retain on average only about 13% of the target model's knowledge in full precision, but this jumps to roughly 54% after 4-bit quantization — a nearly 4× increase." Theorem 1 (local, along the line between retain and forget minima): L_r(w*) ∝ (λ/(F_r/F_f − λ))² d², g_f(w*) ∝ (F_r/(F_r/F_f − λ)) d. Method: Fisher-noisy regularization that replaces top-ρ_i rows by their rounded version during forget-loss backprop (rounding-as-noise, QAT-like), perturbing ~0.1% of parameters; "outperforms SURE by about 5% on average". Curvature-ratio theory competes with the magnitude/Δ theory.

### 7. 2601.15538 — QUAIL: Quantization Aware Unlearning (2026-01-21) — **partially_preempts** (durability corollary, heuristic), priority 2
"weight changes must exceed ∆/2, where ∆ is the quantization step size"; weight-change statistics: "mean absolute change of 2.97×10⁻⁵, a maximum change of 1.65×10⁻², and exact bitwise matches for 82.98% of the weights ... ∆ ≈ 0.268"; method = quantization-aware hinge margin in logit space (∆_q = 1.0). Explicit limitation: "we do not provide formal guarantees about the minimum weight change magnitudes or worst-case quantization robustness". Same crossing argument as the idea's corollary, no expectation/variance analysis.

### 8. 2405.12725 — Nearest is Not Dearest: EFRAP (Li et al., CVPR 2024) — **partially_preempts** (Design 2 rationale, deterministic version), priority 2
"the conditioned backdoor carefully learns a set of full-precision model weights, where the nearest rounding errors of this model can push it to the backdoored ones"; defense = "break the direct connection between quantization and nearest rounding" by flipping the rounding direction of high-rounding-error neurons (binary optimization with activation preservation). Dither is the randomized, secret-keyed analogue; EFRAP is for CNN image classifiers and needs clean data.

### 9. 2606.29239 — QuantGuard: Learnable Rounding for Repairing Quantization-Conditioned Backdoors in LLMs (2026-06-28, v3 Aug 2026) — **partially_preempts** (2026 deterministic baseline; adaptive-attacker protocol), priority 2
Differentiable rounding control variables; evaluated on INT8/FP4/NF4. Defense-aware adaptive attacker "tightens the quantization-feasible intervals based on defense-induced weight perturbations": CodeSec drops to 23.3% (StarCoder-1B) / 18.2% (Phi-2) vs 28.4% / 32.6% non-adaptive; QuantGuard restores to 60.4% / 75.9%. A secret-seed dither cannot be "tightened against" without the seed — this is the differentiator the idea should state and test against the adaptive protocol. Also FlipGuard 2606.28962 (ICME 2026, same group).

### 10. 1802.02271 — Universal Deep Neural Network Compression (Choi, El-Khamy, Lee; 2018) — **partially_preempts** (Design 2 mechanics), priority 2
"randomizes DNN weights by uniform random dithering before lattice quantization"; "dithering values are i.i.d. uniform [of support [−∆/2, ∆/2]], which is sufficient to make the quantization error independent of source statistics"; "the encoder and the decoder are assumed to share the information on random dithers or their random seed ... cancels the dithering vectors v̄_i = q_i − u_i". This is exactly seeded subtractive dither of weights with zero storage overhead, used for entropy coding, not security. Design 2's novelty is confined to the threat-model use and the "attack reduces to FP model + independent noise" argument.

### 11. 2505.18724 — LoTA-QAF: Lossless Ternary Adaptation (2025-05-24) — **partially_preempts** (Design 1 exact-merge mechanism), priority 2
Ternary adapters A_T, B_T ∈ {−1,0,1}; Ŵ = sign(∆W)·1{|∆W| > ω} ∈ {−1,0,1}^{Din×Dout} "used to adjust the quantized weights W_int ... can be losslessly merged into W_int after fine-tuning"; overflow prevention at code boundaries; claims 1.7–2.0× faster than LoRA after merge. This is a cell-aligned ±1-code delta trained with a ternary low-rank parameterization — Design 1's integer-exact merge already exists as PEFT. Not framed as durable unlearning / SNR = ∞. See also 2608.24469 (ECCV 2026, ternary multiplicative adaptation).

### 12. 2405.18137 — Exploiting LLM Quantization (Egashira et al., NeurIPS 2024) — **supports**, priority 2
Table 6 (Phi-2, LLM.int8(), vulnerable-code scenario): "At σ = 10⁻³ adding noise proves to be an effective defense against the attack, removing the security contrast while maintaining utility"; credits Ma et al. [14] for the weight-perturbation defense on small models. Supports the premise that an unpredictable perturbation defeats box-constrained attacks; dither is the calibrated (MSE = Δ²/12, input-independent) version. Also 2505.23786 (GGUF, ICML 2025) for the 9-format extension.

### 13. 2609.18063 — The Other Half of the Memory Wall: Serving 35B MoEs from SSD (Edge0) (2026-09-16) — **supports**, priority 4
Citation fix for the report's [U] entry: the paper is an MoE SSD-serving system, but §3.3 states "LoRA deltas (RMS 10⁻³) sit below the 4-bit group step, so requantization erases most of the weight-level delta (34% of the effect survives on an attention projection, 2% on a dense projection), and at the logits level 18% survives." Adapter is served unmerged. Base is int4 (on-grid), so this is the Scale-QLoRA regime (deterministic erasure), not the uniform-offset regime.

### 14. 2510.00761 — Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning (ICLR 2026) — **new_hypothesis**, priority 3
ZO / sign-based / compressed-gradient optimizers improve robustness to 4-bit (GPTQ) quantization and relearning; explanation offered is "harder-to-disturb basins". Confound for the SNR predictor: sign-type updates have equalized per-coordinate magnitudes (larger mean|d| at equal ‖D‖_F), so the magnitude story and the basin story predict the same direction; experiment 2 should include this optimizer axis.

### 15. 2608.27512 — Quantization-Triggered Backdoors: Cross-Quantizer Transferability (2026-08-27) — **tangential**, priority 3
Defines QBEC (per-weight interval product, Proposition 1: distinct FP models with identical quantized behavior exist); cross-quantizer transfer of INT8-optimized attacks is mixed (NLLB INT8→NF4 T = 0.552; others "collapsed under all 4-bit schemes (T_NF4 = 0.063)"); single-seed. Relevant to Design 2 only as evidence that attacks sometimes survive a *different* deterministic map; dither is a random map per element, which is a stronger change.

## Also seen (not in top 15)
- 2602.13151 Quantization-Robust LLM Unlearning via LoRA (2026-02): same Δ/2 bin-crossing argument ("4-bit step ≈ 16× larger than sint8"); LoRA α-scaling to force crossing; MUSE Llama-2-7B 4-bit utility +7.93 (NPO+GDR BOOKS 50.17→58.10). supports.
- 2411.19530 Quantized Delta Weight Is Safety Keeper (2024-11): quantizing the fine-tune delta (BitDelta-style) "mitigates alignment-breaking risks by up to 66.17%, harmful backdoor vulnerabilities by 64.46%, and targeted output manipulation risks by up to 90.53%" with <10% utility loss — consistent with "low-SNR deltas are fragile". supports.
- 2603.22324 DAQ (Tencent, 2026-03): PTQ objective that preserves sign/cosine of ΔW = post-trained − base because reconstruction-MSE objectives "disproportionately corrupt the small-magnitude parameter deltas"; FP8 pilot only. supports (premise).
- 2606.20254 QVec (2026-06): δ = Q(W) − W in attacked models is "a structured behavioral shift ... rather than random quantization noise"; Gaussian-noise baseline "unstable and attack-dependent". supports framing (attack = structured δ; dither makes δ ⊥ W).
- 2606.28962 FlipGuard (ICME 2026), 2609.14060 AGENTQ (attack on agents; says weight auditing fails), 2601.02680 ACL attack (box-projected PGD; ASR 97.69% jailbreak).
- 2605.13810 Provable Quantization with Randomized Hadamard Transform (2026-05): proves unbiasedness of dithered scalar quantization after HD; cites Schuchman 1964 / Zamir–Feder 1992 — usable as the modern citation for the classical dither theorem (the report currently marks Schuchman/Gray–Stockham [U]).
- 2610.08231 OSFP4 (Ordentlich group, 2026-10): analyzes a multiplicatively dithered FP4 quantizer to smooth the MSE for scale optimization — evidence that dither analysis is entering NVFP4 practice, not a defense.
- 2609.39185 Low-Discrepancy Dither for Quantized Recurrent State Caches (2026-09): dither for state caches; tangential.
- 2605.04209 Undetectable Backdoors in Model Parameters (2026-05): uses Gaussian dither on the *attack* side as a reference distribution for an undetectability proof; opposite direction.
- 2306.11913 Randomized Quantization is All You Need for DP in FL (2023): randomized quantization ⇒ noise independent of data (privacy); conceptual cousin of "dither ⇒ attacker-independent noise".
- 2503.01901 Post-quantization Integral (2025): integrates loss along the rounding path per weight; a tool for the non-high-resolution regime the idea flags as open.
- 2608.11045 ReRound (midpoint ambiguity), 2604.04013 RUQuant: tangential rounding-rule papers.

## What remains open after this sweep (for the hypothesis/judge stages)
1. No paper states Var(D̃) = Δ|d| − d² or the SNR scalar ‖D‖_F²/(Δ‖D‖_1), nor tests one scalar across merge / unlearning / attack. That unification is still unclaimed — but the P(cross) = |d|/Δ, E[D̃] = D part is in 2609.21834.
2. No paper evaluates dithered / stochastic rounding as a defense against quantization-conditioned attacks (arXiv + web, as of 2026-10-10). The nearest evidence (Gaussian pre-noise) is positive for box-constrained attacks (2405.18137) and negative for outlier injection (2605.15152).
3. The idea's merge claim is wrong in the on-grid-base regime (2609.04526, 2609.18063): there the loss is deterministic erasure (cos ≈ 0.02), not unbiased noise. The theorem needs an explicit "FP base with uniform cell phase" scope, and experiment 1 must use FP-base LoRA, not QLoRA-on-int4.
4. Design 1 overlaps MANSU's magnitude floor (unlearning) and LoTA-QAF's ternary code adapters (PEFT); the residual novelty is the exact {−Δ,0,+Δ} formulation as a *proximal/projected step with SNR = ∞* and its use for durable unlearning plus BitDelta-class compression in one object.
