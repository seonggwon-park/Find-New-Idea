# Idea 7 — contrarian sweep (Angle D)

Idea under test: "quantization-cell theorem" — re-quantized post-training deltas D̃ = Q(W+D) − Q(W) are unbiased with Var = Δ|d| − d², SNR ≈ ‖D‖_F²/(Δ‖D‖_1); one scalar explains LoRA-merge collapse ("2–34% remains"), unlearning reversal at 4-bit, and quantization-conditioned attacks; two designs: (1) cell-aligned sparse ternary deltas D ∈ {−Δ,0,+Δ}, (2) secret-seed subtractive-dither quantization as a provable defense.

Date of sweep: 2026-10-10. All paper text quoted below is untrusted data from arXiv full text.

## Headline verdicts

1. **The "2026 systems paper" the report could not identify is 2609.04526 (Scale-QLoRA), not 2609.18063.** 2609.18063 is "The Other Half of the Memory Wall: Serving 35B MoEs from SSD" — unrelated. Scale-QLoRA states the sub-step-delta erasure formally (Voronoi/fixed-point argument), measures it (naive merge "deletes the adaptation, by up to 39 pp"; "RTN retains 15% of the fine-tuning gain ... while GPTQ recovers 91.46 and keeps 94%"). The GPTQ number **contradicts** the idea's claim that SNR = mean|d|/Δ alone predicts survival: an error-feedback quantizer rescues ~94% of a delta that RTN deletes. The "2–34%" figure is not in this paper; QA-LoRA's own Table 1 gives 37–68% retention for QLoRA-merge→GPTQ-4bit (see below).
2. **Unlearning reversal already has an interval theory, and 2026 has at least five quantization-robust-unlearning papers, two with formal statements that overlap the idea's durability corollary and design 1:** MANSU 2605.15138 (Lemma 1 "quantization survival": |Δθ_i| ≥ bin width ⇒ code changes; Proposition 1 sparsity–permanence tradeoff ‖Δθ_i‖ ≤ sqrt(2ε_r/(|C| F̄_C)); Proposition 2 NF4 amplifies floor-crossing updates; "per-parameter updates lie 47–828× below the NF4 quantization bin width") and DurableUn 2605.02196 (Proposition 1 "RA-Preservation Implies Quantization Fragility"; INT8 benign, INT4 recovery up to 22×; sharp phase transition). Zhang et al. 2410.16454 itself has a Section 5 "theoretical analysis" (weight change must exceed Δ; int4 needs 12.5 vs int8 0.78 in their example) — the report's "이론은 없다" is false for the interval-level statement, though no paper gives the unbiased/variance/SNR form.
3. **Design 1 (grid-aligned ternary deltas merged into integer codes) is essentially LoTA-QAF 2505.18724:** ternary matrix Ŵ ∈ {−1,0,1} aligned to the quantization grid, lossless merge W'_int = W_int + Ŵ, z' = z + sµ, trained with t-SignSGD (sign-based projected step). Also MANSU's "magnitude floor" (rescale every cumulative update below the NF4 bin to the floor) is design 1's unlearning variant. ComPEFT 2311.13171 (sparse ternary task vectors) and IntLoRA 2410.21759 (integer-domain adapters merged without PTQ) are adjacent.
4. **Design 2 (secret dither) has no direct prior art found** (no arXiv hit for dither/stochastic-rounding as a QCB defense; WebSearch likewise). But the contrarian evidence against its premises is substantial:
   - 2605.15152 (ICML 2026): the outlier-injection attack is "driven by large inserted outliers that are unaffected by small noise", not by a tight preimage box; Gaussian noise defense fails ("utility begins to drop clearly from σ = 10⁻³, while the attack remains effective"; ASR falls only at σ = 5·10⁻³ with "complete utility collapse"); attacked-model utility stays >90% relative (Table 2), so "outlier injection is detected by accuracy drop" is contradicted; the authors already propose a KS-statistic detector (§C.2) and show rescaling defenses fail at c = 4096 (Table 3).
   - QVec 2606.20254: "random noise does not consistently reduce the ASR; instead, it often degrades clean performance while leaving the backdoor behavior partially intact"; δ = Q(W) − W is a structured task vector, not random rounding noise.
   - Qu-ANTI-zation 2110.13541 (NeurIPS 2021): random noise with the same magnitude as the quantization perturbation leaves backdoor ASR ∼99% (Table 5, CNNs); fine-tuning removes it.
   - Rounding-map defenses already exist in deterministic form: EFRAP 2405.12725 (CVPR 2024, "activation of existing QCBs primarily stems from the nearest rounding operation"; learned non-nearest rounding), FlipGuard 2606.28962 (ICME 2026), QuantGuard 2606.29239 (learnable rounding; includes a defense-aware adaptive attacker that tightens feasible intervals). None evaluates stochastic/dithered rounding — that gap is real.
   - AGENTQ 2609.14060 (EMNLP 2026): the per-weight box attack transfers to RTN and AWQ (ASR_q = 0.87) but GPTQ's Hessian error feedback breaks the box ("a weight's code depends on its neighbors"). The idea's independence assumption (u_ij independent, per-element rounding) is quantizer-specific.
   - Utility cost of random rounding: AdaRound 2004.10568 Table 1 (ResNet18 first layer, 4-bit): RTN 52.29% vs stochastic rounding 52.06 ± 5.52% (high variance; best sample 63.06). Non-subtractive SR has MSE Δ²/6; subtractive dither Δ²/12 only if the dither is regenerated at dequantization.
5. **Corollary "safety SFT deltas are erased at 4-bit" is contradicted by 2606.29581** (8 models, 144 configs, ~2M responses): "standard quantization is approximately safety-neutral: for 7 of 8 models, AWQ INT4 keeps attack success within about 1.6 percentage points of FP16 or lowers it". Decoding Compressed Trust 2403.15447 agrees (4-bit retains trustworthiness). Counter-evidence exists too (2404.04392, 2502.15799, 2511.07842 report degradation), so this is mixed, not settled — but it falsifies the blanket claim.
6. **Theorem 7.1 is the classical stochastic-rounding error model.** With u ~ U[−½,½), E[D̃] = d and Var = Δ²p(1−p), p = |d|/Δ — exactly the SR unbiasedness/variance identity (Gupta et al. 2015, 1502.02551; El Arar et al. 2207.10321 "Stochastic rounding variance and probabilistic bounds"). Novelty lies only in applying it to (base-offset, delta) pairs and in the SNR aggregate.
7. **"LoRA deltas have mean|d| ≪ Δ" is not uniformly true:** 2602.13151 (IJCNN 2026) shows LoRA *concentrates* unlearning updates so they survive 4-bit better than full-parameter FT ("standard full-parameter finetuning often induces parameter changes that are too small to survive 4-bit quantization"), and 2510.00761 (ICLR 2026) shows sign-based (1-bit Adam) and ZO optimizers give more quantization-robust unlearning than Adam/Sophia — magnitude/shape of the update, not rank, is what matters.

## Candidate table

| id | title | threat | prio | key evidence |
|---|---|---|---|---|
| 2605.15138 | Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution (MANSU) | partially_preempts | 1 | updates 47–828× below NF4 bin width; Lemma 1 quantization survival by magnitude floor; Prop 1 sparsity–permanence; Prop 2 NF4 amplifies; PTQ gap −0.040 on Llama-3.1-8B/WMDP-bio |
| 2609.04526 | Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs | partially_preempts (and contradicts SNR-only) | 1 | naive merge deletes adaptation up to 39 pp; RTN keeps 15%, GPTQ keeps 94%; formal fixed-point/Voronoi claim for sub-step deltas; base is reconstruction optimum on 99.4% of 140k blocks |
| 2605.15152 | Widening the Gap: Exploiting LLM Quantization via Outlier Injection (ICML 2026) | contradicts | 1 | attack not box-based; Gaussian noise fails; utility >90% preserved; KS detection; rescaling fails at c=4096 |
| 2505.18724 | LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning | partially_preempts | 1 | grid-aligned ternary Ŵ∈{−1,0,1}, lossless merge into integer codes + zero offset, t-SignSGD; beats QA-LoRA on MMLU by up to 5.14% |
| 2605.02196 | DurableUn: Quantization-Induced Recovery Attacks in Machine Unlearning | partially_preempts | 2 | Prop 1 RA-preservation ⇒ quantization fragility; INT8 benign, INT4 up to 22× recovery; sharp phase transition; STE-based SAF |
| 2609.27355 | Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes (COLM 2026) | contradicts | 2 | Theorem 1: curvature ratio F_f/F_r governs robustness; perturbing 0.1% of parameters suffices — sharpness, not |d|/Δ, is the predictor |
| 2601.15538 | QUAIL: Quantization Aware Unlearning | partially_preempts | 2 | bucket overlap >99.9% at 4-bit; "weight changes must exceed Δ/2"; logit-space hinge margin fix |
| 2602.13151 | Quantization-Robust LLM Unlearning via Low-Rank Adaptation (IJCNN 2026) | contradicts | 2 | LoRA concentrates magnitude so unlearning survives 4-bit; full-FT changes too small |
| 2410.16454 | Catastrophic Failure of LLM Unlearning via Quantization (ICLR 2025) | supports | 2 | 21%→83%; Section 5 interval theory already exists (contradicts "no theory") |
| 2606.20254 | Quantization as a Malicious Task: Removing QCBs via Task Arithmetic (QVec) | contradicts | 2 | random noise unreliable; δ structured; Gaussian noise "unstable and attack-dependent" |
| 2606.29239 | QuantGuard: Learnable Rounding for Repairing QCBs in LLMs | partially_preempts | 2 | learnable (deterministic) rounding defense; defense-aware adaptive attacker tightening intervals; no dither baseline |
| 2606.29581 | Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment | contradicts | 2 | AWQ INT4 within 1.6 pp ASR of FP16 for 7/8 models — safety SFT survives 4-bit |
| 2309.14717 | QA-LoRA (ICLR 2024) Table 1 | contradicts | 2 | LLaMA-7B Alpaca: base 34.6 → QLoRA 38.4 → QLoRA w/ GPTQ-4bit 36.0 (37% retained 5-shot; 68% 0-shot) — not 2–34% |
| 2405.12725 | Nearest is Not Dearest (EFRAP, CVPR 2024) | partially_preempts | 3 | nearest rounding is the root cause; flipped-rounding defense; adaptive attacks incl. random flipping |
| 2609.14060 | AGENTQ (EMNLP 2026) | supports (scope limit) | 3 | box attack transfers to RTN/AWQ (ASR_q 0.87), GPTQ error feedback breaks per-weight box |

Secondary (not in structured list): 2110.13541 Qu-ANTI-zation (noise defense fails, ASR ~99%); 2505.23786 Mind the Gap (Gaussian noise works for k-quants but σ must be calibrated per model: 1e-3 Qwen2.5-3b, 1e-4 Llama3.1-8b); 2311.13171 ComPEFT (sparse ternary task vectors); 2410.21759 IntLoRA (ICML 2025, integer adapters merge without PTQ); 2510.00761 (ICLR 2026, sign/ZO optimizers ⇒ quantization-robust unlearning); 1502.02551 / 2207.10321 (SR unbiasedness + variance identity = Theorem 7.1); 2004.10568 AdaRound Table 1 (SR 52.06±5.52 vs RTN 52.29); 2609.39185 (deterministic Weyl dither beats SR for recurrent state caches — dither in LLM context, tangential); 2610.01889 (SR vs RN in transformer inference depends on site); 2606.28962 FlipGuard; 2601.02680 (adversarial contrastive QCB attack).

## Queries run (arXiv API unless noted)
- abs:quantization AND abs:unlearning
- abs:quantization AND abs:attack AND abs:(backdoor OR malicious) AND abs:LLM
- abs:LoRA AND abs:merge AND abs:quantized
- abs:dither AND abs:quantization AND abs:("neural network" OR LLM OR weights)
- abs:"stochastic rounding" AND abs:(defense OR security OR attack OR backdoor) AND abs:quantization
- abs:"quantization-aware" AND abs:(unlearning OR "safety alignment") AND abs:robust
- abs:delta AND abs:compression AND abs:quantized AND abs:"fine-tuned"
- abs:"preimage" AND abs:quantization
- abs:"task vector" AND abs:quantization
- abs:(randomized OR random OR stochastic) AND abs:quantization AND abs:backdoor
- abs:quantization AND abs:(safety OR jailbreak OR alignment) AND abs:LLM AND abs:(degrad OR compromis OR preserv)
- abs:(dithered OR dithering OR "subtractive dither") AND abs:quantization AND abs:(LLM OR "language model" OR "deep learning" OR "deep neural")
- abs:"stochastic rounding" AND abs:("post-training quantization" OR "weight quantization") AND abs:(LLM OR "language model" OR network)
- abs:(LoRA OR adapter) AND abs:(merging OR merged OR merge) AND abs:(requantiz OR "re-quantiz" OR "quantized weights" OR "quantized base") AND abs:(loss OR degrad OR lossless)
- abs:"model editing" AND abs:(quantization OR quantized)
- abs:(fingerprint OR watermark) AND abs:LLM AND abs:quantization AND abs:(robust OR surviv)
- abs:(quantization-conditioned OR "quantization conditioned" OR "quantization attack" OR "quantization backdoor")
- abs:(quantized OR quantization) AND abs:(jailbreak OR "harmful" OR "safety") AND abs:("4-bit" OR "4 bit" OR INT4 OR GPTQ OR AWQ) AND abs:LLM
- ti:"stochastic rounding"
- abs:("randomized rounding" OR "random rounding" OR "stochastic rounding") AND abs:(LLM OR "language model") AND abs:(quantization OR quantized)
- abs:(secret OR seed OR key) AND abs:quantization AND abs:(backdoor OR "attack") AND abs:(rounding OR dither OR random)
- abs:("weight perturbation" OR "parameter perturbation" OR "weight noise") AND abs:(backdoor OR "quantization") AND abs:defense AND abs:LLM
- abs:quantization AND abs:backdoor AND abs:(remov OR mitigat OR eras OR neutraliz) AND abs:(fine-tuned OR fine-tuning) AND abs:(LLM OR "language model")
- abs:(ternary OR "sign-based" OR signSGD OR "sparse update") AND abs:unlearning AND abs:(quantization OR quantized)
- abs:(quantization OR quantized) AND abs:("fine-tuning" OR "fine-tuned") AND abs:(revert OR undo OR "wiped" OR "erased" OR "vanish") AND abs:(weight OR delta)
- abs:unlearning AND abs:("4-bit" OR INT4 OR NF4 OR "low-bit") AND abs:(recover OR restor OR revers OR resurfac)
- abs:("randomized smoothing" OR "random noise") AND abs:(weights OR parameters) AND abs:backdoor AND abs:(certif OR provabl)
- WebSearch: dithered quantization OR "stochastic rounding" defense "quantization-conditioned" backdoor LLM
- WebSearch: LoRA adapter merge re-quantize 4-bit fine-tuning gain lost "quantization step" delta below group step erased
