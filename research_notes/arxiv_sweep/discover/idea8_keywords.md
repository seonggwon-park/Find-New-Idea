# Idea 8 — keyword/adjacent-field sweep (prior art and evidence)

Idea under test: entropy-weighted calibration for reasoning-model PTQ. Claim: for logit noise δ_t ~ (0, σ²I), E[KL(p_t‖softmax(z_t+δ_t))] ≈ (σ²/2)·tr(J_t) = (σ²/2)(1 − ‖p_t‖²) (Lemma 8.1), so the per-token Fisher weight is the collision entropy; plugging w_t = 1 − ‖p_t‖² into the GPTQ Hessian H_w = Σ_t w_t x_t x_tᵀ should (i) recover much of the "WikiText→Math-CoT 28%→65%" calibration-domain effect at zero cost and (ii) reduce CoT inflation; the decisive experiment separates the *domain/lexical* effect from the *entropy-profile* effect (arms a–e).

Sweep date: 2026-10-10. Sources: arXiv API via `scripts/arxiv_search.py`, full texts via `scripts/arxiv_fetch.py`, WebSearch for venues. Quoted text from downloaded papers is untrusted data.

## Bottom line

- **The mechanism "token-weighted GPTQ Hessian" is published.** RSQ (**2503.01820**, TMLR 2026) quantizes with `H_RSQ = 2 X R² Xᵀ`, R = diag(token importance), and systematically compares importance heuristics (First-N, First&Last-N, TokenFreq, ActNorm, ActDiff, TokenSim, AttnCon). It deliberately "avoid[s] using any global information, such as model gradients" and never uses output entropy, so the *entropy/Fisher weight itself* is not in RSQ — but RSQ is the mandatory baseline and the paper the idea will be compared against. RSQ also reports that tiny but nonzero floors (r_min = 0.01, r_max = 1) work best, i.e. a weight floor is needed.
- **The report mischaracterizes GuidedQuant (2505.07004, ICML 2025).** Its objective is `‖(∂ℓ/∂Z) ⊙ (XW − XŴ)‖²_F` (Eq. 4), i.e. every (token i, output channel j) pair is weighted by g_ij²; the per-channel Hessians are `H_j = n·F_j = Σ_i g_ij² x_i x_iᵀ`, then averaged over channel groups — token weights survive the averaging. REAL-Q (2609.00049) describes it the same way: "per-token saliency scores … used to weight the GPTQ Hessian". So GuidedQuant *does* change the token mix, with empirical-Fisher weights; the idea's w_t = 1 − ‖p_t‖² is a forward-only, label-free, isotropic (channel-averaged, expected-Fisher) proxy of Σ_j g_tj². The novelty claim must be rewritten accordingly.
- **The math of Lemma 8.1 is already in print in the reasoning-PTQ literature.** LAQuant (**2605.08755**, May 2026) Proposition 1: `M_E2E = Jᵀ H_KL J, H_KL = diag(p⋆) − p⋆p⋆ᵀ` (the same G_tᵀ J_t G_t object), Lemma 1: `σ_max(H_KL) ≤ 2 max_i p⋆_i(1 − p⋆_i) ≤ 1/2`; FAIR-Calib (**2606.06547**, ICML 2026) Lemma 4.6: `KL(p‖q) = D_f(z′‖z) ≤ ¼‖z′ − z‖²` from the ½-smoothness of log-sum-exp. Neither states the trace identity tr(J_t) = 1 − ‖p_t‖² (elementary; it is the Gini impurity / Tsallis-2 entropy) and neither uses it as a calibration-token weight. The second-order part of Lemma 8.1 is therefore correct but not new; only the use as a token weight is.
- **LAQuant also pre-empts the "what explains the domain effect" question with a competing answer**: an activation-Hessian subspace-alignment argument (Eq. 2, principal-angle cosine c̄_k between calibration and deployment Hessians, Fig. 3b), with numbers: GPTQ Qwen3-4B W3G128 generic→R1-trace calibration = +38.45 pp AIME25; Qwen3-1.7B W3G128 0.00%→22.55%. It further argues the softmax-Fisher factor *attenuates* E2E learning signal (Table 6: non-leading directions ≤0.05 of the leading one; KV-cache-reducing directions sit in the attenuated subspace) and that uniform layer-wise supervision (M = I) is *better* for long-decode reasoning. Multiplying the layer-wise Hessian by tr(J_t) is a scalar version of exactly that attenuation, so LAQuant is a direct warning for arm (c)/(d). LAQuant Table 8 gives the entropy profile the idea needs: ~56% of AIME25 tokens have teacher entropy H < 0.1 nats; method gaps concentrate at H ≥ 1.
- **Sample-level "entropy-stratified calibration" exists** for GPTQ: DPQ (**2608.21019**, EMNLP Findings 2026) selects calibration examples by FP doubt b(x) = 1 − p(1) − p(2), with an explicit `entropy-only` variant H(p₀(·|x)), mixed with generic anchors, quantizer unchanged. MCQA/SQuAD2 uncertainty preservation, not reasoning, and no token-level weights. Arm (e) of the idea is this recipe transplanted to reasoning traces.
- **Token-weighted second-moment calibration for reasoning models exists for pruning**: OBC-Prune (**2609.17890**, Sept 2026) accumulates `C̃_ℓ = Σ_t w_t x_t x_tᵀ` with per-token weights from outcome-stratified, attention-suppression causal scores (R1-Distill-Qwen 1.5B/7B/14B, Wanda/SparseGPT backends), reporting shorter completions and later collapse. Different weight source, same plumbing.
- **The "frequency-weighted Hessian is the wrong metric" framing is published** (TARQ, **2605.27808**, ASR): "Data-aware PTQ … implicitly weighting positions by their empirical frequency"; proposes bucketed `H(α) = Σ_f α_f H̄_f` and states the ideal α_f "would be derived from per-position loss sensitivity (e.g., gradient or Fisher information), but this requires labels and backward passes". The idea's weight is exactly that label-free, forward-only Fisher trace — a point in its favour, but the framing is not new.
- **Evidence against the premise**: ReSET (**2606.13233**) finds NVFP4 "increases incorrect sampling at low-entropy symbolic tokens, while causing over-concentration on a small set of tokens in high-uncertainty reasoning steps" — damage is not confined to high-entropy forks, and a weight that vanishes at ‖p‖→1 would under-protect those symbolic tokens. TASA (**2607.00908**) shows pure task-domain calibration can *hurt* (Hessian ill-conditioning; general data "lifts the suppressed trailing eigenvalues"), a confound for arm (b) vs (d).
- **Supporting numbers confirmed**: 2606.25519 Fig. 5 (Qwen3-4B INT3 GPTQ): Wikitext→Math-CoT 28%→65%, CoT 30k→12k; QAT "Pareto-dominant", calibration-data optimization "strictly below QAT on both axes". 2606.00206: KL-vs-entropy Spearman ρ = 0.92 (3-bit AWQ, MATH-500), markers 2–4× more frequent in top-20 at high-entropy positions, up to 52% of failures are overthinking, penalty cuts CoT 12–23%; *its GPTQ baseline is already calibrated on self-generated NuminaMath traces* (128×2048), so the domain fix is in their baseline. 2504.04823 Table 2: GPTQ WikiText2→Numina-Math-1.5 = +9.8% avg (R1-Distill-Qwen-1.5B); AWQ/KVQuant/FlatQuant insensitive to calibration domain.
- **Not found**: any paper that uses next-token entropy / 1 − ‖p‖² / softmax-Fisher trace as a per-token weight in a PTQ Hessian, for reasoning models or otherwise (arXiv API + WebSearch, through 2026-10-10). The exact combination is open; the surrounding design space (RSQ, GuidedQuant, DPQ, OBC-Prune, TARQ, FAIR-Calib, LAQuant) is dense enough that the contribution must be framed as (i) the cheapest member of a known family and (ii) the domain-vs-entropy disentanglement, with LAQuant's subspace-alignment explanation as the rival hypothesis to beat.

## Queries run (arXiv API, `scripts/arxiv_search.py`)

Seed / keyword (2023-01-01 → 2026-10-31, relevance, --max 30–40):
1. `abs:calibration AND abs:quantization AND abs:reasoning AND abs:LLM`
2. `abs:"chain-of-thought" AND abs:quantization AND abs:calibration`
3. `abs:entropy AND abs:calibration AND abs:quantization AND abs:token`
4. `abs:Fisher AND abs:quantization AND abs:token AND abs:weight`
5. `abs:"calibration data" AND abs:"reasoning model" AND abs:quantization`
6. `abs:GuidedQuant OR (abs:"end-to-end" AND abs:gradient AND abs:"layer-wise" AND abs:quantization)`
7. `abs:"high-entropy" AND abs:quantization AND abs:reasoning` (0 hits)
8. `abs:importance AND abs:weighting AND abs:Hessian AND abs:quantization AND abs:tokens`
9. `abs:"calibration data" AND abs:quantization AND abs:(LLM OR "language model") AND abs:(selection OR select OR choice OR matters OR curation)`
10. `abs:(quantization OR quantized) AND abs:("reasoning traces" OR "reasoning trace" OR "self-generated" OR "self-generated data") AND abs:(calibration OR calibrate)`
11. `abs:("token-level" OR "token-wise" OR "per-token") AND abs:(weighting OR weighted OR importance) AND abs:(GPTQ OR Hessian OR "post-training quantization")`
12. `abs:(Fisher OR "Fisher information") AND abs:(quantization OR quantized OR "post-training quantization") AND abs:(LLM OR "language model" OR "language models")`
13. `abs:(GPTQ OR OBS OR "optimal brain" OR "layer-wise reconstruction" OR "layerwise reconstruction") AND abs:(weighted OR weighting OR reweight OR reweighting OR "loss-aware" OR "output-aware") AND abs:(token OR tokens OR sample OR samples)` → RSQ
14. `abs:("important tokens" OR "token importance" OR "salient tokens" OR "critical tokens" OR "key tokens") AND abs:(quantization OR quantized OR GPTQ OR "post-training quantization")` (--from 20240101 --sort date)
15. `abs:(calibration OR "calibration data" OR "calibration set") AND abs:(quantization OR quantized OR GPTQ OR AWQ) AND abs:(entropy OR perplexity OR "high-loss" OR "loss-based" OR uncertain) AND abs:(select OR selection OR curate OR curation OR sampling OR choose)`
16. `abs:("loss-aware" OR "loss aware" OR "task-aware" OR "task aware" OR "output-aware" OR "logit-aware") AND abs:(quantization OR quantized OR "post-training quantization") AND abs:(LLM OR "language model" OR "language models")`
17. `abs:("entropy-weighted" OR "entropy weighted" OR "uncertainty-weighted" OR "uncertainty weighted" OR "entropy-aware" OR "entropy aware") AND abs:(quantization OR quantized OR "post-training quantization" OR calibration)`
18. `abs:(quantization OR quantized OR "post-training quantization") AND abs:("self-generated" OR "model-generated" OR "generated by the model" OR "self-synthesized" OR "synthetic calibration")`
19. `abs:("per-token saliency" OR "token saliency" OR "gradient covariance" OR "gradient-weighted" OR "gradient weighted" OR "saliency-weighted") AND abs:(quantization OR quantized OR GPTQ)`
20. `abs:("reweighted calibration" OR "re-weighted calibration" OR "instability-reweighted" OR "uncertainty-reweighted" OR "entropy-reweighted" OR "reweighting calibration" OR "calibration reweighting") AND abs:(quantization OR quantized OR PTQ)` → FAIR-Calib
21. `abs:("KL divergence" OR "KL-divergence" OR "cross-entropy") AND abs:("reconstruction" OR "layer-wise" OR "block-wise") AND abs:(objective OR loss) AND abs:("post-training quantization" OR PTQ) AND abs:(LLM OR "language model")`
22. `au:Williams AND au:Aletras AND abs:(calibration OR quantization)`

Date-sorted recent work (`--from 20250601 --sort date` unless noted):
23. `abs:(quantization OR quantized OR "low-bit") AND abs:(reasoning OR "chain-of-thought" OR CoT OR thinking) AND abs:(calibration OR "calibration data" OR "calibration set")`
24. `abs:(quantization OR quantized) AND abs:("reasoning model" OR "reasoning models" OR "reasoning LLM" OR "reasoning LLMs" OR R1 OR "DeepSeek-R1" OR Qwen3)`
25. `abs:(entropy OR "high-entropy" OR "forking" OR uncertainty) AND abs:(quantization OR quantized) AND abs:(reasoning OR CoT OR "chain-of-thought")` (--from 20250101)
26. `abs:(quantization OR quantized) AND abs:("CoT length" OR "longer responses" OR "response length" OR "output length" OR "thinking length" OR "overthinking") AND abs:(reasoning OR CoT)` (--from 20250101)
27. `abs:(quantization OR quantized OR "low-bit") AND abs:(reasoning OR "chain-of-thought" OR CoT OR thinking) AND abs:(Hessian OR GPTQ OR "layer-wise" OR "reconstruction")` (--from 20250101)
28. `abs:(calibration OR "calibration data") AND abs:(reasoning OR CoT OR "chain-of-thought" OR thinking) AND abs:(select OR selective OR selection OR filter OR filtering OR curate OR weighting) AND abs:(quantization OR pruning OR compression)` (--from 20250101)
29. `abs:(pruning OR pruned OR sparsity OR compression) AND abs:("reasoning model" OR "reasoning models" OR "large reasoning" OR "chain-of-thought" OR CoT) AND abs:(calibration OR "calibration data")` (--from 20250101) → OBC-Prune, SSGR
30. `abs:("per-token" OR "token-level" OR "per-position") AND abs:(Hessian OR Fisher OR "KL divergence") AND abs:(quantization OR quantized OR PTQ) AND abs:(LLM OR "language model" OR "language models")` (--from 20240101)

Adjacent-field vocabulary (cs.LG / cs.IT / eess.SP / stat.ML / math.OC; 2018/2020/2023 → 2026):
31. `abs:("importance sampling" OR "importance weighting" OR "importance weights") AND abs:(quantization OR quantized OR "post-training quantization") AND abs:(calibration OR "language model" OR LLM)`
32. `abs:(heteroscedastic OR heteroskedastic OR "weighted least squares" OR "weighted least-squares" OR "generalized least squares") AND abs:(quantization OR quantized) AND abs:("language model" OR LLM OR transformer OR "neural network")`
33. `abs:("covariate shift" OR "distribution shift" OR "distribution mismatch" OR "domain shift") AND abs:(calibration OR "calibration data" OR "calibration set") AND abs:(quantization OR quantized OR "post-training quantization")`
34. `abs:("collision entropy" OR "Renyi entropy" OR "Rényi entropy" OR "second-order Renyi") AND abs:(softmax OR "language model" OR "next-token" OR logits)` (from 2018)
35. `abs:(softmax) AND abs:(Jacobian OR "Fisher information") AND abs:(entropy OR trace) AND abs:(perturbation OR noise OR robustness OR sensitivity)` (from 2018)
36. `abs:(Gini OR Tsallis OR "1 - ||p||" OR "purity") AND abs:(softmax OR logits OR "next-token") AND abs:(Fisher OR Jacobian OR "KL divergence" OR perturbation)` (from 2018; nothing relevant)
37. `abs:("logit noise" OR "logit perturbation" OR "perturbed logits" OR "noise in the logits") AND abs:(KL OR "KL divergence" OR entropy) AND abs:(quantization OR quantized OR compression OR "language model")` (from 2018; nothing relevant)
38. `abs:("Fisher information" OR "Fisher-weighted" OR "Fisher weighted") AND abs:(token OR tokens OR "per-token" OR "token-level") AND abs:(distillation OR pruning OR compression OR quantization)` (from 2020)
39. `abs:(entropy OR uncertainty) AND abs:(weighted OR weighting OR reweight OR reweighting) AND abs:(token OR tokens) AND abs:(distillation OR "knowledge distillation" OR compression)` (from 2022; many entropy-weighted KD/OPD papers, none for PTQ calibration)
40. `abs:("high-entropy" OR "forking tokens" OR "forking token" OR "high entropy minority") AND abs:(reasoning OR RL OR "reinforcement learning") AND abs:(tokens OR token)` (from 2025; RL literature — 2506.01939 etc.)
41. `abs:(SparseGPT OR Wanda OR "layer-wise pruning" OR "layerwise pruning") AND abs:(token OR tokens OR sample) AND abs:(weighted OR weighting OR importance OR reweight)`
42. `abs:("sample weighting" OR "sample reweighting" OR "sample weights" OR "data weighting" OR "data reweighting" OR "weighted calibration") AND abs:(quantization OR quantized OR "post-training quantization")` (from 2020)
43. `abs:(Hessian OR "second-order" OR "second moment" OR XX) AND abs:(calibration OR "calibration data") AND abs:(quantization OR quantized) AND abs:(mismatch OR misaligned OR alignment OR "distribution of") AND abs:(deployment OR inference OR "test-time" OR downstream)`
44. `abs:("noise shaping" OR "sigma-delta" OR "error feedback" OR "Jensen gap" OR "Jensen bias") AND abs:(quantization OR quantized) AND abs:("language model" OR LLM OR transformer OR softmax)` (nothing relevant to token weighting)
45. `abs:("diffusion language model" OR "diffusion LLM" OR dLLM OR "masked diffusion") AND abs:(quantization OR PTQ) AND abs:(calibration)`

WebSearch (2026 venues / phrasing): "entropy-weighted calibration GPTQ reasoning model quantization token weighting Hessian 2026"; `"RSQ" "Learning from Important Tokens Leads to Better Quantized LLMs" ICML 2025` (→ TMLR 2026, not ICML); `"token" "entropy" weighted Hessian calibration quantization reasoning LLM "GPTQ" high-entropy tokens calibration weighting arxiv 2026` (extended); `openreview "calibration" quantization reasoning model "entropy" token weighting Hessian GPTQ "reasoning traces" 2026 submission` (no OpenReview hit).

## Candidates (full text read via arxiv_fetch unless marked)

### Priority 1

**2503.01820 — RSQ: Learning from Important Tokens Leads to Better Quantized LLMs** (Sung, Yadav, Li, Yoon, Bansal; arXiv 2025-03-03; TMLR 2026 per JMLR bib; code github.com/ylsung/rsq). *partially_preempts.*
- Sec. 4.2: objective `‖(WX − W̃X)R‖² = Σ_i ‖r_i (W x_i − W̃ x_i)‖²`, "a modified Hessian matrix H_RSQ = 2XR²Xᵀ". Exactly H_w = Σ w_t x_t x_tᵀ.
- Sec. 4.3: "we avoid using any global information, such as model gradients, as it would violate the layer-wise assumption"; importance computed per layer from inputs; candidates TokenFreq, ActNorm, ActDiff, TokenSim, AttnCon (+ positional First-N, First&Last-N). No output-entropy or Fisher weight. Dataset expansion (shift by T/M, M = 8) to de-bias positions.
- Sec. 5.2: AttnCon best (Wiki ppl 9.028 at r_min = 0.01 vs ActNorm 9.075, TokenSim 9.047 at r_min = 0.005); "these relatively small r_min indicate that placing lower emphasis on less important tokens … is beneficial". Table 2 (3-bit, WikiText-2 256×4096 calibration): RSQ beats QuaRot by +1.6 / +0.9 / +0.4 avg-acc points on LLaMA3-8B-Instruct / Mistral-NeMo-12B / Qwen2.5-7B-Instruct; GSM8K is among the 11 tasks.
- Gap left open: no entropy/Fisher weight, no reasoning models, no CoT length; generic WikiText calibration.

**2505.07004 — GuidedQuant** (Kim et al.; ICML 2025; v4 2025-09-22). *partially_preempts.*
- Eq. (4): "scale each output error by the gradient of the end loss with respect to that output": `‖(∂ℓ/∂Z^(l)) ⊙ (X W − X Ŵ)‖²_F = Σ_i Σ_j (∂ℓ/∂Z_ij)² (Z_ij − Ẑ_ij)²` — token index i and channel j both weighted. Block-diagonal Fisher `F_j = (1/n) Σ_i (∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)ᵀ`, `H_j = n F_j`; Sec. 3.2 averages H_j within output-channel groups → token weights (mean_j∈group g_ij²) persist.
- Hence the report's sentence "weights output channels but does not change the token mix" is false; GuidedQuant is a gradient-based token-and-channel weighting. The idea = expected-Fisher, channel-isotropic, forward-only special case. Compute: appendix tables list "0.3 h (1×A100)" rows for 7B-class and "2.7 h (1×R6A)" rows; the 70B multi-GPU figure in the report was not verified from the rows read.
- Calibration: RedPajama 1024×4096 (generic); no reasoning-model evaluation.

**2605.08755 — LAQuant: Layer-wise Lookahead Loss for LRM quantization** (Choi, Song, et al., SNU; arXiv 2026-05-09). *partially_preempts (theory + domain explanation); partially contradicts (design direction).*
- Prop. 1: `M_E2E = Jᵀ H_KL J, H_KL = diag(p⋆) − p⋆(p⋆)ᵀ` (categorical Fisher); Lemma 1 / Prop. 3: `‖M_E2E v‖ ≤ 2 max_i p⋆_i(1 − p⋆_i) · σ_max(J)² ‖v‖` while `M_lw = I`. Same object as Lemma 8.1's G_tᵀ J_t G_t; trace identity not stated.
- Sec. 4.1 Eq. (2): calibration-domain effect explained by Hessian-subspace alignment `H_act^cal = E φ_ℓ(x)φ_ℓ(x)ᵀ`, diagnostic top-k principal-angle cosine c̄_k; Fig. 3b reasoning corpora overlap more with inference traces. This is the rival "domain = activation subspace" hypothesis to the idea's "domain = entropy profile".
- Numbers: GPTQ Qwen3-4B W3G128 generic→R1-trace calibration +38.45 pp AIME25 (Table 10); Qwen3-1.7B W3G128 0.00%→22.55% (App. G.1); 24 checkpoints: K/V-cache RMSE on AIME25 traces anti-correlates with Pass@1, r ≈ −0.85.
- App. C Table 6: E2E metric anisotropy ≥30× (non-leading directions median ≤0.05); KV-cache-reducing directions sit in the attenuated subspace → "no scalar reweighting can repair it"; controlled E2E vs LW top-1 agreement 88.44% vs 93.95%. Table 8: ~56% of AIME25 tokens have H < 0.1 nats; LA−LW gap +0.004–0.050 pp at H<0.1, +0.77 pp at H∈[1,2), +2.50 pp at H∈[2,5).
- Implication: the softmax-Fisher factor the idea wants to *multiply in* is the factor LAQuant blames for E2E gradient dilution; arms (c)/(d) need a weight floor and a KV-fidelity check.

### Priority 2

**2606.00206 — Quantized Reasoning Models Think They Need to Think Longer, but They Do Not** (Lotfi, Kirichenko, Li, Liu; Meta FAIR; 2026-05-29). *supports.*
- Fig. 4c: position-level KL (BF16 vs 3-bit AWQ, MATH-500, teacher-forced identical prefixes) vs BF16 next-token entropy, Spearman ρ = 0.92. Fig. 7: overthinking markers 2–4× more likely in top-20 at high-entropy positions. Up to 52% of failures = correct intermediate answer not committed. Logit penalty on 50 markers: CoT −12–23% (range 4.1–28.0%), 5 models, 3 PTQ methods.
- App. A.1: "GPTQ is calibrated on reasoning data to mitigate domain shift: … model-generated reasoning traces on NuminaMath-1.5, 128 sequences of length 2048"; AWQ uses Pile 128×512. So their GPTQ already has the domain fix; the AWQ KL–entropy correlation is with generic calibration.
- Sec. 7: "adapting λ dynamically based on local entropy is a natural extension" — decoding side only. Calibration-side entropy weighting is not considered.

**2606.25519 — Quantization Inflates Reasoning: Token Inflation as a Hidden Cost** (Lian, Krichene, Huang, Tanaka, Ruwase, Zhang, Zhang; UIUC/Microsoft/Anyscale/Snowflake; 2026-06-24). *supports.*
- Sec. 7.2 / Fig. 5 (Qwen3-4B INT3 GPTQ): five corpora (ultrachat_200k, wikitext, c4, OpenR1-Math-220k self-generated, codeforces-cots self-generated); "switching from Wikitext to Math-CoT raises accuracy from 28% to 65% while cutting CoT length from 30k to 12k tokens"; Math-CoT adopted as default. QAT "is the only Pareto-dominant mitigation"; "calibration-data optimization remains strictly below QAT on both axes".
- No analysis of *why* (lexical vs entropy); no token weighting.

**2608.21019 — Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized LMs (DPQ)** (Yang, Hou, et al.; EMNLP Findings 2026). *partially_preempts (arm e, sample level).*
- Doubt score `b(x) = 1 − p0,(1)(x) − p0,(2)(x)`; variants "1 − p0,(1) for confidence-only, H(p0(·|x)) for entropy-only, answerability log-odds, gold-label NLL"; top ⌊sr⌋ by b(x) + generic anchors (WikiText/RandomQA), "run unchanged GPTQ". Prop. 1 top-two boundary fragility. 8 models, 9 benchmarks, 22 comparisons; best recipe is target-dependent (DPQ-r75 for SQuAD2 boundary, milder/entropy-only for broad MCQA).
- Selection acts on whole examples at the answer position, not per-token weights; no reasoning traces.

**2605.27808 — TARQ: Tail-Aware Reconstruction Quantization for Rare-Word Robust ASR** (Wang, Zhao, et al.; McGill/Boson AI; 2026-05-27). *partially_preempts (framing).*
- "Data-aware PTQ minimizes a per-token reconstruction loss … implicitly weighting positions by their empirical frequency"; family `H_ℓ(α) = Σ_f α_f H̄_ℓ^f`, "the loss-sensitivity-aware ideal would set α_f from per-bucket gradient or Fisher information"; RAREBAL = two-bucket trace equalization; "An ideal weighting would be derived from per-position loss sensitivity (e.g., gradient or Fisher information), but this requires labels and backward passes". 8 ASR backbones, W4G128, rare-WER improves without aggregate-WER regression.
- The idea's w_t is a label-free forward-only instance of TARQ's "ideal"; TARQ should be cited as motivation and as evidence that frequency-weighting is the issue.

**2609.17890 — OBC-Prune: Outcome-Based Calibration for LRM Pruning** (Nguyen, Tran, Tran, Le; VinUni/JHU; 2026-09-15). *partially_preempts (plumbing, pruning).*
- Algorithm 1 Stage 3–4: per-token weights `w_t^correct = 1 + β ẽ`, `w_t^wrong = max(ε, 1 − γ ẽ)` (ε = 0.1 "to keep the Hessian positive"), accumulated `C̃_ℓ += Σ_t w_t x_t x_tᵀ`, fed unchanged to Wanda/SparseGPT. Difficulty-matched correct/incorrect rollouts (n = 16, T = 0.8); attention-suppression causal scoring per sentence. DeepSeek-R1-Distill-Qwen 1.5B/7B/14B; shorter completions at 50% sparsity; delays collapse to 70–80% sparsity.
- Weight source is outcome/causal, not entropy; pruning not quantization.

**2606.13233 — ReSET: Step-Aware Temperature Scaling for NVFP4 Reasoning** (Lee, Lee, et al.; Hanyang/Xenoscube; 2026-06-11). *contradicts (partially).*
- Abstract: "quantization increases incorrect sampling at low-entropy symbolic tokens, while causing over-concentration on a small set of tokens in high-uncertainty reasoning steps"; step-level entropy temperature scaling, +~2 points NVFP4. If low-entropy symbolic tokens are a real failure locus, a weight ∝ 1 − ‖p‖² (→0 there) under-protects them; the idea needs a floor and a low-entropy-token error metric.

### Priority 3

**2609.00049 — REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent** (Zhang et al.; PKU/ZTE; 2026-08-30). *tangential / partially_preempts (formulation).*
- Eq. (4): `E_t[L_KL(y_t)] ≈ ½ E_t[Δy_tᵀ H_t Δy_t]`, H_t per-token KL Hessian; Fisher–Hessian identity `E_t[H_t] = E_t[g_t g_tᵀ]`; Prop. 4.1 mean-field decoupling to aggregated F = (1/T)Σ g_t g_tᵀ, explicitly trading token-wise variation for cross-channel structure. The idea is the complementary approximation (keep per-token scalar tr, drop channel structure). Describes GuidedQuant as "per-token saliency scores … used to weight the GPTQ Hessian". G²PTQ (2609.31009) Theorem 4 similarly recovers per-token squared gradients from one backward pass under causal-sparsity assumptions.

**2606.06547 — FAIR-Calib (ICML 2026 poster)** (Huang et al.). *tangential.*
- dLLM PTQ: "reweighted hidden-state MSE" layer-wise calibration with position prior = frontier-hit accumulation + teacher sharpness gate ("token-probability, negative entropy, or margin"); Lemma 4.6 `KL(p‖q) = D_f(z′‖z) ≤ ¼‖z′ − z‖²` via ½-smoothness of LSE (worst-case, entropy-independent constant); "theoretically justify our weighted objective as a surrogate for output KL". Same proof shape as Lemma 8.1 but without the entropy-dependent trace; weights are not Fisher-trace weights.

**2607.00908 — Beyond Activation Alignment: Alignment-Diversity Tradeoff in Task-Aware LLM Quantization (TASA)** (Wang et al.; 2026-07-01). *contradicts (partially).*
- "using only target-task calibration data can degrade post-quantization performance, whereas incorporating general-domain data stabilizes sensitivity estimation"; "Diversity as Hessian regularization … General-domain data lifts the suppressed trailing eigenvalues". Mixed-precision setting (PPL/Hessian proxies). Confound for arm (b): CoT-only calibration may not be the upper bound; a mixed arm is needed.

**2511.18864 — Think Before You Prune: Selective Self-Generated Calibration for Pruning LRMs (SSGR)** (Xiang, Ji, Li, Zhang; Soochow; 2025-11-24; under review). *supports.*
- Self-generated reasoning traces as calibration improve LRM pruning (7.1–8.4% on complex tasks); model-defined difficulty (dense-model accuracy ≤50% = Hard) and "moderately long" traces are best. Problem-level analog of entropy stratification (harder problems ≈ more high-entropy forks), without token weights.

**2603.16105 — Frequency Matters: ZipCal** (Monaco, Cunegatti, Vella, Iacca; EMNLP 2026 main). *tangential (confound).*
- "calibration entropy is the dominant predictor of post-compression resilience, correlating strongly with performance retention (ρ = 0.94, p ≤ 0.005)" — but this 'entropy' is lexical diversity / activation entropy of the calibration set (Zipfian sampling), not next-token predictive entropy. Any entropy-stratified arm must control lexical diversity.

### Priority 4

**2504.04823 — Quantization Hurts Reasoning?** (Liu et al.; COLM 2025). *supports.*
- Table 2 (R1-Distill-Qwen-1.5B): "switching the GPTQ calibration set from WikiText2 to Numina-Math-1.5 leads to an average accuracy gain of 9.8%"; "The domain of calibration data affects GPTQ, but not the rest" (AWQ, KVQuant*, FlatQuant); Fig. 9 activation-distribution gaps. Confirms the effect is specific to Hessian-compensation methods — consistent with a Hessian-weighting explanation, agnostic between subspace vs entropy.

## Also seen (not promoted)
- 2608.01078 AYOT/BitTern (Intel, 1.58-bit PTQ of reasoning LLMs; self-generated CoT calibration from MetaMathQA/OpenCodeInstruct; calibration-size and domain ablations) — supports domain effect.
- 2510.10618 (NeurIPS 2025) calibration-data curation; activation-space patterns determine optimality — supports, no entropy.
- 2609.39801 RATIO (decoding-side token-specific penalties for quantized reasoning models; GPTQ-W3 R1-Distill-Qwen-1.5B +3.58 pp, CoT −25.7%) — decoding side.
- 2601.14888 (QAT for reasoning LLMs; align QAT data domain with calibration set) — supports.
- 2405.00632 (GPTQ 4-bit "disproportionately affects samples where the full model exhibited low confidence") — supports premise at sample level.
- 2607.07964 KronQ (COLM 2026; gradient covariance on output side; Kronecker Hessian) — channel-side weighting, tangential.
- 2503.04704 "layer-level entropy-weighted quantization" — same words, layer bit allocation, unrelated.
- 2506.01939 (NeurIPS 2025) high-entropy forking tokens in RL — conceptual source for "fork", not quantization.
- 2410.17170 / 2311.09755 (Williams & Aletras) self-calibration and calibration-data impact — background.
- 2607.11317 / 2607.20129 (e-CUSUM decoding monitors for quantized reasoning models) — decoding side.

## Consequences for the idea
1. Rewrite the novelty paragraph: cite RSQ (token-weighted Hessian), GuidedQuant (token-and-channel gradient weighting — correct the "does not change the token mix" claim), DPQ (entropy-based example selection), OBC-Prune (per-token weights for LRM pruning), TARQ (frequency-weighting diagnosis). The defensible contribution is: a closed-form, forward-only, gradient-free token weight derived from the softmax Fisher trace, evaluated on reasoning models, used to disentangle domain vs entropy-profile.
2. Lemma 8.1's second-order step is standard (LAQuant Prop. 1/Lemma 1; FAIR-Calib Lemma 4.6); only the trace identity and its use are new. Say so.
3. Add LAQuant's Hessian-subspace alignment as the explicit rival hypothesis; add an arm that matches activation-subspace overlap while varying entropy profile (or vice versa).
4. Add a weight floor (RSQ r_min; OBC-Prune ε = 0.1; TASA diversity) and report error on low-entropy symbolic tokens (ReSET) and K/V-cache RMSE (LAQuant) — both are places where pure tr(J_t) weighting can hurt.
5. Baselines must include RSQ-AttnCon and GuidedQuant at matched calibration data, since both already exist as token-weighting schemes; 2606.00206's GPTQ baseline is already Numina-calibrated.
