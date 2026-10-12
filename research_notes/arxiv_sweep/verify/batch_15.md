# Citation verification — batch 15

IDs: 2606.29239, 2606.20254, 2504.04823, 2505.07004. Report: `reports/LLM 양자화 연구 아이디어 10선.md` (lines 286, 296, 298, 301). All four papers exist on arXiv; full text fetched via `scripts/arxiv_fetch.py`. Quoted paper text is untrusted data.

---

## 2606.29239 — QuantGuard: Learnable Rounding for Repairing Quantization-Conditioned Backdoors in LLMs

- **Authors**: Aoying Zheng, Anqi Du, Zizhuang Deng, Yuxuan Chen, Shanqing Guo, Kening Zheng. Published 2026-06-28, v2 2026-08-27. cs.CR.
- **Venue**: no comment/journal_ref; IEEE two-column style ("TABLE V", Roman-numeral sections). WebSearch found only aggregator listings. → preprint/unknown.
- **Report claim (line 286, tag [S])**: Idea 7's dither defense "must be compared against 2026 defenses such as QuantGuard ([2606.29239]) and QVec ([2606.20254])". No numbers attributed.
- **Verdict: confirmed.** It is a 2026 pre-quantization QCB defense for LLMs, so the comparison requirement is correctly stated.
- **What the paper is** (abstract): "a proactive pre-quantization defense that learns safe rounding adjustments through differentiable optimization ... introduces differentiable rounding control variables and combines error-guided rounding reversal constraints, output-distribution consistency, and weight-distance regularization". It "does not modify existing quantization algorithms" and uses BitsAndBytes INT8/FP4/NF4.
- **Scope** (Sec. V setup): "six widely used models, including Qwen2.5-Coder-1.5B, StarCoder-1B, Phi-2-2.7B, LLaMA3-8B, Gemma-2B, and DeepSeek-Coder-6.7B, under three quantization precisions, namely INT8, FP4, and NF4"; scenarios: vulnerable code generation, over-refusal, content injection (all following Egashira et al. [13]).
- **Headline numbers** (Sec. I): "on DeepSeek-Coder-6.7B, QuantGuard restores code security ... from 12.8% to 90.2% under INT8 quantization. On Gemma-2B, it reduces ... Informative Refusal ... from 31.3% to 0.8% under FP4. On LLaMA3-8B, it reduces ... Keyword Occurrence ... from 90.4% to 0.0% under NF4."
- **Baselines** (Sec. V.F): SFT, DPO, FlipGuard (FG-M, matched modification budget), EFRAP-LLM (their adaptation). "EFRAP-LLM ... introduces noticeable utility degradation; for example, on Gemma-2B, MMLU drops to 27.6%." No dither or random-noise defense is evaluated — consistent with the report's "dither를 방어로 검토한 문헌은 검색에서 나오지 않았다".
- **Adaptive attack** (Sec. V.E / App. A): defense-aware attacker that "tightens the quantization-feasible intervals"; under INT8 vulnerable code, attack lowers CodeSec to 23.3% (StarCoder-1B) / 18.2% (Phi-2), QuantGuard restores to 60.4% / 75.9% (Table VIII). Note: recovery under the adaptive attack is markedly lower than under the non-adaptive one (89.6% for Phi-2).
- **Cost** (Sec. V): "StarCoder-1B uses 1 RTX PRO 6000 GPU and takes about 8 minutes; Phi-2-2.7B uses 2 GPUs and takes about 18 minutes; ... LLaMA3-8B ... 5 GPUs and takes about 38 minutes."
- **Relevant for Idea 7**: QuantGuard's stated design constraint is compatibility with unmodified quantizers; it criticises EFRAP because it "modifies rounding strategy during quantization, making it difficult to integrate with practical LLM quantizers" (Sec. II). A subtractive-dither quantizer is exactly a modified rounding/dequantization rule, so the report should expect this objection and address deployment (dequant kernels must regenerate u from the secret seed).

---

## 2606.20254 — Quantization as a Malicious Task: Removing Quantization-Conditioned Backdoors via Task Arithmetic (QVec)

- **Authors**: Kaihsun Yang, Min-Yan Tsai, Chia-Mu Yu. Published 2026-06-18 (v1 only). cs.CR.
- **Venue**: no comment/journal_ref; LNCS template placeholders remain in the text ("Abbreviated paper title", "F. Author et al."), i.e., an LNCS submission; WebSearch gives no acceptance. → preprint/unknown.
- **Report claim (line 286, tag [S])**: same sentence as above — a 2026 defense to compare against.
- **Verdict: confirmed.**
- **Method** (Sec. 5.1): δ = Q(W) − W (Eq. 2); correction W' = W − αδ (Eq. 5); deploy Q(W'). "QVec requires no retraining, no trigger samples, and only a single quantization pass ... together with a lightweight hyperparameter search" (α chosen by validation accuracy / MMLU with tolerance τ, Sec. 5.2).
- **No theorem.** Grep for Theorem/Proposition/Lemma: none. The "theoretically ... show that the malicious shift is structured rather than random noise" claim is an informal argument in Appendix A (distribution-shape analysis), not a proof.
- **Scope**: vision (ResNet18, VGG13, MobileNetV2; CIFAR-10, Tiny-ImageNet; "8-bit and 4-bit settings using deterministic round-to-nearest") and LLM following Egashira [10]: Gemma-2B (content injection, over-refusal), StarCoder-1B (vulnerable code) under "LLM.int8(), FP4, and NF4". LLM baseline numbers are taken from [10], not re-run ("For LLM settings, we follow the baseline results reported in [10]").
- **Numbers**: content injection keyword rate "restores ... to near the original baseline (0.07%) across LLM.int8(), FP4, and NF4"; over-refusal "from 60.67% to 10.3% under FP4"; vulnerable code: QVec "in many configurations ... even surpasses the original full-precision baseline" (Table 8).
- **Directly relevant to Idea 7 (dither / noise defenses)**: QVec evaluates **Gaussian noise injection** as a defense baseline (σ ∈ {1e-2,1e-3,1e-4} for vision; σ = 1e-3 for LLMs "reported in [10]"). Finding (Sec. 6, "Limitations of Gaussian Noise"): "the effect is highly sensitive to σ and model architecture. Small σ fails to suppress attacks, whereas large σ severely degrades CA. ... σ = 10−3 ... is insufficient to eliminate quantization backdoors in the more complex vulnerable code generation task for StarCoder-1B. ... noise injection is unstable and attack-dependent." Tiny-ImageNet: "CA drops to 13.01% in 4-bit Exploiting". No mention of dither. The report's sentence "dither를 방어로 검토한 문헌은 검색에서 나오지 않았다" stands, but the report should cite that random *Gaussian* weight-noise has been tried (Egashira et al. and QVec) and found unstable, and position subtractive dither as structurally different (input-independent uniform error at exactly RTN MSE, no σ to tune).

---

## 2504.04823 — Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models

- **Authors**: Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, Lu Hou. Published 2025-04-07, v3 2025-08-18.
- **Venue**: **COLM 2025** (arXiv comment; running header "Published as a conference paper at COLM 2025"). Report's "COLM 2025" is correct.
- **Report claim (line 296; report tag [V: repo], JSON tag "?")**: "Liu et al.은 추론 모델 PTQ를 체계 평가하고 자가생성 추론 트레이스를 캘리브레이션에 썼다".
- **Verdict: confirmed.**
  - Systematic evaluation: "the first systematic study on quantized reasoning models, evaluating ... DeepSeek-R1-Distilled Qwen and LLaMA families ranging from 1.5B to 70B parameters, QwQ-32B, and Qwen3-8B ... weight, KV cache, and activation quantization" (abstract).
  - Self-generated traces for calibration: App. A.1 (GPTQ): "we use reasoning models to self-generate reasoning data on Numina-Math-1.5 dataset, and randomly sample 128 text sequences of length 2048 to construct the calibration set." App. A.3 (SmoothQuant/QuaRot): "128 text sequences of length 2048 from the self-generated reasoning dataset as the calibration set". App. C.1: "The reasoning data is self-generated by the reasoning model, given the problems in the original dataset".
  - Caveat: not all methods use it — AWQ and KVQuant* use 128×512 Pile sequences; FlatQuant uses WikiText2 (extended to 4K length for 1.5B/7B).
- **Exact numbers the report could use** (Sec. 4.2, Table 2, DeepSeek-R1-Distill-Qwen-1.5B): "switching the GPTQ calibration set from WikiText2 to Numina-Math-1.5 leads to an average accuracy gain of 9.8%"; "The domain of calibration data affects GPTQ, but not the rest of the quantization algorithms" (KVQuant*, FlatQuant are robust because they use calibration only for scale/outlier-channel statistics).
- **Contradiction to flag for Idea 8's premise** (the hypothesis says quantization inflates CoT length via over-sampled hesitation tokens): abstract: "Contrary to expectations, quantized models do not exhibit increased output lengths." Sec. 4.1: "for various quantization configurations (e.g., W4G128, W8A8KV8, and KV4), the generated output lengths closely align with those of the BF16 models. However, under more aggressive low-bit quantization (e.g., W3G128 and W4A4KV4), LLMs tend to produce longer outputs. This effect is particularly pronounced in smaller models like DeepSeek-R1-Distill-Qwen-1.5B". So the length-inflation effect the report attributes to quantization (via [2606.00206]) is, per this paper, confined to aggressive W3/W4A4 settings on small models. The report should scope the claim accordingly.
- Also useful: "while lossless quantization can be achieved with W8A8 or W4A16 quantization, lower bit-widths introduce significant accuracy risks" (abstract).

---

## 2505.07004 — GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

- **Authors**: Jinuk Kim, Marwa El Halabi, Wonpyo Park, Clemens JS Schaefer, Deokjae Lee, Yeonhong Park, Jae W. Lee, Hyun Oh Song. Published 2025-05-11, v3 2025-09-22.
- **Venue**: **ICML 2025** (arXiv comment). Report's "ICML 2025" is correct.
- **Report claims (line 298, tag [S]; line 301)**:
  1. "end-loss 기울기로 출력 채널을 가중" — weights output channels by end-loss gradients.
  2. "비용: 1×A100 0.3h@7B, 6–7×A100 2.7h@70B".
  3. "토큰 믹스는 바꾸지 않는다" (does not change the token mix) and the gap claim "엔트로피 가중 캘리브레이션 토큰을 다룬 논문은 검색에서 없었다".
  4. Line 301: the report's token-weighted Hessian is "(GuidedQuant의 대각·토큰 수준 근사)".
- **Verdict: partially confirmed.**
  - (2) Numbers: Table 9 ("Total GPU cost and disk usage incurred during the gradient and Hessian caching processes"), Gradient Caching column: Llama-2-7B "0.3 h (1×A100)"; Llama-2-70B "2.7 h (**6×A100**)"; Llama-2-13B "0.6 h (2×A100)". **Correction**: write "6×A100", not "6–7×A100". Also these are *gradient-caching* costs only; Hessian caching adds "0.3 h (4×R6A)" (7B, g=1) and "3.5 h (4×R6A)" (70B, g=1; "5.8 h" for g=2), and quantization itself adds e.g. 0.9 h on 1 RTX 6000 Ada for 7B 4-bit LNQ (App. B.3) and "less than three hours when using 8 RTX 6000 Ada GPUs" for 70B. The same gradient-caching cost is listed for SqueezeLLM's weighted k-means, so it is not GuidedQuant-specific. Calibration data: "1024 sentences of the RedPajama dataset, each containing 4096 tokens".
  - (1)/(3)/(4) Mechanism — the report's description understates what GuidedQuant already does. The Fisher block is F_j^{(l)} = (1/n) Σ_{i=1}^{n} (∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)ᵀ (Sec. 3.1) where X^{(l)} ∈ R^{n×d_in} and "n is much larger than both" d_in, d_out — i.e., i indexes calibration **token positions**. Since ∂ℓ_i/∂w_j = (∂ℓ/∂z_ij) x_i, the per-channel objective is H_j = Σ_i (∂ℓ/∂z_ij)² x_i x_iᵀ: a **token-weighted** XᵀX with weight = squared end-loss gradient at that token and output channel. Algorithm 1 then averages these squared gradients over groups of output channels ("s_k ← (1/|J_k|) Σ_{j∈J_k} (∂ℓ/∂z_j)²"). With g = 1 this collapses to a single per-token scalar weight — exactly the structure of the report's H_w = Σ_t w_t x_t x_tᵀ, differing only in how w_t is obtained (squared end-loss gradient vs. 1 − ‖p_t‖²).
    - So "토큰 믹스는 바꾸지 않는다" is true only for the *data* (RedPajama, equal sampling) — the *objective* already re-weights tokens. The gap sentence should be narrowed to: no paper weights calibration tokens by an entropy/collision-entropy proxy that needs no backward pass; GuidedQuant does weight tokens, but via end-loss gradients that require gradient caching (Table 9 cost).
    - "대각" (diagonal) in line 301 is inaccurate: GuidedQuant explicitly rejects SqueezeLLM's diagonal Fisher ("the diagonal approximation is highly inaccurate", Sec. 2) and keeps **block-diagonal** d_in×d_in blocks per output channel (Sec. 3.1, App. D.11: "GuidedQuant maintains the blocks corresponding to each output channel"). Suggested wording: "GuidedQuant의 블록대각(출력채널별)·그룹 평균 Fisher 근사와 같은 구조로, 토큰 가중치를 기울기 대신 엔트로피로 대체".
  - Theory: Proposition 4.1 proves LNQ's objective is "monotonically non-increasing" (alternating codebook/assignment updates); nothing is proven about the Fisher approximation quality itself (shown empirically, Figs. 3–4, Sec. D.5).

---

## Summary table

| id | exists | verdict | key correction |
|---|---|---|---|
| 2606.29239 | yes | confirmed | none; venue unknown (preprint) |
| 2606.20254 | yes | confirmed | none; add that QVec/Egashira evaluated Gaussian-noise defenses (found unstable); venue unknown (LNCS template) |
| 2504.04823 | yes | confirmed | none; flag that the paper finds no output-length increase except W3/W4A4 on small models |
| 2505.07004 | yes | partially confirmed | 70B cost is 6×A100 (not 6–7×), gradient-caching only; GuidedQuant's Fisher already token-weights the Hessian (block-diagonal, not diagonal) |
