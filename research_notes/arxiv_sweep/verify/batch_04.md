# Citation verification — batch 04

IDs: 2606.08635, 2606.13233, 2604.08976, 2605.17757
Source texts: `scratchpad/papers/<id>/<id>.txt` (arxiv_fetch.py, 2026-10-10). Quotes below are verbatim from the paper text (untrusted data, quoted only). Line numbers refer to the fetched .txt files.

Summary table

| id | paper | report claim | verdict |
|---|---|---|---|
| 2606.08635 | SpectrumKV (Yang, single author, preprint) | e^{±2η} multiplicative softmax bound; "sharper attention more sensitive" direction | **confirmed** (with caveats: inline bound, informal proofs, main entropy claim is a Conjecture) |
| 2606.13233 | ReSET (Hanyang Univ. / Xenoscube, preprint) | "NVFP4 shifts token entropy", observed at the LM-head side, no derived correction | **partially confirmed** (entropy shift is real but bidirectional; paper is whole-model W4A4, says nothing about the LM head) |
| 2604.08976 | Cacioli, metacognitive geometry (independent, preprint) | "measurement that GPTQ 4-bit worsens ECE" | **incorrect attribution** — that is Proskurina et al. 2024 (arXiv 2405.00632, Findings of NAACL 2024), cited in this paper's related work; 2604.08976 itself uses Q5_K_M GGUF vs f16 and never measures ECE or GPTQ |
| 2605.17757 | OSCAR (Together AI / U. Sydney / UIUC, preprint) | "query-covariance weighted transform beats Hadamard in MHA"; adjacency risk to the heteroscedastic-bias idea | **partially confirmed** (OSCAR composes U_Q *with* Hadamard rather than replacing it; theorem is for a frozen-error surrogate; no softmax-bias analysis, so adjacency is weaker than feared) |

---

## 2606.08635 — SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer for Prefill-Decode Disaggregated LLM Serving

- Exists: yes. Author: Pengju Yang (single author; contact `yangsteve1223@github`, no institutional affiliation). Published 2026-06-07, v1. Comment: "28 pages, 13 figures, 8 tables". Venue: none stated; no venue found via web search → preprint/unknown.
- What it is: a systems paper on PD-disaggregated serving. Per-token precision tiers (FP16 / INT8 / INT4) for KV transfer, with a deployment-time NIAH probe to decide whether INT4 is tolerable. Theory is a short Section 2 (Lemma 1 value tail-mass, Eq. 10 softmax bound, Lemma 2 amplification bound, Conjecture 1, Proposition 1 exchange argument, Proposition 2 layer-wise propagation).

### Report claims checked

Report (idea on heteroscedastic softmax bias): "어텐션 민감도 이론은 1차 Jacobian 상한([2608.20988] [S])과 e^{±2η} 승법 상한([SpectrumKV 2606.08635] [S])뿐이고, 둘은 '날카로운 어텐션이 더 민감하다/덜 민감하다'로 서로 반대 직관을 준다."

1. e^{±2η} multiplicative bound — **confirmed**. Sec. 2.4 "Key quantization and softmax amplification", Eq. (10), lines 268–272:
   > "Let zj = q⊤kj/√d and z̃j = zj + ηj with |ηj| ≤ η. Then the softmax probabilities obey e−2η pj ≤ p̃j ≤ e2η pj. (10)"
   The paper itself says: "The bound is loose, but it captures the qualitative risk: if attention is low-entropy and dominated by a few positions, small key perturbations around those positions can cause large relative changes in the normalized distribution." (lines 273–275)

2. Direction "sharper attention is MORE sensitive" (attributed to SpectrumKV) — **confirmed** as the paper's claim, in *relative* terms. Lemma 2 (Sec. 2.5, lines 297–309):
   > "Let S = {j : zj ≥ maxk zk − log(1/ϵ)} ... Then for any j ∈ S: |p̃j − pj|/pj ≤ e2η − 1 + (|S|·e2η − |S|)/(1 + |S|·e−2η). (11) When |S| is small (low-entropy regime) and η is moderate, the bound scales roughly as e2η/|S|, meaning fewer dominant tokens yield larger relative perturbations."
   > "The practical implication is that models with local-dominant attention patterns (small |S|, low entropy) are more vulnerable to key quantization perturbations than models with broader sink or hybrid patterns." (lines 331–333)

### Caveats the report should carry
- Eq. (10) is an inline, unnamed bound, not a theorem; it is entropy-independent. The entropy dependence comes only from Lemma 2, whose "proof" is a sketch: "The tightest bound arises when the partition is sharp ... Then Z̃ ≈ Z · e±η · |S|, giving the stated result. The full derivation tracks the contribution of non-dominant tokens, which adds the correction term." (lines 327–330). The full derivation is not in the paper.
- The entropy-threshold claim is explicitly a conjecture: "Conjecture 1 (Entropy-dependent INT4 tolerance) ... H∗ ≈ log n − 2 ... We state this as a conjecture rather than a theorem because the precise threshold H∗ depends on layer-wise interactions that we have not fully characterized." (lines 334–343)
- Lemma 2 bounds *relative* error |p̃j − pj|/pj. A first-order Jacobian bound (diag(p) − pp⊤) bounds *absolute* error and vanishes as p → one-hot. The "opposite intuitions" the report mentions are therefore relative-vs-absolute, not a genuine contradiction; the report should say so explicitly when it claims the second-order term "reconciles" them.
- Empirical support in the paper: Qwen2.5-7B int4 key cosine 0.9770–0.9792 across models, value cosine 0.9895 (highest) yet PPL explodes under INT4 (Sec. 7/8, lines 2010–2020) — used to argue the failure is "distribution-level, not vector-level".
- Single-author, unaffiliated preprint; weight accordingly. The [S] tag is appropriate; do not upgrade to [V] on the theory.

---

## 2606.13233 — ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling

- Exists: yes. Authors: Sihwa Lee*, Janghwan Lee*, Donghoon Yoo, Jae Gon Kim, Hanyul Ryu, Soojung Ryu, Jungwook Choi† (Hanyang University; Xenoscube Korean Inc.). Published 2026-06-11, v1. No comment/journal_ref; web search shows only mirrors → preprint/unknown. Code: github.com/aiha-lab/ReSET.
- What it is: decoding-time temperature scaling for NVFP4 W4A4 LRMs, driven by token- and step-level entropy, plus a CUDA-core small-M NVFP4 GEMM kernel.

### Report claims checked

Report: "LM head 쪽은 ReSET([2606.13233] [S])이 'NVFP4가 토큰 엔트로피를 이동시킨다'고 관찰했고, ... 유도된 보정은 없다."

1. "NVFP4 shifts token entropy" — **confirmed**, but the shift is *bidirectional* and step-dependent. Appendix (lines 1585–1596), Fig. 8:
   > "Fig. 8 (left) shows the mean entropy shift ∆Ht = HtPTQ − HtFP16 across token entropy bins under low- and high-uncertainty reasoning steps. For low-entropy tokens, quantization consistently increases entropy, with larger shifts observed in high-uncertainty steps. In contrast, for very high-entropy tokens (Ht ≥ 2.0), the entropy shift becomes negative in high-uncertainty steps, indicating that quantization instead sharpens the token distribution."
   > "Overall, quantization introduces two distinct effects: entropy inflation for low-entropy symbolic tokens and entropy collapse for high-entropy tokens in uncertain reasoning steps."
   Also Sec. 4 (line 451): "NVFP4 compresses the entropy of high-entropy tokens in uncertain steps (e.g., 'Alternatively'; Fig. 8)". Sec. 3.1 (lines 211–214): errors "are not driven by top-1 flips ... the correct token ('three') remains the most likely, but a non-top-1 alternative receives enough probability mass to be occasionally sampled".

2. "LM head 쪽" framing — **not supported by the paper**. The setting is whole-model W4A4: "All models are quantized from public BF16 checkpoints to real NVFP4 W4A4 with NVIDIA ModelOpt (E2M1 elements, E4M3 block scales, group size 16); the KV cache remains in BF16." (Sec. 6.1, lines 655–657). The text never mentions `lm_head` or the output projection; the entropy shift is a property of the output distribution under full-network quantization, not a measured LM-head effect. Report should say "출력 분포(로짓) 쪽" or "디코딩 분포 쪽", not "LM head 쪽".

3. "No derived correction" — **confirmed**. Temperatures are empirical hyper-parameters: "Tlow is selected per (model, task) on a held-out calibration split disjoint from evaluation (Table 13); τ0 is the 80th-percentile token entropy on the same split. We use w=32 (Table 9)." and "empirically, Thigh = 1.0 consistently improves accuracy (Table 10)" (lines 451–455). No quantization-noise model, no closed-form temperature. The paper's own limitations: "the optimal entropy thresholds and temperature hyperparameters may vary depending on model-specific entropy characteristics" (lines 2160–2162).

### Numbers worth recording
- Models: R1-Distill-Qwen-7B/14B, Qwen3-8B/14B/32B; 8 seeds; AIME-120, GPQA-Diamond, LiveCodeBench; top-p 0.95, 32k max tokens; baselines RTN, BRQ, 4/6, MR-GPTQ at T=0.6; "ReSET is applied on top of RTN" (Sec. 6.1).
- Accuracy: "up to ∼2 points over the NVFP4 baseline" (abstract); "largest gain on AIME-120 (+2.6 over the NVFP4 baseline)" (Sec. 6.2, line 666, Table 1).
- Kernel: "1.57–2.49× kernel-level speedup over vLLM-CUTLASS at M=1–8, and up to 1.97× E2E speedup over BF16 on B200" (line 87–88, Qwen3-32B).
- Entropy analysis sample: 1.5M tokens from R1-Qwen-14B (Sec. 3.1).

---

## 2604.08976 — Quantisation Reshapes the Metacognitive Geometry of Language Models

- Exists: yes. Author: Jon-Paul Cacioli (Independent Researcher, Melbourne). Published 2026-04-10, v1. Comment: "10 pages, 2 figures, 5 tables. Pre-registered study. Code and data: https://github.com/synthiumjp/sdt-calibration". Venue: none → preprint/unknown.
- What it is: a signal-detection-theory metacognition study (meta-d′, M-ratio = meta-d′/d′, Type-2 AUROC) on Llama-3-8B-Instruct, comparing llama.cpp GGUF **Q5_K_M vs f16** on 3,000 questions across four knowledge domains; plus a pre-registered (null) domain-conditional SFT intervention.

### Report claim checked

Report: "GPTQ 4-bit가 ECE를 악화시킨다는 측정([2604.08976] [S])이 있으나 유도된 보정은 없다."

**Incorrect attribution.** 2604.08976 does not use GPTQ, does not quantise to 4-bit, and does not compute ECE. The sentence the report paraphrases is this paper's *related-work summary of another paper*, Sec. "Quantisation and confidence" (lines 91–93):
> "Proskurina et al. (2024) showed that GPTQ 4-bit quantisation decreases confidence on true labels, increases calibration error, and disproportionately affects samples where the unquantised model was already uncertain."
Reference entry (lines 857–858): "Proskurina, I., Brun, L., Metzler, G., & Velcin, J. (2024). When quantization affects confidence of large language models? Findings of NAACL 2024, 1918–1928." → arXiv **2405.00632** (fetched; comment "Accepted to NAACL 2024 Findings"). Its abstract: "quantization with GPTQ to 4-bit results in a decrease in confidence regarding true labels"; its conclusion (line 745): "We demonstrate that quantization leads to an increase in calibration error and statistically significant changes in confidence levels for correct predictions." It uses ECE for binary tasks and ACE otherwise (lines 526–531).

What 2604.08976 actually measures (its own grep: no "ECE", no "GPTQ" outside related work/limitations):
- Confidence = "NLP (mean token log-probability of the generated answer)" (line 209); metrics meta-d′, M-ratio, AUROC2; llama-cpp Vulkan on AMD RX 7900 GRE, T = 1.0.
- Finding: "M-ratio profiles across four knowledge domains are uncorrelated between formats (Spearman ρ = 0.00). Arts & Literature moves from worst-monitored (M-ratio = 0.606 at Q5_K_M) to best-monitored (1.542 at f16). Geography moves from well-monitored (1.210) to under-monitored (0.798). However, Type-2 AUROC profiles are perfectly stable across formats (ρ = 1.00)" (abstract). "Arts d′ drops by 37% from Q5_K_M to f16, Science d′ increases by 26%" (line 400).
- The "NLP gap doubling 0.076 → 0.152 in Science" is an effect of the SFT intervention, not of quantisation.
- Limitations (lines 795–808): the format comparison "is reported as a post-hoc diagnostic analysis"; "The study examines one model at two quantisation levels. Whether the metacognitive geometry restructuring generalises across models, model scales, and quantisation methods (GPTQ, AWQ, INT8) is unknown. The pre-registered replication on Mistral-7B was not conducted"; "Several conditions produce d′ values below 0.5 ... where M-ratio becomes unstable".

### Correction for the report
Replace "GPTQ 4-bit가 ECE를 악화시킨다는 측정([2604.08976])" with "GPTQ 4-bit가 참 라벨 신뢰도를 낮추고 calibration error(ECE/ACE)를 키운다는 측정([Proskurina et al. 2405.00632, Findings of NAACL 2024] [V])". If 2604.08976 is kept, cite it for a different fact: "Q5_K_M vs f16에서 도메인별 M-ratio 순위가 완전히 재배열되지만(ρ=0.00) Type-2 AUROC 순위는 불변(ρ=1.00)이다 — 양자화가 신뢰도 분포/d′를 비례적이지 않게 이동시킨다는 단일 모델·사후 분석 증거 [S]". Note that the 2604.08976 result is "confidence structure changes" — it is not evidence about calibration error magnitude at all.

---

## 2605.17757 — OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization

- Exists: yes. Authors: Zhongzhu Zhou*, Donglin Zhuang*, Jisen Li, Ziyan Chen, Shuaiwen Leon Song, Ben Athiwaratkun, Xiaoxia Wu† (Together AI; University of Sydney; UIUC). Published 2026-05-18, v1. Comment: "35 pages, 10 figures". Venue: none stated; web coverage (MarkTechPost 2026-05-25, techstrong.ai) describes it as open-sourced in SGLang → preprint/unknown.
- What it is: INT2 KV-cache quantization with offline-calibrated rotations. Key rotation R_K = U_Q · H_Had · P_br where U_Q is the eigenbasis of the query covariance; value rotation R_V = U_S · H_Had · P_br where U_S is the eigenbasis of V⊤S⊤SV (score-weighted). Per-layer percentile clipping, BF16 sink (S=64) + recent window (R=256), custom INT2 paged-attention kernels (SGLang/vLLM).

### Report claims checked

Claim A (MLA idea): "MHA에서는 query 공분산 가중 변환이 Hadamard를 이긴다는 주장이 있다 (OSCAR [2605.17757] [S], ...)."

**Partially confirmed.**
- The query-covariance weighting of key error is exactly the paper's starting identity, Eq. (1), lines 269–275:
  > "∥QK⊤ − QK̂⊤∥²_F = tr((K − K̂)Q⊤Q(K − K̂)⊤), (1) which is controlled by the query covariance Q⊤Q rather than by K⊤K alone."
  Appendix A.3 (line 2253): "This identity shows that key errors are weighted by the query covariance Q⊤Q."
- But OSCAR does **not** replace Hadamard; it composes with it. Eq. (3), lines 464–468: "RK = UQ HHad Pbr, RV = US HHad Pbr. Here Rk and Rv are the covariance base rotations, while HHad further improves quantization geometry by redistributing channel energy, and Pbr interleaves large- and small-variance channels". Lemma 1 (App. A.4, line 2341) shows H_Had exactly equalizes the diagonal of Λ_Q after the U_Q step.
- Theory: Theorem 1 (line 479) proves optimality of U_Q / U_S only for "frozen-error surrogate objectives" L̃_K(R_k) = tr(R_k⊤ C_Q R_k E_K) under the assumption that "the frozen (independent of input and rotation) residual covariances are diagonal in the ambient basis" (lines 479–495). This is a strong, acknowledged surrogate assumption, not a result about the actual quantizer.
- Empirics supporting "beats Hadamard": Table 2 (lines 593–700), 5 seeds, 32K generation, five tasks (GPQA, HumanEval, LCB v6, AIME25, MATH500), BPE 2.25 vs 2.28:
  - Qwen3-8B: BF16 mean 70.84; QuaRot-INT2 10.14 (drop −60.70); Naive INT2 0.00; OSCAR 69.416 (drop −1.42).
  - Qwen3-4B-Thinking-2507: BF16 75.64; QuaRot-INT2 1.40 (−74.24); OSCAR 71.864 (−3.78).
  - Qwen3-32B: BF16 74.19; QuaRot-INT2 7.90; OSCAR 74.17.
  - Text (line 787–788): "OSCAR reduces the BF16 gap to 3.78 and 1.42 points, respectively, while TurboQuant drops 43.90 and 13.96 points and rotation-only INT2 baselines largely collapse."
  - Table 4 (Qwen3-8B, 3 runs, same INT2 budget with sink/recent, lines 1096–1103): Full OSCAR U·HHad·Pbr 55.89±1.27; K⊤K/V⊤V PCA target 38.72±1.27; w/o Pbr 52.86±1.62; w/o HHad (only U·Pbr) 51.35±1.91; w/o U (QuaRot + Pbr) 40.74±3.44; no rotation 18.52±3.29. So the covariance eigenbasis alone beats Hadamard alone (51.35 vs 40.74), the raw K⊤K target is *worse* than Hadamard, and the combination is best. The paper's own framing (line 1041): "which covariance matrix one diagonalizes matters more than whether one diagonalizes at all."
  - Note Table 4 means (55.89) are far below Table 2 OSCAR (69.416) for the same model; the ablation is a different configuration. Quote them with their table.
- Nuance for the MLA idea's "structural difference" argument: OSCAR already head-aggregates the query covariance for GQA — "We use the production importance metric — Q-side query covariance, averaged within each GQA group of 4 query heads and then across all 8 kv-heads" (App., line 2456). So "M defined as a head sum" is not unique to MLA; the report should instead stress that in MLA the latent is shared by *all* heads and is pre-RoPE/up-projected, which changes which covariance is the right target.
- Nuance for "변환을 가중치에 접을 수 있다는 점은 MHA의 online Hadamard(QuaRot)와 달리 디코드 비용이 0이다": OSCAR already absorbs the *value* rotation into weights — "OSCAR then optimize value rotation by absorbing RV into the model's projection weights, achieving compute saving and latency reduction" (line 541–542) — while the key rotation remains an online fused op ("the fused online rotation does not introduce an ..." line 1634). So the zero-decode-cost advantage over MHA is specific to the key/latent side (where RoPE blocks folding in MHA).
- Value side is *not* query covariance: it is the score-weighted covariance C_S = (1/N) V⊤S⊤SV (line 455), "heuristically defined".

Claim B (heteroscedastic softmax-bias idea): "리스크: D'Alberto([2605.08114])와 OSCAR([2605.17757])가 인접하므로 '이분산 편향과 보정'을 전면에 세워 차별화해야 한다."

**Adjacency is weaker than stated.** OSCAR's analysis is entirely pre-softmax Frobenius logit distortion (Eq. 1, App. A.7 ℓ_K(R_k) = Σ_i Σ_j |q_i (k_j − k̂_j)⊤|²) and post-softmax output distortion ∥SV − SV̂∥²_F with S treated as fixed. Grep for "bias", "Jensen", "second-order", "Taylor", "softmax ... error" finds no analytical treatment of the softmax nonlinearity, no token-wise variance heterogeneity, and no bias-correction term; KL-to-FP16 attention drift is measured empirically only (Fig. 3, "Qwen3-8B OSCAR vs QuaRot vs Clip-only KL drift", line 913). OSCAR is adjacent on *where* key error is weighted (Q⊤Q), not on *softmax-induced systematic bias*. The report's differentiation plan is correct; it can state the gap more concretely: "OSCAR minimizes a Q⊤Q-weighted pre-softmax Frobenius surrogate with a frozen-residual assumption; it has no expression for the post-softmax expected probability shift."

### Other numbers (abstract / Sec. 5–6)
- "reduces KV-cache memory by approximately 8×, improves throughput by up to 7× at large batch sizes under the same memory budget, and accelerates batch-size-1 decoding by up to 3× over BF16".
- RULER-NIAH 4k–128k: "On Qwen3-4B-Thinking-2507, QuaRot-INT2 is already near zero at short contexts, whereas OSCAR stays close to BF16 through 16k and retains non-trivial retrieval at 64k and 128k." (lines 799–801)
- Typical clip thresholds c_K ≈ 0.96, c_V ≈ 0.92 (line 770); sink/recent knee at (64, 256) (Table 5).
- Baseline caveat: TurboQuant run "with mixed precision disabled" (line 768) — the TurboQuant drops are therefore not its default configuration.
