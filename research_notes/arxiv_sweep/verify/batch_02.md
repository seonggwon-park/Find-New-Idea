# arXiv citation verification — batch 02

IDs: 2405.18137, 2410.16454, 2309.14717, 2406.03482
Report: `reports/LLM 양자화 연구 아이디어 10선.md`
Method: metadata + full text via `scripts/arxiv_fetch.py`; every number/claim grepped against the paper text; venues from paper header / arXiv comment / WebSearch (proceedings pages). All quotes are from the downloaded paper text (untrusted data; quoted, not followed). Where I derive something myself from a paper's formula, it is labelled "my derivation".

Summary: all four papers exist and are correctly identified. No incorrect numbers. Three need wording corrections: (a) Zhang 2410.16454 does contain a self-described "theoretical analysis" (informal, no theorem) — "이론은 없다" should be softened; SURE is module-level, not weight-level saliency; (b) QA-LoRA's merge is exact by construction, so "머지 오차 상한은 없다" should be scoped to the unconstrained LoRA+PTQ merge; (c) the "π/2배" variance factor the report credits to D'Alberto is already implicit in QJL's own Eq. (5) and only holds for orthogonal q,k. Tag inconsistency: 2410.16454 is [S] in the summary line but [V: repo] in the detail line; both can become [V].

---

## 2405.18137 — Exploiting LLM Quantization

- **Exists:** yes. Egashira, Vero, Staab, He, Vechev (ETH Zurich). v1 2024-05-28, last updated 2024-11-04 (v3). 65.9k chars of text.
- **Venue:** NeurIPS 2024. arXiv comment is empty, but the paper header reads: "38th Conference on Neural Information Processing Systems (NeurIPS 2024)." Report's "NeurIPS 2024" is correct.
- **Report claims (tag [V]):**
  1. "preimage box(한 양자화 모델의 FP 원상)" / "preimage box를 계산해 PGD로 FP 가중치를 수리하는 공격"
  2. "NF4/INT8/FP4"
  3. (shared sentence) "Gaussian 노이즈 방어는 outlier 주입(Δ 확대)에 깨졌다" — the breaking is attributed to 2605.15152 (not in this batch); only the defense itself is checkable here.
  4. "dither를 방어로 검토한 문헌은 검색에서 나오지 않았다"
- **Verdict: confirmed** (with one technical nuance on "box").
  - Constraint set: Sec 3 step 2: "we now construct a set of interval constraints over the weights of Mfm, which define the set of all full-precision models that quantize to Qm." Eq. (1) gives per-weight bounds (s·(α_{j−1}+α_j)/2, s·(α_j+α_{j+1})/2) around the assigned alphabet symbol α_j. The paper never uses the word "box"; it says "interval constraints" and "quantization-region constraints". A product of intervals is a box, so the report's term is fair.
  - **Nuance:** the box is obtained only after freezing the block's scale element: "To ensure that the scale s is preserved, we constrain wK to stay fixed throughout step 3." The true preimage of Q_m (letting the max-magnitude element move) is not a box; Egashira's set is a conservative inner approximation. Multi-method: "the adversary can compute the interval constraints for each method and use the intersection as the final constraint."
  - PGD: step 3: "we optimize Lr with projected gradient descent (PGD) to project the weights of M_fb s.t. they satisfy our constraints from step 2."
  - Methods: "LLM.int8() [8], NF4 [9], and FP4, which all rely on a scaling operation" (Sec 3). "INT8" in the report means LLM.int8() specifically (with its mixed-precision outlier decomposition, Appendix). Appendix: constraints "for LLM.int8() can [be] computed in < 1 minute. However, for nf4 and fp4, the process takes approximately 30 minutes".
  - Gaussian-noise defense (Sec 4 "Noise Defense", Table 6, Phi-2, LLM.int8(), vulnerable-code scenario): σ=0: code security FP32 98.0 vs Int8 18.5; σ=1e-4: 97.9 vs 32.6; σ=1e-3: 98.4 vs 97.5 (HumanEval 48.0/47.8, TruthfulQA 40.4/39.7); σ=1e-2: 99.8 vs 98.8 but HumanEval collapses to 9.8/13.8. Paper: "At σ = 10−3 adding noise proves to be an effective defense against the attack, removing the security contrast while maintaining utility." Caveat in the paper itself: "potential consequences beyond benchmark performance of the noise addition remain unclear and have to be thoroughly investigated before noise-based defenses can be adopted ... We leave the study of this problem as a future work item". Defense is tested on one model, one scenario, one quantizer.
  - "dither": 0 hits in the paper. Consistent with the report.
- **New relevant facts for the report's Δ-widening framing:** Sec 4 "Constraint Width" already identifies interval width as the attack's lever: "Phi-2 contains a larger fraction of weights with higher magnitudes than StarCoder-1b. Due to the scaling parameter s being defined as max|w| across all investigated zero-shot quantization methods, this leads to almost 2× wider quantization intervals ... models with long-tailed weight distributions result in easier optimization problems for adversaries" (security contrast up to 80.1% on Phi-2 vs 56.3% on StarCoder-1b, Table 1 / Fig. 3). So the 2605.15152 "outlier injection" is a deliberate exploitation of a lever Egashira observed passively. Table 4 ablation: PGD-only (no quantization-aware regularizer) suffices, "18 minutes" overhead vs standard finetuning.
- **Correction:** none required. Optional precision: "interval(box) constraints with the block's scale element frozen — a conservative inner approximation of the true preimage" if the report's unified-theorem idea depends on the preimage being exactly a box.

---

## 2410.16454 — Catastrophic Failure of LLM Unlearning via Quantization

- **Exists:** yes. Zhang, Wang, Li, Wu, Tang, Liu, He, Yin, Wang. v1 2024-10-21, last updated 2025-03-21. Code: github.com/zzwjames/FailureLLMUnlearning.
- **Venue:** ICLR 2025 (arXiv comment "ICLR 2025"; header "Published as a conference paper at ICLR 2025"). Report's "ICLR 2025" is correct.
- **Report claims:** summary line tag [S]; detail line tag [V: repo].
  1. "효용 제약 unlearning 모델은 FP에서 잊은 지식 21%를 유지하는데 4-bit 후 83%로 복귀"
  2. "SURE(salient 가중치에 큰 LR)가 완화한다"
  3. "이론은 없다"
- **Verdict: partially confirmed** (numbers exact; "이론은 없다" overstated; SURE granularity slightly off).
  - 21%/83%: verbatim in abstract and Sec 4.2: "for unlearning methods with utility constraints, the unlearned model retains an average of 21% of the intended forgotten knowledge in full precision, which significantly increases to 83% after 4-bit quantization." Setting: MUSE benchmark (BOOKS/NEWS), methods GA/GA_GDR/GA_KLR/NPO/NPO_GDR/NPO_KLR; Q1 uses 4-bit round-to-nearest ("we apply 4-bit quantization using round-to-nearest (RTN)"), with AWQ and GPTQ 4-bit also in the tables (rows "GA_GDR + Quan.(AWQ)", "... (GPTQ)"). 8-bit: "models with 8-bit quantization perform similarly to full-precision models". Unconstrained GA is an exception the paper calls misleading: "this is misleading as it results from a complete loss of model utility due to a lack of constraints." Additional benchmark RWKU in Appendix F.
  - SURE: "We name this approach Saliency-Based Unlearning with a Large Learning Rate (SURE)." (Sec 6.1). **Granularity:** saliency is at module level, not weight level: "constructing module-level saliency maps", "we update only the salient modules identified by the module-level saliency map, leaving the rest of the network unchanged" (module = "specific attention head or feed-forward sub-layer"). LR grid [5e−5, 1e−4, 2e−4]; threshold γ as a percentile (90/95/99) of saliency scores (Tables 5, 6). Effect: "For quantized models, incorporating SURE significantly improves forgetting performance compared to original methods without SURE" (Table 3 BOOKS, Table 8 NEWS). Caveat in paper: for GA_GDR, SURE costs utility/fluency ("This verifies the concern of potential bias introduced by a large learning rate").
  - Theory: **no Theorem/Lemma/Proposition anywhere** (grep count 0). But the paper explicitly claims one: abstract "we provide a theoretical explanation for the observed phenomenon"; Sec 5 is titled as a theoretical analysis and argues from interval size: "a weight w within a quantization interval Ii is mapped to ... qi = i∆ ... Ii = [(i − 1/2)∆, (i + 1/2)∆)"; "∆int4 = max(|w|)/8 ... ∆int8 = max(|w|)/128"; "To differentiate the quantized weights of the original model f from those of the unlearned model funlearn, the weight changes in funlearn must exceed the quantization step size ∆"; "effective unlearning methods that preserve utility typically have minimal weight changes, resulting in ftarget and funlearn being highly similar, i.e., Q(funlearn) ≈ Q(ftarget)". Worked example with max|w|=200: int-4 interval [−12.5, 12.5) vs int-8 step 0.78125 (Fig. 2). This is exactly the "preimage box" argument stated informally, with no bound, no probability statement, no dependence on group size or on the fraction of weights near interval boundaries.
- **Correction:**
  - Replace "이론은 없다" with "형식적 정리는 없다; Sec 5의 '이론적 분석'은 Δ_int4 = max|w|/8 ≫ Δ_int8 = max|w|/128 라는 간격 크기 비교에 기반한 비형식적 설명이며 상한·확률 진술이 없다." The report's claim that nobody wrote the unified preimage theorem survives, but Zhang Sec 5 should be cited as the informal precursor.
  - "salient 가중치" → "salient 모듈(어텐션 헤드/FFN 서브레이어)".
  - Make the tag consistent: summary line [S] → [V] (text verified here).
- **Other facts:** RTN vs AWQ vs GPTQ all show the 4-bit failure ("regardless of the quantization technique used"); 2-bit excluded from study ("We exclude 2-bit precision models").

---

## 2309.14717 — QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

- **Exists:** yes. Xu, Xie, Gu, Chen, Chang, Zhang, Chen, Zhang, Tian (Huawei). v1 2023-09-26, v2 2023-10-09. Code: github.com/yuhuixu1993/qa-lora.
- **Venue:** ICLR 2024 (proceedings.iclr.cc/paper_files/paper/2024/hash/e6c2e85db1f1039177c4495ccd399ac4; iclr.cc/virtual/2024/poster/18462; mlanthology "International Conference on Learning Representations, 2024"). arXiv comment is only "16 pages" and the arXiv v2 text predates acceptance, so no venue header in the downloaded text. Report's "ICLR 2024" is correct.
- **Report claims:** summary tag [V]; detail tag [V: repo].
  1. "LoRA 머지 실패"
  2. "머지 문제의 유일한 닫힌 해법은 QA-LoRA의 'zero-point에 접기'이며 어댑터를 그룹 상수로 제약한다"
  3. "머지 오차 상한은 없다"
  4. "'델타가 4-bit 그룹 스텝 아래라 66–98%가 지워진다'는 2026 시스템 논문 보고 ([2609.18063] [U])" — not in this paper; not checkable here.
- **Verdict: confirmed** for 1, 2; **partially confirmed** for 3 (true but needs scoping).
  - Merge failure (empirical): Sec 3.2: "After fine-tuning, the side weights s·AB must be added back to W̃, making the final weights W′ in FP16 again. Indeed, one can perform post-training quantization (PTQ) upon W′, but this strategy can cause a significant loss in accuracy especially when the bit width is low." Fig. 1 / Table 1 (QLoRA w/ GPTQ vs QA-LoRA); Sec 4.2: "QA-LoRA reports a remarkable accuracy gain of 15.0% over the 2-bit post-quantized QLoRA."
  - Zero-point folding: Algorithm 1: `def merge_with_quantization(beta, lora_A, lora_B): beta_new = beta - s * (lora_B @ lora_A).transpose(0,1) / alpha`. The derivation (Sec 3.3): the delta c_{i,j} = s·Σ_k a_{i,k} b_{k,j} must be absorbable into the zero factor; "we must guarantee that for any j, all possible values of c_{i,j} form an arithmetic set with the common difference being α_j. This is intractable in continuous and gradient-based optimization unless we ask that c_{i,j} is a constant".
  - Group-constant adapter: "We partition each column of W into L groups ... we only require the row vectors of A within the same group to have the same value. In our implementation, this is achieved by doing summation within each group of the input vector, x. This parameter-free operation reduces the dimension of x from Din to L, hence we can set A to be a L × Dint matrix". Pseudocode: `QA = nn.AvgPool1d(D_in//L)` then `QA(x)*(D_in//L)`. Main experiments: "group-wise asymmetric quantization (with a group size of 32)", min-max INT4 (Eq. 1: α = (max−min)/(2^N−1), β = min). Degrees of freedom: quantization Dout → L×Dout, adaptation Din×Dint+Dint×Dout → L×Dint+Dint×Dout.
  - Error bound: no theorem, lemma, or bound anywhere (grep "bound", "guarantee", "error" yields only prose). **But** the merge in QA-LoRA is exact by construction — the group-constant delta is absorbed into the FP16 zero factor β_{l,j} with zero rounding error — so for QA-LoRA no error bound is needed. What lacks a bound is the *unconstrained* LoRA → PTQ merge (QLoRA + GPTQ), which the paper only measures.
  - "유일한 닫힌 해법": a literature claim, not checkable against this paper; QA-LoRA presents itself as the first ("To the best of our knowledge, the most related work is QLoRA").
- **Correction:** rewrite "머지 오차 상한은 없다" as "QA-LoRA의 머지는 그룹 상수를 β에 흡수하므로 구성상 정확하다(오차 0); 비제약 LoRA+PTQ 머지의 오차 상한은 문헌에 없고 QA-LoRA는 이를 경험적으로만 보인다(2-bit에서 15.0%p 격차, Sec 4.2)." Tag [V: repo] → [V].
- **New relevant fact for the preimage/lattice framing:** footnote 1 states the exact mergeability condition: "For any j, there exists a new zero factor β′_j and a set of integers ĉ_{i,j} so that c_{i,j} = α_j × ĉ_{i,j} + β′_j. Additionally, the difference between the minimum and maximum of ŵ_{i,j} + ĉ_{i,j} is not greater than 2^B − 1 so that the summed weights can still be quantized into B-bit integers." I.e., the delta must lie on the shifted lattice α_j·Z + β′_j within the code range — a lattice/preimage condition that QA-LoRA states and sidesteps (by forcing constancy) but never analyses when violated. This is the closest thing in the three "preimage" papers to the report's proposed unified statement.
- **Other facts:** Table 5 group-size ablation (smaller group / larger L → higher accuracy, more storage); inference "more than 50% faster than QLoRA" because weights stay INT4; evaluated on LLaMA 7B–65B and LLaMA2, MMLU 0/5-shot, Alpaca and FLAN v2.

---

## 2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead

- **Exists:** yes. Zandieh (independent), Daliri (NYU), Han (Adobe Research on arXiv; KAIST on AAAI page). v1 2024-06-05, v2 2024-07-18. Code: github.com/amirzandieh/QJL.
- **Venue:** AAAI 2025 (ojs.aaai.org/index.php/AAAI/article/view/34773; Proceedings of the AAAI Conference on Artificial Intelligence vol. 39, pp. 25805–25813; mlanthology zandieh2025aaai-qjl). arXiv comment is only "13 pages". Report's "AAAI 2025" is correct.
- **Report claims (tag [V]):**
  1. "QJL은 '비편향 내적 추정량, 최소 왜곡'을 증명했지만"
  2. "vLLM 구현은 QJL을 제외했고, 그 사유를 '5개 이상 독립 그룹의 합의: softmax가 분산을 증폭'으로 적었다 [V]" — about vLLM, not this paper.
  3. (context) D'Alberto 2605.08114: "QJL이 K의 내적 분산을 π/2배 키우고 softmax가 Jensen으로 증폭한다" — the report presents this as D'Alberto's claim; relevant because the π/2 is derivable from QJL itself.
  4. (context) QJL has no analysis of "토큰 간 분산 이질성이 만드는 체계적 편향, 보정식, K-vs-V 할당 규칙".
- **Verdict: confirmed** (1, 4), with a precision note on what "증명" covers; (2) not checkable against the paper.
  - Estimator (Def. 3.1, Eq. 3–4): H_S(k) := sign(Sk); Prod_QJL(q,k) := (√(π/2)/m)·‖k‖₂·⟨Sq, H_S(k)⟩, S Gaussian m×d. Asymmetry is essential: "if we applied QJL to both vectors q and k ... we would obtain an unbiased estimator for the angle ... one needs to apply the cosine function on top of the angle estimator, which results in a biased estimation. Thus, to achieve an unbiased inner product estimator, it is necessary to asymmetrically apply quantization".
  - **Lemma 3.2** (unbiased): "E_S[Prod_QJL(q,k)] = ⟨q,k⟩". Proved.
  - **Lemma 3.5** (distortion): "if ... m ≥ (4/3)·(1+ε)/ε²·log(2/δ), then Pr_S[|Prod_QJL(q,k) − ⟨q,k⟩| > ε‖q‖₂‖k‖₂] ≤ δ." Proved via moments (Eq. 5) + Bernstein.
  - **Theorem 3.6** (attention scores): "If the key embeddings have bounded norm max_i‖k_i‖₂ ≤ r and m ≥ 2r²ε⁻² log n, then for any query ... ‖q_n‖₂ ≤ r ... with probability 1 − 1/poly(n) simultaneously for all i: |Ŝcore(i) − Score(i)| ≤ 3ε·Score(i)". Proof: per-score additive error ≤ ε, union bound, softmax ⇒ e^{±2ε} ⊂ (1±3ε). So the softmax step is handled by a uniform worst-case bound only; nothing on variance propagation, on bias of E[softmax], or on heterogeneity across tokens. Confirms the report's "no systematic bias / correction / K-vs-V rule" statement.
  - "최소 왜곡" is the abstract's "minimal distortion". The paper's stronger prose claim — "the distortion bound in Lemma 3.5 has remarkably small constants, even smaller than those of the original unquantized JL transform. This indicates that quantizing one of the vectors to just a single sign bit does not result in any loss of accuracy." — is **not** established by the proof (a Bernstein-constant comparison is not a variance comparison; see next item). The report should not inherit "no loss of accuracy".
  - **The π/2 factor is already in QJL (my derivation from the paper's Eq. 5).** Proof of Lemma 3.5 writes the per-row terms z_i := √(π/2)·‖k‖₂·(s_iᵀq)·sign(s_iᵀk) and gives "E|z_i|^ℓ = (√π·‖k‖₂‖q‖₂)^ℓ · Γ((ℓ+1)/2)/√π" (Eq. 5). At ℓ=2: E[z_i²] = (π/2)‖q‖²‖k‖². Hence Var[Prod_QJL] = [(π/2)‖q‖²‖k‖² − ⟨q,k⟩²]/m, versus the unquantized JL estimator (1/m)Σ(s_iᵀq)(s_iᵀk) with Var = [‖q‖²‖k‖² + ⟨q,k⟩²]/m (Isserlis). Ratio with cos∠(q,k)=ρ: [(π/2) − ρ²]/[1 + ρ²]. This equals π/2 ≈ 1.571 only at ρ=0; it falls below 1 for ρ² > (π/2 − 1)/2 ≈ 0.285 (|ρ| > 0.53), and is ≈ 0.29 at ρ=1. So "QJL이 분산을 π/2배 키운다" is the orthogonal-key limit, and the keys that dominate softmax (most aligned with q) have *lower* QJL variance than plain JL. This cuts both ways for the report's hypothesis: the variance inflation is concentrated on low-score keys, which is where the softmax denominator (not the numerator) is affected — relevant to any "systematic bias" derivation.
  - vLLM claim: docs.vllm.ai is unreachable from this sandbox and the vLLM repo is not attached, so no verbatim quote. WebSearch's summary of the vLLM TurboQuant docs page is consistent with the report ("QJL is intentionally omitted ... community consensus from 5+ independent groups ... degrades attention quality by amplifying variance through softmax"), and the same search surfaced 2605.08114 as the only analysis. Keep [V] on the strength of the earlier session's check, but note it is a doc-comment assertion, not a published benchmark.
- **Correction:**
  - Keep "비편향 내적 추정량(Lemma 3.2), 왜곡 상한(Lemma 3.5; Theorem 3.6: ‖q‖,‖k‖ ≤ r, m ≥ 2r²ε⁻²log n 하에 점수당 1±3ε)". Do not import "no loss of accuracy".
  - Reattribute the π/2: "π/2 인자는 QJL 자신의 Eq. (5) 2차 모멘트에서 직접 나오며(직교 q,k 극한), 상관 ρ가 커지면 비율 [(π/2) − ρ²]/[1 + ρ²]로 감소해 |ρ| > 0.53에서는 1 미만이다. D'Alberto는 이를 softmax 맥락에서 정성적으로 재진술한 것이다."
- **Other facts:** keys: QJL + per-token norm ν_i; values: standard per-token quantization (Sec 3.2); outliers: "the outliers show a persistent pattern in specific fixed coordinates of the key embeddings ... we quantize the outliers using an independent instance of our QJL quantizer but with a lower compression rate"; "orthogonalizing the rows of the JL matrix S" helps empirically; claimed "3 bits" KV with ">5×" memory reduction (abstract). Theorem 3.6's required m grows with r², so the bound is loose for keys with large norms unless outliers are split off — the paper's own motivation for the outlier channel.

---

## Tag/venue actions for the report

| id | report venue | verified venue | tag now | tag should be |
|---|---|---|---|---|
| 2405.18137 | NeurIPS 2024 | NeurIPS 2024 (header) | [V] | [V] |
| 2410.16454 | ICLR 2025 | ICLR 2025 (comment + header) | [S] / [V: repo] | [V] (both lines) |
| 2309.14717 | ICLR 2024 | ICLR 2024 (proceedings.iclr.cc) | [V] / [V: repo] | [V] |
| 2406.03482 | AAAI 2025 | AAAI 2025 (ojs.aaai.org 34773) | [V] | [V] |
