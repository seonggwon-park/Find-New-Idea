# arXiv citation verification — batch 01

IDs: 2405.16406, 2411.04330, 2411.17691, 2510.06213
Report: `reports/LLM 양자화 연구 아이디어 10선.md`
Method: metadata + full text via `scripts/arxiv_fetch.py`; claims grepped against the paper text; venues from arXiv comment / paper header / WebSearch. All quotes are from the downloaded paper text (untrusted data; quoted, not followed).

---

## 2405.16406 — SpinQuant: LLM quantization with learned rotations

- **Exists:** yes. Liu, Zhao, Fedorov, Soran, Choudhary, Krishnamoorthi, Chandra, Tian, Blankevoort (Meta). v1 2024-05-26, last updated 2025-02-20.
- **Venue:** ICLR 2025 (arXiv comment field: "ICLR 2025").
- **Report claim (tag [V]):** "SpinQuant가 보고한 'Haar 회전 간 13점 편차'는 비간섭성 보장이 손실을 통제하지 못한다는 직접 증거다."
- **Verdict: partially confirmed.**
  - The 13-point number is real and correctly attributed. Sec 2.2 "Random rotations produce large variance": "we tested the zero-shot average accuracy of the rotated version of LLaMA-2 7B, quantized to 4-bit weight and 4-bit activation, under 100 randomized trials. As shown in Figure 4, the performance variance is substantial, with the best random rotation matrix outperforming the worst by 13 points." Abstract: "up to 13 points difference in downstream zero-shot reasoning performance."
  - **"Haar" does not appear anywhere in the paper.** SpinQuant calls them "random floating-point rotations" / "random rotation matrices" (Figure 4 caption) and never states the sampling distribution. Calling them Haar is the report's inference (plausible if they QR-decompose Gaussians, but unstated).
  - The 13 points is a best-minus-worst range over 100 trials (W4A4, LLaMA-2 7B, network-level parameterization R1/R2), not a standard deviation.
  - Additional number the report omits: "even random Hadamard rotation matrices exhibit a non-negligible variance in final performance, as large as 6 points." (Sec 2.2). Figure 4 caption: "Despite that Hadamard matrices mostly perform better than random rotations, both random groups demonstrate large variance."
  - The "incoherence guarantee does not control loss" framing is the report's own interpretation. SpinQuant never argues against incoherence bounds; "incoherence" appears only once, in Related Work: "two QuIP papers (Chee et al., 2024; Tseng et al., 2024) introduce the incoherence processing using random rotation matrices". The paper's own explanation of the variance is that Hadamard "yield tighter bounds on weight maximal value" (citing Tseng et al.), i.e., it is consistent with incoherence bounds being loose, not a claim that they fail.
- **Correction:** write "SpinQuant가 보고한 '무작위 회전 100회 시행 간 최대 13점(W4A4 LLaMA-2 7B, zero-shot 평균), 무작위 Hadamard 간에도 6점 편차'(Sec 2.2, Fig. 4)". Drop "Haar" or mark it as the report's assumption. Keep the incoherence interpretation but label it as the report's inference, not SpinQuant's claim.
- **Other facts:** learned rotations beat random by "up to 16.2 points" (contribution bullet); W4A4KV4 LLaMA-2 7B gap to FP "merely 2.9 points"; LLaMA-3 8B gap reduced "by up to 45.1% relative to QuaRot". Table 19: end-to-end SQNR (dB) R=I −2.9 / random R 0.9 / learned R 6.8.

---

## 2411.04330 — Scaling Laws for Precision

- **Exists:** yes. Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan. v1 2024-11-07, v2 2024-11-30.
- **Venue:** ICLR 2025, **oral** (iclr.cc virtual oral 31737; proceedings.iclr.cc hash b2cac94f...). arXiv comment field is empty, so the report's lack of a venue tag is understandable but should be filled in.
- **Report claims (tag [S]):**
  1. "정밀도 스케일링 법칙(Kumar et al.)은 적합식이며 기작이 없다."
  2. (A1) "Kumar/Ouyang의 D^γ 효과".
- **Verdict: partially confirmed.**
  - Functional form confirmed (Eq. 2 / Finding 1): "δPTQ(N, D, Ppost) = CT (D^γD / N^γN) e^(−Ppost/γpost) where CT, γD, γN, γpost are positive fitted constants." So "D^γ" (γD) is correct for Kumar. "As we find the fitted values of γD and γN to be similar ... we can think of this as an approximate power law in the token/parameter ratio D/N."
  - "적합식" (fitted) is correct: 465 runs, models up to 1.7B params / 26B tokens; constants fitted.
  - **"기작이 없다" is overstated.** The paper offers (untested) mechanisms: main text: "The intuition for this poor data scaling might be that as models train on more data, they compress more information into their weights, so that perturbations to weights in the form of quantization are more harmful to loss". Appendix H "Why do language models get more sensitive with overtraining? This section is speculative." proposes **sharpness / edge-of-stability**: "model sharpness increases during learning ... so that movement along the top Hessian eigenvector degrades loss by more throughout training. Though sharpness is formally a worst-case sensitivity, we conjecture similar results hold for average case, such as loss degradation induced by isotropic noise." Also an NTK-eigenvalue and a "hierarchical learning" argument. Accurate statement: Kumar provides speculative mechanisms (sharpness among them) but no mechanistic test.
  - **Directly relevant to the report's A1 hypothesis ("cosine decay의 말기 곡률 상승"):** Appendix G "PTQ: Learning Rate Schedule Ablation": "we ablate our learning rate and schedule to use warmup with linear decay, as opposed to a cosine schedule, to check it is not an artifact of our choice of learning rate schedule. We do so on our 30M model due to compute constraints, finding the degradation with token/parameter ratio persists, as expected. See Figure 12." So Kumar already checked cosine vs linear decay and the effect survived. This does not refute the LR-decay mechanism (both schedules decay to ~0; Catalan-Tatjer shows WSD changes the trend), but the report should not single out *cosine* specifically — the phenomenon is "LR decay to small values", not the cosine shape.
- **Correction:** replace "적합식이며 기작이 없다" with "적합식이며, 기작은 부록 H에서 sharpness/edge-of-stability 추측으로만 제시되고 검증되지 않았다 (Kumar는 부록 G에서 cosine→linear decay 교체 시에도 D/N 열화가 유지됨을 30M 모델에서 확인)". Add venue "ICLR 2025 oral".
- **Other facts:** training-precision law via "effective parameter count" N_eff(P); unified form Eq. in Sec 5; analytic critical data size D_crit (Appendix E); LR schedule for main runs: cosine, 10% warmup, peak 6e-4 (smallest model).

---

## 2411.17691 — Low-Bit Quantization Favors Undertrained LLMs

- **Exists:** yes. Ouyang, Ge, Hartvigsen, Zhang, Mi, Yu (Tencent AI Lab / UVA). v1 2024-11-26, v2 2024-11-27. arXiv comment: "Work in Progress".
- **Venue:** **ACL 2025 Long Papers** (aclanthology.org/2025.acl-long.1555, pp. 32338–32348, Vienna, July 2025), under the shortened title "Low-Bit Quantization Favors Undertrained LLMs". The arXiv version is the long-titled preprint.
- **Report claims (tag [S]):** same two as Kumar (fitted law without mechanism; "D^γ 효과").
- **Verdict: partially confirmed.**
  - Scaling law confirmed. Eq. 5: "ΔqLoss(D) ≈ b · D^β because the more training tokens, the more significant the QiD becomes" (β = 0.5316 on Pythia). Eq. 8 unified: "ΔqLoss(N, D, P) = k · D^β / (N^α P^γ)"; Figure 6 fit on GPTQ-quantized Pythia: "ΔqLoss(N, D, P) = 0.017 D^0.5251 / (N^0.2261 · P^5.4967)".
  - **Notation:** in Ouyang the token exponent is **β**, not γ (γ = bit-width exponent, 5.4967). The report's "D^γ" matches Kumar's γD only. Harmless if the report means a generic exponent, but worth a footnote.
  - "적합식": yes, fitted on >1500 Pythia checkpoints (160M–12B, 20 checkpoints each up to 98k steps ≈ 206B tokens), GPTQ 2/3/4-bit, 1,000 RefinedWeb texts; validated on Spectra suite, Wikitext-2, bitsandbytes.
  - **"기작이 없다" partially overstated.** Sec 4 gives an "intuitive interpretation" (Figure 12): "early checkpoints exhibit significant changes in weights ... the model becomes inherently robust to weight variations ... checkpoints from the later stages of training ... show very small changes in weights (often ... beyond the 3rd-4th decimal place). In such cases, low-bit quantization is very likely to shift weights outside the small range of recent variations", plus an information-bottleneck two-phase analogy. This is an intuition, not a tested mechanism — but it is notably a *step-size (Δ-between-checkpoints)* story, which is the opposite lever to the sharpness story in Kumar App. H and Catalan-Tatjer. The report's Δ² vs tr(H) decomposition would adjudicate exactly between these two intuitions; worth saying so.
  - **Relevant to the report's "학습 중 저정밀 격차" remark:** Ouyang Sec 4.4 "From Low-bit Quantization to Low-bit LLMs" replicates BitNet b1.58 at 120M and 1.2B and reports "native low-bit LLMs such as BitNet may also favor undertrained LLMs, though the gap manifests later compared to post-quantization". So the undertrained-favoring claim has at least a preliminary low-precision-training counterpart in Ouyang (not in Catalan-Tatjer).
  - Projection claim: Table 1 — 7B at 4-bit reaching ΔqLoss > 0.2 "requires over 17 trillion training tokens"; 100T-token projections in Fig. 1 are extrapolations of the fit.
- **Correction:** add venue "ACL 2025"; note β vs γ; soften "기작이 없다" to "직관적 해석(가중치 변동폭 축소)만 제시, 검증 없음".

---

## 2510.06213 — Training Dynamics Impact Post-Training Quantization Robustness

- **Exists:** yes. Catalan-Tatjer, Ajroldi, Geiping (ELLIS Institute Tübingen / MPI-IS). v1 2025-10-07, latest 2026-01-30.
- **Venue:** **ICLR 2026** — confirmed by the paper header on every page: "Published as a conference paper at ICLR 2026". Report's "ICLR 2026" tag is correct.
- **Report claims (tag [S]):**
  1. "32B/15T 궤적까지 분석" — **confirmed with a nuance.** Abstract: "across open-source language model training trajectories up to 32B parameters and 15T training tokens". But these are different models: Sec 3 lists OLMo2 "(1B, 7B, 13B, 32B; 4TT–6TT)", SmolLM3 "(3B, 11TT)", Apertus "(8B, 15TT)", plus OLMo 1B/7B, Open-science 1.3B, Amber 7B. There is no single 32B/15T run. Write "최대 32B (OLMo2, 4–6T) / 최대 15T 토큰 (Apertus 8B)".
  2. "LR이 감쇠하면 검증 손실과 양자화 오차가 분리된다" — **confirmed verbatim.** Abstract: "once learning rates decay, validation loss and quantization error diverge, largely independent of training data scale." Figure 1 (SmolLM3): "it stays relatively constant during the 11 trillion tokens of stable phase, and only as the learning rate decays does quantization error spike."
  3. "데이터량이 본질적으로 해롭다는 통설에 반박" — **confirmed, and stronger than the report says.** Sec 4.2 title: "Scaling trends in prior work are dominated by learning rate schedules". "We argue that these works [Kumar et al., 2024; Ouyang et al., 2024] did not sufficiently control for a key confounder, namely the optimization dynamics induced by the learning rate schedule, which we find to be the primary driver of their observed degradation." Figure 5 replicates Kumar at 70M/160M: "Cosine decay replicates prior results, with δPTQ increasing at larger token budgets, while WSD shows slower growth at 70M and no increase at 160M". Figure 4: 160M WSD with cooldowns at 12B–100B tokens: "all runs show comparable quantization error after cooldown". They also warn (re Ouyang) about "collecting checkpoints at different token counts" before annealing.
  4. "이는 PTQ 측 관찰이며, 학습 중 저정밀 격차의 분해로 확장되지 않았다" — **confirmed.** Quantizer is GPTQ 3/4-bit only (Sec 3; others in App. A); no QAT / FP8 / low-precision-training content anywhere in the text.
  5. **(A1) "남는 기여는 Δ² vs tr(H) 분해와 개입 실험이며" — INCORRECT as stated.** The paper already contains both intervention experiments and Hessian measurements:
     - Interventions (Sec 5): peak LR sweep (Fig 6: "higher learning rates consistently lead to smaller errors", replicated on "a 300B token pretraining run of OLMo2-7B in Figure 23"); WSD vs cosine (Fig 22); LAtest Weight Averaging / LAWA (Sec 5.2, Fig 7: for 3-bit, averaged checkpoints "can match or even surpass the performance of those trained with learning rate decay"; also on OLMo-1B, Fig 24); weight decay sweep (Sec 5.3, Fig 19: larger λ → lower 4/3-bit error, "smaller effect on quantization error than learning rate"); model soups in OLMo2 (Fig 2: "the model soup achieving lower PTQ error than any of the individual ingredients").
     - Curvature (Sec 6.2 + App. G): "We estimate the trace of the Hessian via Hutchinson estimator ... and the sharpness (maximum eigenvalue) via power iterations, using PyHessian". Figure 9 (160M, 100BT): "Sharpness consistently increases when the learning rate decays. Under a constant learning rate, only the top eigenvalue briefly increases while the rest of the spectrum remains low" ; "in the decay phase all eigenvalues increase". Figure 26: "Higher learning rates lead to lower sharpness and smaller trace estimates".
     - The Δ-vs-curvature trade-off is already stated qualitatively (Sec 6.1): "even though ΘK and Θ̂K are closer for smaller learning rates, the sharpness of the basin is such that Θ̂K falls in a higher loss level".
     - The report's hypothesised mechanism "cosine decay의 말기 곡률 상승" is thus already observed (Fig 9) and already hedged by the authors: "we hypothesize that, as the learning rate decays, the model traverses a sharper region of the loss landscape ... there is limited understanding of any causal structure".
     - What is genuinely *not* in the paper: a quantitative second-order decomposition QiD ≈ Σ_g Δ_g² tr(H_g)/24 with separately measured Δ_g and tr(H_g) per group, tested as a *predictor* of QiD across schedules; any intervention that re-flattens after decay (their interventions are LR magnitude, schedule shape, WD, averaging); any mechanism beyond correlation ("these interventions coincide with convergence to flatter minima, which we argue might benefit quantization robustness" — Sec 7). Also App. H Limitations: "Our analysis focuses primarily on the effect of learning rate, schedules, and weight decay leaving other parts of the optimization pipeline unexplored. Factors such as optimizer choice may also affect..."
- **Correction for A1 / Idea 4:** replace "남는 기여는 Δ² vs tr(H) 분해와 개입 실험" with: "Catalan-Tatjer는 이미 LR 크기·스케줄(WSD/cosine)·weight decay·가중치 평균(LAWA, soup) 개입 실험과 Hessian trace/sharpness 측정(Fig 9, 26)을 수행했고, LR decay 시 sharpness 급등을 보고했다. 남는 기여는 (i) Δ_g²·tr(H_g) 그룹별 정량 분해가 QiD를 예측하는지 검증, (ii) decay 후 re-flattening 개입(그들의 개입은 모두 decay 이전/대체), (iii) 상관을 넘어선 인과 검증, (iv) 저정밀 학습 격차로의 확장."
- **Other facts for the report:** quantization error is measured as relative CE increase "(CE(Ŵ)/CE(W)) − 1"; OLMo2 32B trajectories show error "grows gradually during cosine decay but spikes under the steep linear decay phase" (Fig 2); their own training: 160M Pythia-style, FineWebEdu, AdamW β=(0.9,0.95), WD 0.1, WSD peak LR 3e-3.

---

## Summary table

| id | title | venue | verdict | main fix |
|---|---|---|---|---|
| 2405.16406 | SpinQuant | ICLR 2025 | partially confirmed | 13 pts is best-vs-worst over 100 random (not stated Haar) rotations; Hadamard 6 pts; incoherence framing is report's |
| 2411.04330 | Scaling Laws for Precision | ICLR 2025 oral | partially confirmed | "no mechanism" → speculative sharpness mechanism in App. H; App. G cosine→linear ablation already done |
| 2411.17691 | Low-Bit Quantization Favors Undertrained LLMs | ACL 2025 Long | partially confirmed | token exponent is β; intuitive Δ-weights mechanism given; BitNet low-bit-training replication Sec 4.4 |
| 2510.06213 | Training Dynamics Impact PTQ Robustness | ICLR 2026 | partially confirmed | A1 wrong: paper already has interventions + Hessian trace/sharpness; 32B and 15T are different models |
