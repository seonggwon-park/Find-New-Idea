# Citation verification — batch 10

IDs: 2607.04422, 2606.15007, 2103.00065, 2207.14484. All four exist on arXiv. Texts fetched to
`/tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers/<id>/<id>.txt`.

---

## 2607.04422 — Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention

- Authors: Siyu Ding, Mingchuan Ma, Jiabo Tong, Xingrun Xing, Ziming Wang, Guoqi Li (CASIA)
- Published 2026-07-05, updated 2026-08-07 (v2: "Fix experiment bugs and some statement, update ... hardware efficiency result on 5090")
- Venue: preprint (text header says "Preprint."; no journal_ref; web search found no acceptance)
- Report tag: [S]

**Report claim:** listed among 2026 groups that "each claim a different bottleneck" for FP4 pretraining — "Full-Stack FP4 = unquantized components (미양자화 구성요소)".

**Verdict: partially_confirmed.**

What the paper actually says:
- It identifies the still-high-precision modules as the *coverage gap* in prior NVFP4 work, not as the cause of the FP4 loss/stability gap: "most NVFP4 recipes retain higher-precision optimizer states, optimizer computation, and attention" (Intro); abstract: "leaving persistent optimizer states, optimizer computation, and low-precision attention forward--backward paths less explored".
- The motivation for quantizing those modules is memory/throughput: "2.50--2.83x Root speedups over optimized BF16 and 37.9--42.5% lower AdamW peak memory" (abstract; four-block RTX 5090 measurements, "not the throughput or memory of the complete distributed 3B run", §4.1).
- For the *loss* bottleneck it defers to FP4-All-the-Way: "Projection error remains relevant as optimization signals weaken later in training: FP4-All-the-Way [2] observes that gradient signals decrease while quantization noise remains approximately constant" (Intro). Its own loss-gap fix is LoRA-SVD on the *projections* (a quantized component), keeping a BF16 subspace: "reducing the linear-only loss gap from 1.40% to 0.61%".
- Table 3 (1B/20B tokens, AdamW): BF16 2.6415; BF16+LoRA 2.6351 (−0.24%); NVFP4 4/6 2.6785 (+1.40%); 4/6+plain LoRA 2.6711 (+1.12%); 4/6+LoRA-SVD 2.6575 (+0.61%); 4/6+Q-AdamW 2.6674 (+0.98%); 4/6+Q-Attn 2.6539 (+0.47%); Full-Stack w/o Root 2.6563 (+0.56%). Root+AdamW: BF16+Root 2.4819; Full-Stack 2.4973 (+0.62%).
- 3B/64B tokens (§4.3, Fig. 5): BF16 2.267 vs Full-Stack FP4 2.286, "a 0.838% relative gap" — i.e. the full stack has a larger gap than the linear-only LoRA-SVD result (0.61%); quantizing the extra modules does not reduce the loss gap.
- Limitations (§Conclusion and Limitations): "Quality experiments cover at most 3B parameters and 64B tokens; they do not establish behavior at 7B+ scale or over longer training." "the present recipes and measurements are specific to NVFP4 and do not establish transfer to MXFP4." Also: "our experiments do not establish long-horizon behavior" (Intro).

**Correction:** Replace "Full-Stack FP4는 미양자화 구성요소" with something like "Full-Stack FP4는 병목을 주장하기보다 optimizer state/연산·attention까지 NVFP4 커버리지를 확장(메모리·속도 동기)하고, 손실 격차는 projection 오차로 보고 LoRA-SVD로 1.40%→0.61% 축소(1B/20B, Table 3); 3B/64B 전체 스택 격차는 0.838%(Fig. 5)". The paper does not run a controlled LR-sweep separating bottlenecks, so the report's follow-up sentence ("같은 LR 지점에서 이들을 분리한 실험이 없다") remains true for this paper.

---

## 2606.15007 — Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

- Authors: NVIDIA (several hundred authors)
- Published 2026-06-12 (v1 only)
- Venue: technical report / preprint (no journal_ref, no comment)
- Report tags: [?], [S]; the report's self-note says "존재·내용을 PDF로 확인하기 전까지 인용하지 말 것" — existence and content are now confirmed.

**Report claim 1:** "Nemotron 3 Ultra는 전 텐서 BF16 전환이 발산을 해결하지 못했다고 보고" (switching all tensors to BF16 did not resolve divergence).

**Verdict on claim 1: confirmed (verbatim).** §2.2 (NVFP4 pretraining): "Switching all tensors to BF16 did not resolve the training divergence discussed in section 2.7 (Figure 3, bottom)." Figure 3 caption: "Bottom: Longer training in BF16 from the 16T token checkpoint. Raw train loss shows similar pattern of divergence for both BF16 and NVFP4 models. Relative difference in train loss (%) converges toward zero during the training divergence. Switching all tensors to BF16 did not resolve the training divergence."

**Report claim 2:** "— NVFP4 논문과 모순" (contradicts the NVFP4 paper).

**Verdict on claim 2: not stated by the paper; weakly supported.** The paper presents itself as *consistent* with the NVFP4 paper: "As seen in prior work, switching all tensors to BF16 substantially recovers the high-precision loss, providing a proxy for high-precision training NVIDIA (2025f) [= Pretraining Large Language Models with NVFP4]." The divergence it reports is not attributed to NVFP4 GEMMs:
- §2.7 "Divergence 1: Local Gradient Accumulation Precision for Output Layer" (~8T tokens): "attributed to a reduction in local gradient accumulation precision for the output layer from FP32 to BF16 ... the MTP blocks' wgrad contribution to the shared output layer is essentially lost when using BF16, which has only 7 mantissa bits." Fixed by rollback + "full FP32 gradient reduction recipe".
- §2.7 "Divergence 2: Undetermined" (~16T tokens): "starting learning rate annealing ... immediately after rolling back to the 15T token checkpoint mitigates divergence"; "We eventually made the practical decision to cut the total pretraining token horizon down to 20T tokens." "While we could not find a smoking gun that caused this instability", two correlates: MoE expert imbalance (MaxVio of the first MoE layer rising from 4.8 to ≈12 by 12T tokens; "Although not causal by itself, MaxVio seems correlated with training instability") and residual-stream norm spiking ("early layer residual norms started spiking about 7.5T tokens into training, with drastic spiking happening around 11T tokens"; residual norms differ by "4 orders of magnitude" across depth for Ultra).
- The NVFP4 paper (2025f) claims a BF16 switch recovers the *loss gap*, not that it cures divergence; Nemotron 3 Ultra reproduces the loss-gap recovery. So "contradiction" is the report's inference, and the paper's own evidence says the divergence is BF16-run-reproducible, i.e. not an NVFP4 artifact.

**Correction:** Keep claim 1 verbatim and drop or soften "NVFP4 논문과 모순": e.g. "Nemotron 3 Ultra(§2.2, Fig. 3 bottom)는 16T 체크포인트에서 전 텐서 BF16 전환 후에도 동일 패턴의 발산이 재현됐다고 보고 — 즉 이 발산은 NVFP4 자체가 아니라 BF16 출력층 grad 누적(Div.1) 및 미규명 원인(Div.2, LR annealing으로 완화)에 기인; NVFP4 논문의 'BF16 전환으로 손실 격차 회복' 주장은 재확인됨(평균 격차 <0.4%)."

Other numbers worth having (§2.2): NVFP4-vs-BF16 relative train-loss gap "below 0.4% on average"; over first 5B tokens of BF16 branch: 0.27%, 0.28%, 0.25% from 5T/10T/16T checkpoints; after 74B tokens: 0.33%, 0.34% (5T, 10T) and 0.03% (16T). Embedding, output, latent, QKV/attention projections and MTP layers kept in higher precision "following NVIDIA (2025c, 2026)". Claims "the largest-scale demonstration of stable and accurate NVFP4 training to date" (550B-A55B, 20T tokens).

---

## 2103.00065 — Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability

- Authors: Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar
- Published 2021-02-26, updated 2022-11-23 (v3)
- Venue: ICLR 2021 (arXiv comment: "ICLR 2021")
- Report tag: [U: 고전, 미재확인]

**Report claim:** "Edge of stability: λ_max(H) ≈ 2/η (GD)".

**Verdict: confirmed.** Abstract: "the maximum eigenvalue of the training loss Hessian hovers just above the value 2/(step size)". §1: "the sharpness hovers right at, or just above, the value 2/η". Also progressive sharpening: "So long as the sharpness is less than 2/η, it tends to continually increase during training" (Fig. 3 caption).

Caveats for the report: the result is *empirical* ("We empirically demonstrate"), for *full-batch* gradient descent (vanilla GD; momentum variants have a different threshold), on small-scale CIFAR-10-type networks; the paper explicitly does not explain the mechanism ("We do not know why progressive sharpening occurs"). The 2/η threshold is the *quadratic* stability bound ("gradient descent on quadratic objectives is ... unstable if any eigenvalue exceeds 2/η"). The [U] tag can be upgraded to [V].

---

## 2207.14484 — Adaptive Gradient Methods at the Edge of Stability

- Authors: Jeremy M. Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E. Dahl, Justin Gilmer (Google)
- Published 2022-07-29, updated 2024-04-15 (v2: "corrects the formula for Adam's preconditioner in Eq 2")
- Venue: preprint; listed at NeurIPS 2023 Heavy Tails in ML workshop (mlanthology/neurips.cc virtual 2023/81576); Google Research page labels it "NeurIPS 2022 (to appear)" but no main-proceedings record was found. Cite as arXiv preprint.
- Report tag: [U]

**Report claim:** "적응형 최적화기는 전처리된 sharpness 기준" (adaptive optimizers use a preconditioned-sharpness criterion).

**Verdict: confirmed.** Abstract: "during full-batch training, the maximum eigenvalue of the preconditioned Hessian typically equilibrates at a certain numerical value — the stability threshold of a gradient descent algorithm. For Adam with step size η and β1 = 0.9, this stability threshold is 38/η." §: "the preconditioned sharpness λ1(P^{-1}H_t) rises until equilibrating at the stability threshold of 38/η". Stability thresholds listed: "2/η for GD(η), 2+2β/η for ..." (momentum variants, §3).

Caveats: empirical; full-batch and "sufficiently large batch" ("as the batch sizes decreases, the trajectory of the preconditioned sharpness shifts downwards", Fig. 6); the raw (un-preconditioned) sharpness "usually keeps rising at the AEoS (see Figure 8)" — adaptive methods "keep advancing into high-curvature regions, while adapting the preconditioner to compensate". The 38/η value is specific to β1 = 0.9 (EMA heavy-ball threshold with frozen preconditioner). The [U] tag can be upgraded to [V]; if the report wants a number, give "38/η (Adam, β1=0.9)".

---

## Summary

| id | exists | verdict | action |
|---|---|---|---|
| 2607.04422 | yes | partially_confirmed | reword: coverage extension, not a "bottleneck" claim; loss fix is LoRA-SVD on projections |
| 2606.15007 | yes | partially_confirmed | BF16-switch-did-not-fix-divergence is verbatim; drop "NVFP4 논문과 모순"; divergence not attributed to NVFP4 |
| 2103.00065 | yes | confirmed | upgrade [U]→[V]; note empirical/full-batch; ICLR 2021 |
| 2207.14484 | yes | confirmed | upgrade [U]→[V]; add 38/η (Adam, β1=0.9); preprint / NeurIPS 2023 workshop |
