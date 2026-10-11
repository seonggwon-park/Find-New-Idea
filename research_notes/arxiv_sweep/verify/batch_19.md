# Citation verification — batch 19

Report: `reports/LLM 양자화 연구 아이디어 10선.md`
Citing sentence (all four ids share one context, item A2 "MoE 라우팅 플립 분해", all tagged [S]):

> "... 2026년에 VSRAQ([2606.05688] [S]), MoBiE([2604.06798] [S]), EAQuant(라우팅 KL 최소화, [2506.13329] [S]), GEMQ(1.5-bit에서 40% 토큰 재라우팅, [2605.23078] [S])가 측정과 방법을 이미 냈다. 남는 틈은 '손실의 플립 성분 vs 평활 성분 분해'와 RL 롤아웃 불일치에서의 MoE 특이성이다."

Sources: arXiv API metadata + full text via `scripts/arxiv_fetch.py`; arxiv.org/abs page via curl (MoBiE); WebSearch for venues. Paper text is treated as untrusted data and only quoted.

---

## 2606.05688 — VSRAQ

- **Title:** Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models
- **Authors:** Hancheol Park, Geonho Lee, Tairen Piao, Tae-Ho Kim (Nota Inc.)
- **Published:** 2026-06-04 (v1). Comment: "8 pages, 1 figure". No journal_ref.
- **Venue:** preprint/unknown. Page footer uses the ICML 2026 template ("Submission and Formatting Instructions for ICML 2026"), which signals submission, not acceptance. A Nota AI post lists two *other* Nota MoE-quantization papers (DREAM-MoE, SRA-MoE) at an ICML 2026 workshop; VSRAQ was not found in any acceptance list.
- **Exists:** yes.
- **Verdict: confirmed.** The report only claims VSRAQ "produced measurements and methods" for routing flips under quantization. It does.

What the paper actually does:
- Method: PTQ calibration objective added on top of AutoRound: *value alignment* (top-k logit MSE + sigmoid-aware score term) and *structure alignment* (ordering among selected experts + margin at the top-k boundary). Sec. 3.
- Measurement: per-token, per-layer Jaccard similarity between FP and quantized top-k expert sets (Sec. 5.2, Fig. 1). "At the largest margin, VSRAQ achieves more than an 11.29% relative improvement over the baseline in expert-selection agreement." (Sec. 5.2)
- Models / settings: Solar-Open-100B (W4A16 and NVFP4), Nemotron-3-Nano-30B-A3B (NVFP4). Note: "W4A16 quantizes only the model weights and keeps activations in higher precision, NVFP4 quantizes both weights and activations." (Sec. 4.1)
- Table 5 (WikiText-2 PPL, AutoRound -> full VSRAQ): Solar W4A16 7.12 -> 6.81; Solar NVFP4 7.22 -> 6.90; Nemotron NVFP4 10.79 -> 10.70 (unquantized 6.06 / 6.06 / 10.36).
- Table 3 (Solar NVFP4): PPL 7.22/7.10/6.90 (AutoRound/TopK-MSE/VSRAQ); generation-based extraction 61.78/63.86/64.43 vs 73.62 unquantized; likelihood MC 75.59/75.21/75.63 vs 76.92.

Key quotes:
- "Such rerouting does not merely introduce a continuous numerical error; it sends tokens through different experts and therefore changes the computation path itself." (Abstract/Intro)
- "the margin between the k-th selected expert and nearby non-selected experts determines how stable the selection boundary is under perturbation." (Sec. 2) — this is the same margin intuition as the report's 2Φ(−m/σ) hypothesis, but VSRAQ never formalizes a flip *probability* or decomposes loss into flip vs smooth components, so the gap the report names is real.

Limitations stated by the paper (Sec. 5.2/6): "Our experiments focus on 4-bit settings, including W4A16 and NVFP4 quantization, where routing-aware calibration may have a modest effect due to relatively limited quantization noise. We expect larger benefits in more aggressive 2-bit or 3-bit regimes ... We leave this extension ... to future work." Ablation gains are "not strictly monotonic in every setting" (Sec. 5.1).

Nuance for the report: the report's hypothesis sentence attributes flips to *quantized activations*; VSRAQ's headline W4A16 result is weight-only quantization (flips come from perturbed hidden states feeding the router). Not a citation error, but the report should not imply these papers measured activation-induced flips specifically.

---

## 2604.06798 — MoBiE

- **Title:** MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization
- **Authors:** Zhixiong Zhao, Zukang Xu, Zhixuan Chen, Dawei Yang
- **Published:** 2026-04-08 (v1); last revised 2026-04-20 (v4, 1 KB).
- **Venue:** none — **WITHDRAWN by the author.**
- **Exists:** the arXiv record exists, but the paper is withdrawn and the PDF/HTML are no longer served (PDF 404; html/v1 is a 477-char stub).
- **Verdict: partially_confirmed.** The abstract matches what the report implies (a MoE binarization method that explicitly targets "quantization-induced routing shifts" via "an error constraint guided by the input null space to mitigate routing distortion"), so the report did not invent the content. But the record is withdrawn with the author admitting derivation errors, so it cannot stand as [S]-grade supporting prior art.

Exact arXiv status (abs page, fetched 2026-10-10):
- "This paper has been withdrawn by Zhixiong Zhao"
- Submission history: "[v1] Wed, 8 Apr 2026 08:12:26 UTC (790 KB) (withdrawn) / [v2] Mon, 13 Apr 2026 / [v3] Tue, 14 Apr 2026 / [v4] Mon, 20 Apr 2026 16:58:22 UTC (1 KB) (withdrawn)"
- Comments: "Although previously revised, per strict university regulations regarding incorrect affiliation, I am unauthorized to retain this manuscript. Furthermore, fundamental derivation errors in the NGES section compromise the mathematical framework, alongside misleading overlapping wording. The paper is therefore withdrawn"

Abstract-level claims (unverifiable against full text now): Qwen3-30B-A3B "reduces perplexity by 52.2%, improves average zero-shot performance by 43.4%, achieves over 2x inference speedup"; code at github.com/Kishon-zzx/MoBiE.

**Correction:** in A2, either drop MoBiE from the list or annotate it: "MoBiE([2604.06798], 저자 철회됨 2026-04-20; NGES 절 유도 오류 인정)". Downgrade from [S] to [U] or remove. Do not rely on any MoBiE number.

---

## 2506.13329 — EAQuant

- **Title:** EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization
- **Authors:** Zhongqian Fu*, Tianyi Zhao*, Ning Ding, Xianzhi Yu, Xiaosong Li, Yehui Tang, Yunhe Wang (Huawei / BUAA)
- **Published:** 2025-06-16 (v1); v3 2026-02-01. Text header: "Preprint. February 3, 2026."
- **Venue:** preprint/unknown (web search found no conference acceptance).
- **Exists:** yes.
- **Verdict: confirmed** for the report's parenthetical "라우팅 KL 최소화", with one precision nuance below.

What the paper does (Sec. 2.3 "Expert-Aware Routing Consistency Alignment", Eq. 10):
> "min_θ E_x̃ ||x̃W̃gate − x̃Q(W̃gate)||²₂ + E_x̃ [D_KL-Top (p_fp || p_quant)]" — labelled "Logit MSE" + "Routing KL Divergence", where θ are "quantization parameters such as scale and zero-point for Wgate".
> "We introduce the KL-Top divergence proposed in (Hu et al., 2025b), which restricts KL regularization to a subset of routing-relevant experts rather than the full expert set. Specifically, for each token, KL-Top is applied to the union of the top-k experts and an additional α(n − k) experts ranked immediately after the top-k".

Nuance: it is not plain KL minimization — it is a dual objective (router-logit MSE + KL-Top over a top-k-plus-α subset of experts), applied only to the *router's* quantization parameters. Appendix Table 8 (OLMoE-7B, W4A4, router W8A8): α = 0.00 (top-k only) is best, "improving the average accuracy from 66.69 to 67.97 compared to the DuQuant baseline"; expanding α toward 1.00 "leads to diminished gains and, in some cases, performance degradation". If the report wants to be exact: "EAQuant(라우터 로짓 MSE + top-k 제한 KL(KL-Top) 정렬)".

Headline numbers: "average accuracy improvements of 1.15 - 13.81% across three diverse MoE architectures" (abstract); W4A4: +1.37 / +1.15 / +1.15 avg accuracy over DuQuant on OLMoE-7B / DeepSeek-MoE-16B / Mixtral-8x7B (Table 1, Sec. 3); "+13.81% average accuracy on Mixtral-8x7B" under W3A3/W2A4 (Table 1, row 985). Baseline is DuQuant, calibration 128 x WikiText2 segments. Appendix: Mixtral-8x22B PPL (Table 10), C4 calibration (Table 11).

Stated limitation: "its performance degrades in ultra-low-bit regimes (e.g., W1A4) due to the granularity-sparsity trade-off, requiring QAT for accuracy recovery." (Sec. 4)

Also relevant: Appendix C.3 / Fig. 6 shows "small quantization-induced perturbations in router outputs can significantly alter top-k expert assignment" across OLMoE, DeepSeek-MoE, Mixtral-8x7B — a qualitative routing-flip measurement, consistent with the report's "측정" claim. No flip-probability model and no flip/smooth loss decomposition.

---

## 2605.23078 — GEMQ

- **Title:** GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs
- **Authors:** Jianing Deng, Song Wang, Dongwei Wang, Zijie Liu, Tianlong Chen, Huanrui Yang, Jingtong Hu (Pitt / UCF / Arizona / UNC)
- **Published:** 2026-05-21 (v1).
- **Venue:** **ICML 2026** (arXiv comment "ICML 2026"; text footer: "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026.")
- **Exists:** yes.
- **Verdict: confirmed.** "1.5-bit에서 40% 토큰 재라우팅" matches the paper.

Exact source of the number:
- Fig. 1 caption (Sec. 1): "(b) over 40% of tokens are routed to different experts after 1.5-bit quantization, revealing substantial distortion in router distributions. Statistics are computed from the Mixtral-8×7B model on WikiText2."
- Table 6 (Sec. 5.2, "Perplexity and average router change ratio of 1.5-bit quantized Mixtral-8×7B, using different expert importance estimation models"): change ratio vs FP = **41.31%**; vs 4-bit = 39.10%; vs 2-bit = 34.69%; vs 2-bit (RFT) = 38.38%. Text: "using a 2-bit model for importance estimation reduces the average expert selection change rate from 41.31% to 34.69% compared to the FP model".

Scope caveats the report should carry: the 40% figure is one model (Mixtral-8×7B), one dataset (WikiText2), at an *average* budget of 1.5 bits/expert under expert-wise mixed precision (candidates {1,2,3} bits, attention 4-bit), measured against the FP model's selections. It is a motivation figure, not a general law.

GEMQ's response is different from the other three: it does not try to preserve FP routing during calibration; it (1) solves a global LP for expert bit-widths from a Fisher-based importance proxy and (2) fine-tunes the routers with the task loss *after* expert quantization. Table 5 (Mixtral-8×7B 1.5-bit, WikiText2 / C4 PPL): FP 3.84 / 7.40; w/o calibration 9.66 / 23.53; w/ FP router logits 9.29 / 22.72; w/ router fine-tuned 7.69 / 17.18.

A fact directly relevant to the report's remaining-gap hypothesis ("flip component vs smooth component of the loss"): GEMQ's Appendix (Fig. 9 / Table 6) notes that the router-fine-tuned 2-bit reference gives *better* 1.5-bit perplexity "even though the expert selection change ratio does not improve" (38.38% vs 34.69%), with the authors conjecturing a "rugged loss landscape" after fine-tuning. I.e. raw flip rate is not monotone with loss in GEMQ's own data — the report can cite this as evidence that a flip-vs-smooth decomposition is needed rather than a flip count alone.

---

## Summary table

| id | paper | exists | venue | verdict | action |
|---|---|---|---|---|---|
| 2606.05688 | VSRAQ | yes | preprint (ICML-2026 template) | confirmed | optional: note 4-bit-only scope, W4A16 is weight-only |
| 2604.06798 | MoBiE | record only, **withdrawn** | none | partially_confirmed | drop or annotate as withdrawn; downgrade [S] |
| 2506.13329 | EAQuant | yes | preprint | confirmed | optional precision: "logit MSE + KL-Top (top-k 제한 KL)" |
| 2605.23078 | GEMQ | yes | ICML 2026 | confirmed | add scope: Mixtral-8×7B, WikiText2, 41.31% vs FP (Table 6) |
