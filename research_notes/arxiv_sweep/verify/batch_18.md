# Citation verification — batch 18

IDs: 2510.00028, 2602.10718, 2603.04428, 2608.04074. All four exist on arXiv. Checked 2026-10-10 against full text fetched with `scripts/arxiv_fetch.py`. Quotes are from the papers (untrusted data, quoted only).

---

## 2510.00028 — Rethinking RoPE Scaling in Quantized LLM: Theory, Outlier, and Channel-Band Analysis with Weight Rescaling

- **Authors:** Ye Qiao, Haocheng Xu, Xiaofan Zhang, Sitao Huang. Published 2025-09-26 (v1 only). cs.LG.
- **Venue:** preprint/unknown (no comment/journal_ref). Its companion **2509.14391 "Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs"** (Qiao, Huang; 2025-09-17) is an **AAAI-26 Student Abstract** (ojs.aaai.org/index.php/AAAI/article/view/42269). The two IDs the report cites side by side are the short abstract and the full paper of the *same* work (same first author, same method), not two independent reports.
- **Report claim (idea sections, [S]):** "Q-ROAR와 'Rethinking RoPE scaling'은 YaRN × W4가 파괴적으로 상호작용(NTK/linear는 PPL>100)하고 대역별 스케일로 복구한다고 보고"
- **Verdict: partially_confirmed.**
  - Confirmed — interaction and the ">100" statement (Sec. 3.1, text only): "quantized models experience much sharper degradation even under the most favorable configuration, such as quantization combined with YaRN. With more naïve interpolation strategies like NTK or linear scaling, the models fail entirely at long contexts, reaching perplexity values greater than 100. Additional results are provided in the Appendix." No table prints a >100 value; Fig. 1(b) y-axis tops at ~6.5 and Appendix Fig. 2 at 7.5 (AWQ+NTK up to 19.5K only).
  - Confirmed — per-band weight scaling: "groups RoPE dimensions into a small number of frequency bands and performs a lightweight search over per-band scales for Key and Query weights"; Appendix: "B=8 log-spaced bands … K=7 grid points per band"; symmetric option WQ←g_b·WQ, WK←g_b⁻¹·WK.
  - Overstated — "파괴적" and "복구": with moderate YaRN (s=8, ≤16K) all W4 baselines stay close to FP16 (Table 2, Proof-pile, 16K: FP16 2.438 / RTN W4 2.517 / AWQ W4 2.476 / Q-ROAR W4 2.458). Destructive behaviour appears under NTK/linear schemes or aggressive scaling. Q-ROAR *mitigates* rather than recovers: Table 2 (s=32) at 131K: FP16 5.386 / RTN W4 9.958 / AWQ W4 8.700 / Q-ROAR W4 7.869; at 32K: 4.261 / 6.249 / 5.460 / 5.054. Table 1 (GovReport, YaRN=16, LLaMA-2-7B) at 32K: FP16 6.069 / RTN W4 6.783 / AWQ W4 6.302 / Q-ROAR W4 5.833 (below FP16 there; ≤0.04 PPL gap at 2K–16K). Conclusion: "lowers perplexity on long-context workloads by 14% to 21% compared to directly applying quantization".
  - Setup: LLaMA-2-7B (+Vicuna-7B in appendix), weight group size 128, RTN/AWQ W4 and W4-A4 (Table 3), 256-token sliding-window PPL. Diagnostics: "interpolation pressure" and "tail-inflation ratio". Analysis is motivational (aliasing, dynamic-range dilation, anisotropy, outlier shifting); no theorem with stated assumptions.
- **Suggested correction:** "Q-ROAR(2509.14391, AAAI-26 student abstract)과 그 풀 버전(2510.00028)은 W4 PTQ와 RoPE PI 결합이 장문맥 PPL을 악화시킨다고 보고한다 — YaRN s=8(≤16K)에서는 완만하고, NTK/linear 스케일링은 PPL>100으로 실패(Sec. 3.1 본문 서술, 표 없음). 8개 로그 간격 대역별 WQ/WK 스케일 탐색으로 RTN 대비 14–21%, AWQ 대비 7–10% PPL 감소(LLaMA-2-7B, Table 1–2); s=32·131K에서는 FP16 5.386 대비 7.869로 완전 복구는 아니다."

---

## 2602.10718 — SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining

- **Authors:** Yifan Zhang, Zunhai Su, Shuhao Hu, Rui Yang, Wei Wu, Yulei Qian, Yuchen Xie, Xunliang Cai (Meituan LongCat Team). Published 2026-02-11, updated 2026-04-28 (v2).
- **Venue:** a web search returns a NeurIPS 2026 virtual poster page with this exact title (neurips.cc/virtual/2026/poster/148693); neurips.cc is unreachable from this sandbox so this is not directly verified. arXiv comment field is empty. Treat as "NeurIPS 2026 (poster, per search listing)".
- **Report claim ([S]):** "SnapMLA는 위치 부분을 고정밀로 두라고 주장 … — 상반된다" (vs FlashMLA FP8/FP4 formats that also quantize the 64 RoPE dims).
- **Verdict: confirmed.**
  - Abstract: "RoPE-Aware Per-Token KV Quantization: Motivated by our analysis of the heterogeneous quantization sensitivity inherent to the MLA KV cache, this approach preserves the RoPE part in high precision."
  - Sec. 3 (LongCat-Flash-Thinking): "the RoPE component spans a significantly wider dynamic range (reaching ±10³), exhibiting distinct outlier tails, while the content component is tightly concentrated around zero (within ±10¹) … FP8 quantization leads to an order-of-magnitude increase in Mean Squared Error (MSE) for the RoPE component (as shown in Figure 3b) … we propose a RoPE-aware strategy that applies FP8 quantization exclusively to the content components, while retaining the RoPE components in BF16".
  - Appendix G (Table 3 / Fig. 5, 32K context): Config A = per-token with RoPE also quantized → "including RoPE term in the quantization process (Config A) results in a substantial increase in error metrics, particularly in the deeper layers. This 'error explosion' …". Evidence is layer-wise RMSE / cosine distance / relative L2 only; **no end-to-end benchmark accuracy is reported for the RoPE-quantized variant.**
  - Scope caveats for the report's "상반된다": SnapMLA's claim is for **FP8 E4M3 per-token** on the content latent; it does not test FP4/NVFP4-style formats, nor FlashMLA's specific scale granularities (e8m0 per 32 / e4m3 per 16). The contradiction is a design-choice disagreement, not a head-to-head measurement.
  - Other numbers: Table 1 DeepSeek-V3.1 BF16→FP8: MMLU-Pro 84.41→84.43, MMLU-Redux 90.48→90.89, IFEval 86.32→87.25, Arena-Hard 57.10→55.50, MATH-500 98.80→98.20; "up to a 1.91× improvement in throughput" (Fig. 1). Models: DeepSeek-V3.1 and LongCat-Flash-Thinking. Baseline: FlashMLA BF16.
- **Correction:** none required; optionally add "(FP8 E4M3 기준; 수치 오차 분석만, RoPE 양자화 변형의 벤치마크 정확도는 미보고)".

---

## 2603.04428 — Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices

- **Author:** Yakov Pyotr Shkolnikov (single author). Published 2026-02-17 (v1). Comment: "24 pages, 6 figures, 16 tables. Open-source implementation at https://github.com/yshk-mxim/agent-memory".
- **Venue:** preprint/unknown.
- **Report claim ([S]):** "MLA 양자화 민감도 데이터는 단일 프리프린트(4-bit KV PPL +2.8% GQA vs +3.0% MLA, [2603.04428])와 …뿐이다."
- **Verdict: partially_confirmed.** The two numbers exist but are selectively quoted and cannot carry a GQA-vs-MLA sensitivity comparison.
  - Table 9 ("Perplexity with actual Q4 KV caches … group size 64 … 7,935 tokens, 512-token windows, 256-token stride"): Gemma 3 12B (GQA) FP16 14.40 → Q4 14.30 (−0.10, **−0.7%**); DeepSeek-V2-Lite 16B (MLA) 6.26 → 6.45 (+0.19, **+3.0%**); Llama 3.1 8B (GQA) 4.28 → 4.40 (+0.12, **+2.8%**).
  - The report cites only Llama for "GQA"; the other GQA model (Gemma) shows −0.7%, which the paper calls "within measurement noise (Q4 cannot improve over FP16 in principle; the negative sign reflects evaluation variance)". The 0.2 pp gap between +2.8% and +3.0% is below that noise level. Paper: "We do not report confidence intervals." and "The evaluation uses a local WikiText-2 corpus, not the standard benchmark split."
  - What is quantized: mlx-lm `QuantizedKVCache`, uniform 4-bit, group 64 with bf16 scale+bias, applied to the **decompressed per-head K/V** of DeepSeek-Coder-V2-Lite ("16 KV heads, K=192/V=128, MLA"; "K=192 (128 nope + 64 rope), V=128") — **not** to the 512-d latent c_KV and not under weight absorption. Both arms also use 4-bit weights: "Both FP16 baseline and Q4 caches use identical model weights (4-bit quantized via mlx-lm)."
  - The paper is a systems paper (TTFT 11–136× from disk-persisted caches); the PPL table is a side quality check (Sec. 4.8), and its GQA/MLA explanation is explicitly speculative: "The divergence between models likely reflects architectural differences".
- **Suggested correction:** "단일 프리프린트(2603.04428, 시스템 논문 Table 9)에서 group-64 균일 4-bit KV(흡수 전 per-head K/V 기준, MLA latent 아님) PPL 변화는 Gemma 3 12B(GQA) −0.7%(노이즈), Llama 3.1 8B(GQA) +2.8%, DeepSeek-Coder-V2-Lite 16B(MLA) +3.0%; CI 없음·로컬 WikiText-2 7,935 토큰으로 GQA↔MLA 민감도 차이를 뒷받침하지 못한다."

---

## 2608.04074 — Spend Bits Where Queries Look: KV Cache Vector Quantization with Attention-Preserving Transforms (NOVA-KV)

- **Authors:** Samuel Fernández-Menduiña, Amir Ziashahabi, Eduardo Pavez, Antonio Ortega, Salman Avestimehr. Published 2026-08-04 (v1).
- **Venue:** preprint/unknown (no comment/journal_ref; web search found nothing).
- **Report claim ([S]):** "MHA에서는 query 공분산 가중 변환이 Hadamard를 이긴다는 주장이 있다 (OSCAR [2605.17757], Spend Bits Where Queries Look [2608.04074])" followed by "MLA의 latent는 모든 헤드가 공유하므로 가중 행렬 M이 헤드 합으로 정의되는 점이 구조적으로 다르고 … MHA의 online Hadamard(QuaRot)와 달리 디코드 비용이 0이다."
- **Verdict: partially_confirmed.**
  - Confirmed — query-second-moment-weighted, non-orthogonal key transform beats Hadamard-based methods at 2 bits. Theorem 1: "R⋆_{K,p} = M_q^{1/2} E_{1:p}, where EΛE⊤ is the eigendecomposition of M_q^{1/2} S̃_k M_q^{1/2}" (M_q = Q⊤Q, S̃_k = centered key scatter); Prop. 1 (generalized Parseval): "‖r − r̂‖² = ‖k − k̂‖²_{M_q}". Table 1 classifies QuaRot as "rand. Had. orth. / Heuristic", OSCAR as "calib.+Had. orth. / Heuristic", NOVA-KV as "calib. non-orth. / attn. products". Table 2 (RULER NIAH, 2-bit): "on Qwen3-8B at 128K, QuaRot and OSCAR reach 0.0 and 25.3 against 83.4 for BF16, where NOVA-KV reaches 75.4"; Llama-3.1-8B "NOVA-KV reaches 92.4 at 64K and 63.3 at 128K against 76.9 and 36.7 for OSCAR"; on GPT-OSS-20B "QuaRot scores 0.0 at every length".
  - **Not confirmed — "MHA":** every evaluated model is GQA (Llama-3.1-8B, Qwen3-8B, Qwen3-4B-Thinking-2507, GPT-OSS-20B); the string "MHA"/"multi-head attention" does not occur in the paper. The paper treats GQA explicitly: "For GQA, we calibrate across heads sharing the same KV cache" and Appendix F.5 "Query Statistics under Grouped-Query Attention": the logit error for the shared keys is a **sum over the H query heads** sharing one KV head, with M_q accumulated per (layer, KV head). This weakens the report's "구조적으로 다르다" argument: a head-summed weighting matrix is already what NOVA-KV does for GQA; MLA differs in degree (all heads share) and in the W_UK absorption path, not in the head-sum itself.
  - Nuance — the win is transform **+ VQ jointly**, not the transform alone. Table 4 (Qwen3-8B, RULER NIAH): NOVA-KV transform with fixed-rate SQ collapses to 0.0; text: "Under SQ … our transform collapses: it compacts energy rather than flattening it … Replacing SQ by VQ is worth 7.5 and 37.4 points under OSCAR's basis; replacing OSCAR's basis by ours, under VQ, is worth a further 8.3 and 12.7." Rate: 2.22 BPE including a BF16 sink(64)+recent(256) band; layer 0 kept BF16.
  - Decode-cost nuance relevant to the report's "디코드 비용 0" contrast: keys need one online d×d product per step per (layer, KV head) — "The synthesis R_K^{−⊤} is folded into the query once per step: the engine forms the modified query q̃ = q R_K^{−⊤} (a single d × d product, amortized over the whole cache)"; values: "whose inverse is absorbed into the output projection, so it adds no online matmul either."
  - "query 공분산": the paper uses the uncentered query second moment M_q (its own Fig. 2 caption also says "query covariance"), so the report's wording is acceptable.
- **Suggested correction:** "GQA 모델(Llama-3.1-8B, Qwen3-8B/4B, GPT-OSS-20B)에서 query 2차 모멘트 M_q로 가중한 비직교 키 변환(+ VQ)이 Hadamard 기반 QuaRot/OSCAR를 2-bit에서 크게 이긴다고 주장(NOVA-KV, Table 2; Qwen3-8B 128K NIAH 75.4 vs 0.0/25.3). 단 변환 단독(SQ)은 0.0으로 붕괴(Table 4), 평가 모델에 MHA는 없고, GQA에서는 이미 KV 헤드를 공유하는 쿼리 헤드 합으로 M_q를 정의한다(App. F.5)."

---

### Summary table

| id | exists | verdict | one-line |
|---|---|---|---|
| 2510.00028 | yes | partially_confirmed | Interaction + NTK/linear ">100" (text) + per-band scales confirmed; "복구" overstated (mitigation 14–21% vs RTN, still above FP16 at s=32); same work as 2509.14391 (AAAI-26 student abstract) |
| 2602.10718 | yes | confirmed | Keeps RoPE part BF16, content FP8 per-token; evidence = distribution + layer-wise error (no benchmark for RoPE-quantized variant); NeurIPS 2026 poster per search listing |
| 2603.04428 | yes | partially_confirmed | +2.8% (Llama, GQA) / +3.0% (DeepSeek, MLA) real (Table 9) but Gemma (GQA) −0.7%, no CIs, per-head K/V not latent; cannot support MLA>GQA sensitivity |
| 2608.04074 | yes | partially_confirmed | Beats Hadamard methods at 2-bit, but on GQA not MHA; gain needs VQ; GQA already uses head-summed M_q |
