# Citation verification — batch 13

IDs: 2607.26515, 2605.13907, 2606.00206, 2506.01939
Report: `reports/LLM 양자화 연구 아이디어 10선.md` · contexts from `research_notes/arxiv_sweep/report_citations.json`
Source texts: `scratchpad/papers/<id>/<id>.txt` (arxiv_fetch.py). Line numbers below refer to those text files. Paper text is quoted as untrusted data.

Summary: all four papers exist. All four attributed claims are **confirmed**; no numeric errors found. Several caveats worth carrying into the report are listed per paper (HiFloat4 uses TIS=5.0 in every run and simulates FP4; AIS's "monotonic KL" is an empirical single-seed FP8 observation, not a theorem; Lotfi's 52% is one model/one benchmark/one PTQ method; the 80/20 paper's gains are DAPO-on-Qwen3-math-only and flat on 8B).

---

## 2607.26515 — HiFloat4 Format for End-To-End Reinforcement Learning Post-Training of Large Language Models

- **Exists:** yes. First author Hei Yi Mak (Huawei; 13 authors). Published 2026-07-29, cs.LG. No comment/journal_ref. WebSearch (arXiv/alphaXiv/HF mirrors only): no venue → **preprint**.
- **Report context** ([S] tag): `HiFloat4/Rollout-ResQ("학습 정밀도를 올리면 FP4 롤아웃이 더 나빠진다")` listed among papers giving "different prescriptions" for FP4/FP8 rollout mismatch; report also claims no paper jointly reports entropy, KL(μ‖π) and reward in one run.
- **Verdict: confirmed.** The claim is stated verbatim in the abstract and demonstrated in Table 1.
  - Abstract (l.36-38): "Counterintuitively, restoring the training policy to higher precision while keeping the rollout in FP4 makes accuracy worse than full FP4 baseline, exposing rollout–training mismatch as the principal failure mode".
  - Sec 4.2, Table 1 (Qwen2.5-3B, GSM8K test, GRPO, l.512-575) — accuracy (%), HiF4 / MXFP4:
    - BF16 train + BF16 rollout: 86.96
    - FP4 train + BF16 rollout: 86.88 / 86.20
    - FP4 train + W4A16 rollout: 85.22 / 81.57
    - FP4 train + W4A8 rollout: 85.06 / 79.91
    - FP4 train + W16A4 rollout: 84.30 / 77.71
    - FP4 train + FP4 rollout: 82.03 / 73.31
    - **BF16 train + FP4 rollout: 78.01 / 50.57** ("catastrophic drop ... performing significantly worse than even the fully quantized settings", l.563-566)
  - Diagnosis: rollout *activation* quantization (W16A4 worse than W4A16) is the dominant source; outlier-driven underflow to zero (Fig. 1).
- **Correction needed:** none for the quoted claim. Optional precision: the "worse" result is on Qwen2.5-3B/GSM8K (Table 1) only; the 7B-Math experiments (Table 3) do not repeat that ablation.
- **New relevant facts for the report:**
  - **Every experiment already uses truncated importance sampling**: "In all experiments, we apply a token-level truncated importance sampling threshold [59] of 5.0." (Appendix A, l.1490). So HiFloat4's "prescription" (Rollout-ResQ sparse residual on the rollout GEMM) is *on top of* TIS, not instead of IS correction. The report's framing that these papers give "different prescriptions" is fine, but it should not imply HiFloat4 is IS-free.
  - No policy entropy and no KL(rollout‖train) are reported anywhere (0 hits for "entrop"; KL appears only as the GRPO penalty term). Consistent with the report's "no joint entropy/KL/reward" claim.
  - All FP4 is **simulated** ("Due to the lack of native FP4 hardware support, all experiments rely on FP4 simulations ... preventing direct measurement of real speedup", Limitations l.1152-1154); throughput claims are theoretical FLOP counts (Sec 5.4: Rollout-ResQ-S2:4 = 3/4 mdn effective FP4 cost).
  - Headline gaps (abstract/contributions l.145-146): HiF4 BF16-gap 4.9% → 1.1%; MXFP4 13.6% → 5.3%, on Qwen2.5-3B (GSM8K) and Qwen2.5-Math-7B (DAPO-Math-17K; Table 3 Mean@32).
  - Positions itself against QeRL ("unlike QeRL, we do not" keep gradients in high precision, l.86) and cites FP8-RL/Jet-RL-style TIS and "update-aware quantization" as prior mismatch handling (l.153-156).

---

## 2605.13907 — AIS: Adaptive Importance Sampling for Quantized RL

- **Exists:** yes. First author Jiajun Zhou (with Wei Shao, Lingchao Zheng, Yuwei Fan, Ngai Wong). Published 2026-05-13, stat.ML. No comment/journal_ref; WebSearch finds only arXiv mirrors → **preprint**.
- **Report context** ([S] tag): `AIS("KL이 정책 날카로워짐과 함께 단조 증가, 초기엔 탐색 도움 후기엔 편향")`.
- **Verdict: confirmed.** Both halves of the quoted paraphrase appear in the paper.
  - Monotonic KL growth (Sec 2, l.243-246): "The induced bias scales with DKL (πθ̂old ∥πθold ), which we observe grows monotonically during training as the policy sharpens and quantization artifacts interact with increasingly peaked distributions."
  - Fig. 4 (LLaDA-8B, GSM8K, l.342-348): tracks |KL(πroll‖πtrain)|, E[|Δp|] and ESS ratio over training; "the uncorrected FP8 rollout steadily drifts away from the learner as distributional sharpening intensifies, with its ESS ratio decaying from above 0.95 to below 0.70."
  - Double-edged sword (abstract l.14-18; intro l.86-89): "early in training it provides a stochastic exploration bonus, exposing the gradient to trajectories the trainer would otherwise under-sample, but the same perturbation transitions into a destabilizing source of bias as the policy concentrates."
- **Correction needed:** none. Suggested hedge: "단조 증가" is the authors' *empirical observation* ("we observe"), on FP8 rollout / BF16 trainer, one seed per config; it is not a proved property. The KL direction in the paper is KL(rollout ‖ train), matching the report's KL(μ‖π) if μ = rollout policy.
- **New relevant facts:**
  - **No policy entropy reported** (0 hits for "entrop"). Reports KL, E|Δp|, ESS (Fig. 4, 6) and reward curves (Fig. 1 Qwen3-8B GSM8K; Fig. 7 Countdown/MATH) — on different figures/tasks. Does not contradict the report's "no joint entropy+KL+reward in one run" claim.
  - Self-stated limitations (Appendix F, l.1897-1908): 8–9B models only; FP8 only ("scaling to ... FP4, INT4 remains an open question that may require re-tuning C, δ, γ"); "our exploration-bonus claim is supported empirically rather than by a formal causal analysis"; "compute constraints limit each configuration to a single seed".
  - Theory: Prop. 1 (oracle mixing coefficient α* under assumptions A1–A4, surrogate MSE criterion), Prop. 2 (bounded second moment of truncated AIS estimator), Prop. 3 (reduces to on-policy when πrollout = πtrain). These are about the estimator, not about KL dynamics.
  - Numbers: naive FP8 rollout drops AIME25 on Qwen3-8B by 6.63% (l.77-78); FP8 rollout speedup 1.5–2.76× (2.76× at BS=4, G=4 on LLaDA-8B); ~48–50% memory reduction. Models: LLaDA-8B-Instruct (diffusion), Qwen3-8B, Qwen3.5-9B; GRPO.
  - Note for the report's "서로 다른 처방" sentence: AIS explicitly *embraces* BF16-train/FP8-rollout mismatch and argues against uniform-precision pipelines (l.80-83) — the opposite stance to Jet-RL/HiFloat4's end-to-end low precision.

---

## 2606.00206 — Quantized Reasoning Models Think They Need to Think Longer, but They Do Not

- **Exists:** yes. Sanae Lotfi, Polina Kirichenko, Steven Li, Zechun Liu; affiliation "FAIR at Meta, Meta AI" (l.5). Published 2026-05-29, cs.LG. No comment/journal_ref; WebSearch: arXiv/alphaXiv/HF + a blog post → **preprint**.
- **Report contexts** (tags ["?","V"]):
  1. "양자화 추론 모델에서 토큰 KL이 고엔트로피 위치에 집중된다는 관찰은 Meta FAIR가 보고"
  2. "양자화-FP 간 토큰 KL이 고엔트로피 위치에서 최대이고 그곳에서 'wait/but/alternatively'가 과표집되어 CoT가 길어지며 실패의 최대 52%가 정답을 중간에 얻고도 확정하지 않는 경우라고 보고 ... 디코딩 측 로짓 페널티만 제안했다."
  3. Experimental plan reuses the "'정답 미확정' 비율(2606.00206 프로토콜)".
- **Verdict: confirmed** on every specific point. The "?" tag can be resolved to [V].
  - KL ↔ entropy: Fig. 4c / Sec 6 (l.1410-1414): "the correlation coefficient between the KL divergence computed on MATH-500 under identical generation prefixes and the next-token entropy of the full-precision model is ρ = 0.92 across all positions" (Spearman, BF16 vs AWQ 3-bit, DeepSeek-R1-Distill-Qwen-1.5B).
  - Over-sampled markers: Fig. 4 (l.739-741): highest-KL tokens are "Wait", "But", "Alternatively", "if", "maybe", "verify", "think"; lowest-KL are math/format tokens. Density ratio high-KL/low-KL per CoT: 0.57 (BF16) → 1.15 (AWQ 3-bit) (l.746-750). Markers are "2–4× more likely to appear among the top-20 ... at high-entropy positions" (Fig. 7; τ = per-model 80th percentile of entropy, "similarly to Wang et al. (2025)" = 2506.01939).
  - Longer CoT: Fig. 2 (l.367-369): 3-bit AWQ on 1.5B raises MATH-500 CoT from 5.2K to 23.4K tokens (4.5×) while accuracy falls 85.6% → 47.0%.
  - 52%: Sec 4.2 / Fig. 3 (l.609-613): DeepSeek-R1-Distill-Qwen-1.5B, MATH-500: BF16 72 errors/500, 19 overthinking (26%); AWQ 3-bit 265 errors, 139 overthinking (52%, "7.3× increase"); GPTQ 3-bit 39; FlatQuant W4A4KV4 64. "up to 52%" is this single setting.
  - Decoding-side only: the intervention is a "training-free logit penalty" on 50 curated markers (Sec 5.1, Eq. 1); CoT −12–23% on average (Table 1 per-model range 4.1–28.0%); overthinking errors 139 → 58 (−58%) on AWQ 3-bit; no training-time or calibration-time fix is proposed. Future work mentions only adaptive λ and automatic marker lists (l.1442-1446).
- **Correction needed:** none. Precision suggestions:
  - "실패의 최대 52%" → specify "R1-Distill-Qwen-1.5B, AWQ 3-bit, MATH-500 (139/265 errors; BF16 26%)".
  - The "protocol" the report plans to reuse = four-way error categorization (overthinking / logical / arithmetic / formatting-hallucination-other) with manual annotation of a subset then a **GPT-5 LLM judge** tuned to >95% human agreement; prompt in Table 12 / Sec D (l.597-599, 3083-3090). The report's experiment plan should budget for an LLM judge.
- **New relevant facts:**
  - Scope: 5 models (R1-Distill-Qwen 1.5B/7B/14B, R1-Distill-Llama-8B, QwQ-32B), 3 PTQ families (GPTQ, AWQ at 3/4-bit g128; FlatQuant W4A4KV4 / W8A8KV8), 5 benchmarks (GSM8K, MATH-500, AIME-120, LiveCodeBench, GPQA-Diamond). **Qwen3 and GPTAQ are not evaluated** — the report's plan (Qwen3-8B thinking, GPTAQ) extends beyond this paper's coverage.
  - The KL-vs-entropy analysis itself is only reported for the 1.5B model under AWQ 3-bit on MATH-500 (Fig. 4); the top-20 marker analysis adds QwQ-32B (Fig. 7).
  - Authors' own hedge (l.1419-1420): "this argument provides intuition for one specific failure mode, not a complete causal explanation of all types of degradations caused by PTQ."
  - Penalizing the 50 *lowest*-KL tokens is harmful (CoT +41%, accuracy −9.5%); random tokens have no consistent effect (l.158-161) — useful control design.
  - Overthinking is also present in BF16 (penalty cuts BF16 CoT 5.6–10.0%); "Quantization amplifies a failure mode already present".
  - It cites 2506.01939 for the entropy-threshold convention but does not connect to RL rollout mismatch — consistent with the report's "둘을 롤아웃 불일치 보정에 연결한 연구는 없다".

---

## 2506.01939 — Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

- **Exists:** yes. First author Shenzhi Wang (Qwen team / Tsinghua; 18 authors). Published 2025-06-02, v2 2025-11-13, cs.CL. Comment: "Accepted to NeurIPS 2025. 25 pages, 17 figures, 2 tables"; text l.64 "Accepted to NeurIPS 2025." → **NeurIPS 2025** (report's venue tag is correct).
- **Report context** ([V]): "RLVR의 이득이 20% 고엔트로피 fork 토큰에서 온다는 결과도 있다 ([Beyond the 80/20 Rule 2506.01939, NeurIPS 2025])".
- **Verdict: confirmed.**
  - Sec 1 (l.107-120): "retaining policy gradient updates for only 20% of tokens with the highest entropy and masking gradients of the remaining 80% ... +11.04 on AIME'25 and +7.71 on AIME'24 for the Qwen3-32B base model, and +4.79 on AIME'25 and +5.21 on AIME'24 for the Qwen3-14B base model ... training exclusively on the 80% lowest-entropy tokens results in severe performance degradation ... the high-entropy minority tokens ... account for nearly all performance gains in RLVR."
  - Table 2 (Acc@16, DAPO all tokens → DAPO forking tokens): Qwen3-32B AIME'24 55.83 → 63.54, AIME'25 45.63 → 56.67; Qwen3-14B AIME'24 45.21 → 50.42, AIME'25 38.13 → 42.92; Qwen3-8B AIME'24 33.33 → 34.58, AIME'25 25.42 → 26.25 ("performance remains unaffected", l.800).
- **Correction needed:** none. Precision suggestions for the report:
  - The 20% is a **per-batch top-ρ entropy mask** (ρ = 20% in Eq. 6) inside the DAPO recipe (clip-higher, dynamic sampling), not a fixed entropy threshold; the base-model statistic that 20% of tokens have entropy > 0.672 (and 50.64% < 1e-2) is for Qwen3-8B (Sec 3, l.295-296).
  - Gains are size-dependent: large on 32B/14B, ~flat on 8B; ablation (Fig. 7) shows 10%/50%/100% are worse than 20% on 14B/32B but similar on 8B.
- **New relevant facts / limitations (l.1024-1032):** Qwen3 base models + math only; Llama-3.1-8B attempts "struggled to achieve meaningful performance on the AIME benchmarks" and the Llama results are "less convincing" (l.899-908); "the effective proportion of 20% ... may need to be adjusted to a different value" in other RLVR settings. Forking-token-only training **increases response length** (Table 2 Len@16: 32B AIME'24 9644 → 12198 tokens). Minor internal inconsistency: Fig. 6 caption says AIME'24 +7.73 for 32B while text/Table 2 say +7.71. Entropy threshold convention (80th percentile) is reused by 2606.00206 (Fig. 7), which is the only link between the two papers; neither touches rollout–training mismatch.
