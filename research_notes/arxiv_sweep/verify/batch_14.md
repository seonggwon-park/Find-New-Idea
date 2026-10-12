# Citation verification — batch 14

IDs: 2512.23075, 2609.18063, 2505.23786, 2605.15152. Checked 2026-10-10 against the arXiv full text (export.arxiv.org API + extracted text) and OpenReview API for venues. Paper text is quoted as untrusted data.

---

## 2512.23075 — Trust Region Masking for Long-Horizon LLM Reinforcement Learning

- **Exists:** yes. Yingru Li, Jiacai Liu, Jiawei Xu, Yuxuan Tong, Ziniu Li, Qian Liu, Baoxiang Wang. v1 2025-12-28, latest 2026-06-26. cs.LG. arXiv comment: none.
- **Venue:** ICML 2026 main conference. OpenReview note `dPaBOu0Xwk` has `venue = "ICML 2026 regular"`, `venueid = ICML.cc/2026/Conference`, with a `Camera_Ready_Revision` invitation; icml.cc lists it as poster 62782.
- **Report claim** (Idea risk paragraph, tag [S]): "IcePop, Trust Region Masking([2512.23075]) 등 **토큰 마스킹 변형**이 이미 유사한 효과를 낼 수 있어".
- **Verdict: partially confirmed.** The paper exists and is a masking method, but TRM is *sequence-level* masking, and the paper explicitly argues that token-level masking does **not** restore the guarantee. IcePop is not mentioned anywhere in the paper.
- **Quotes:**
  - Abstract: "We propose Trust Region Masking (TRM), which masks entire sequences violating the trust region, enabling the first non-vacuous monotonic improvement guarantees for long-horizon LLM-RL."
  - §4.1: "Token masking preserves vacuous bounds. Zeroing the gradient of outlier tokens (where |log ρt| > τ) prevents immediate gradient explosion but does not change the underlying divergence between πroll and πθ."
  - Remark 4.1: "Token masking does not restore the guarantee ... Token masking therefore alters the optimization direction without restoring the monotonic-improvement guarantee."
  - §4.2 / Alg. 1: mask M(x,y) = I[max_t D_KL(c_t) ≤ δ]; "The gradient is normalized by the total batch size N (not the accepted count), so rejected sequences contribute zero gradient. This is a rejection sampling mechanism".
- **Correction:** replace "토큰 마스킹 변형" with "마스킹 변형(IcePop는 토큰 단위, TRM은 max_t D_KL(c_t) ≤ δ를 위반하는 시퀀스 전체를 배제하는 시퀀스 단위)". Upgrade tag to [V] and add venue "ICML 2026".
- **Other facts the report may use:** classical bound O(T²) vs. Pinsker-Marginal O(T^{3/2}), Mixed O(T), Adaptive bound; all bounds depend on D_KL^{tok,max}, "a sequence-level quantity uncontrollable by token-level PPO clipping". Remark 4.3: masking "does not close a population-level gap" — the guarantee for the masked objective (Prop. G.1) carries an explicit masking-rate term r = Pr_{πroll}[reject]. Appendix: "Length-Neutral Trust Region Masking" (Table 2, length-bias properties). KL is computed exactly over the full vocabulary from stored rollout logits. Empirical part is "training stability on mathematical reasoning benchmarks" (not checked numerically here).

---

## 2609.18063 — The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction

- **Exists:** yes. Yu Lin, Yiming Wang, Runyuan Cai, Hanze Liu, Xiaodong Zeng. v1 2026-09-16, v2 2026-09-17. cs.AI. No comment / journal_ref. Preprint, no venue found.
- **What it is:** a systems paper for "Edge0", an SSD-streaming MoE inference engine with a trained "prerouter" and an *unmerged* recovery LoRA; 35B MoE at ~20 tok/s in 3 GiB active memory on a 24 GB machine. Routing family named as "Softmax-top-k (Qwen3.6)"; the 35B tier is not otherwise named in the text.
- **Report claim** (Idea on QA-LoRA merge, tag [U]): "'델타가 4-bit 그룹 스텝 아래라 66–98%가 지워진다'는 2026 시스템 논문 보고는 이 세션에서 ID를 확인하지 못했다 ([2609.18063] [U])"; and in the caveats list: "2609.18063(LoRA 머지 소거)은 존재·내용을 PDF로 확인하기 전까지 인용하지 말 것."
- **Verdict: confirmed.** The ID is right, it is a 2026 systems paper, and the numbers match exactly (100−34 = 66%, 100−2 = 98%).
- **Quotes (§3.3 "Recovery LoRA: Unmerged by Design"):**
  - "The alternative, merging the delta into the dequantized weight and re-quantizing to 4-bit, fails on arithmetic rather than implementation: LoRA deltas (RMS 10⁻³) sit below the 4-bit group step, so requantization erases most of the weight-level delta (34% of the effect survives on an attention projection, 2% on a dense projection), and at the logits level 18% survives."
  - Footnote 2: "Retention at the logits level is measured as 1 − ‖U−M‖/‖U−B‖, which is 0.18 for the merged model, where U = unmerged, M = merged, B = base, first-token logits on a fixed prompt; weight-level retention is Σ(actual · δ)/Σδ² per target."
  - Contribution 3: "Merging an adapter into an int4 base and re-quantizing destroys most of its effect; we show that serving it as a parallel delta avoids this at negligible cost (§3.3)." Unmerged path "costs 42 MB of adapter weights and no measurable decode time".
- **Correction:** the [U] hedge can be removed: cite as "[2609.18063] (Lin et al., 2026, arXiv preprint; Edge0)". Precision: 66% erased on *an attention projection*, 98% on *a dense projection* (weight level); 82% erased at the *logits* level. Caveat to keep: this is a one-prompt, first-token-logits, two-projection measurement reported in a paragraph plus footnote, with no error bound, no table, and no model named — not a systematic study. The "인용하지 말 것" line in the caveats list should be dropped for this ID.

---

## 2505.23786 — Mind the Gap: A Practical Attack on GGUF Quantization

- **Exists:** yes. Kazuki Egashira, Robin Staab, Mark Vero, Jingxuan He, Martin Vechev. v1 2025-05-24, v2 2025-06-03. cs.CR.
- **Venue:** ICML 2025 (arXiv comment "ICML 2025"; text header: "Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025").
- **Report claim** (tag [S]): "Egashira et al.은 preimage box를 계산해 PGD로 FP 가중치를 수리하는 공격을 ... GGUF 9종 ([2505.23786, ICML 2025]) ... 확장했다."
- **Verdict: partially confirmed.** Existence, first author, ICML 2025, "9 GGUF types", and the use of PGD for the repair step are all correct. The *mechanism* description is wrong for this paper: for GGUF k-quants the authors state that exact preimage intervals are infeasible and instead use *error-based interval estimation* (intervals derived from the observed quantization error, heuristically widened).
- **Quotes:**
  - Notation (§3): "we consider nine widely used k-quant data types: Q2_K, Q3_K_{S, M, L}, Q4_K_{S, M}, Q5_K_{S, M}, Q6_K." (all k-quants; reference llama.cpp release b3612)
  - §4.1 "Limitations of Exact Intervals for GGUF": "... it infeasible to compute exact intervals for each weight."
  - §4.2: "Instead of using intractable constraints that always preserve quantization, we propose tractable intervals that are likely to [preserve it] ... we derive these intervals directly from the distance [between full-precision and dequantized weights]."
  - Table 4: "The error-based vs exact interval results on zeroshot quantizations. With 3−4× larger interval, slightly larger Code Security is achieved with the exact interval. However the security with error-based interval is already as [high]".
  - B.6: "two epochs for repair (removal) using Projected Gradient Descent (PGD)".
  - Abstract: "insecure code generation (Δ=88.7%), targeted content injection (Δ=85.0%), and benign instruction refusal (Δ=30.1%)" on "three popular LLMs" (Qwen2.5-1.5b/3b, Llama3.1-8b).
  - Gaussian-noise defense (Fig. 4 + text): "while Gaussian noise can be a promising defense even for k-quants, the noise level has to be calibrated separately for each model" (σ=1e-3 for Qwen2.5-3b, σ=1e-4 for Llama3.1-8b).
- **Correction:** say "GGUF k-quant 9종 ([2505.23786, ICML 2025])에는 정확한 preimage 구간 계산이 불가능해 양자화 오차 기반 근사 구간(error-based interval)으로 제약을 대체했다". Upgrade tag to [V]. Note that this paper *does not* cover I-quants or AWQ/GPTQ, and reports Gaussian noise as a per-model-calibrated but workable defense for k-quants — the opposite of what 2605.15152 finds for outlier injection.

---

## 2605.15152 — Widening the Gap: Exploiting LLM Quantization via Outlier Injection

- **Exists:** yes. **Xiaohua Zhan**, Kazuki Egashira, Robin Staab, Mark Vero, Martin Vechev (ETH Zurich). v1 2026-05-14, v2 2026-06-03. cs.LG. arXiv comment: none; paper header line 50 reads "Preprint."
- **Venue:** NOT ICML 2026 main conference. OpenReview: `VSxP0tARSE` venue "ICML 2026 AIWILD" (ICML.cc/2026/Workshop/AIWILD, with camera-ready invitation); `19ETSXISYL` venue "AI4GOOD Workshop 2026 Regular" (ICML.cc/2026/Workshop/AI4GOOD, camera-ready); `OVlOj6bcY0` venue "ICLR 2027 Conference Submission" (under review). Also note: v1 was posted after ICML 2026 main-track decisions, consistent with workshop-only status.
- **Report claim** (tag [S]): "Egashira et al.은 ... outlier 주입으로 AWQ/GPTQ까지 ([2605.15152, ICML 2026]) 확장했다. Gaussian 노이즈 방어는 outlier 주입(Δ 확대)에 깨졌다."
- **Verdict: partially confirmed.** Technical content is correct (outlier injection; AWQ/GPTQ and more; Gaussian-noise defense ineffective). Two errors: (i) venue "ICML 2026" should be "ICML 2026 workshop (AIWILD / AI4GOOD); ICLR 2027 submission; arXiv preprint"; (ii) first author is Zhan, not Egashira.
- **Quotes:**
  - Abstract: "we introduce the first quantization-conditioned attack that consistently induces malicious behavior that can be triggered by a broad range of advanced quantization techniques, including AWQ, GPTQ, and GGUF I-quants. Our attack exploits a simple property shared by many modern quantization methods: large outliers can cause other weights to be rounded to zero."
  - §3: targets "DIOB: HQQ, SINQ, GGUF k-quant, and DDOB: GPTQ, AWQ, AutoRound, GGUF i-quant"; "while, to our knowledge, no prior attack has successfully targeted this family [DDOB], our attack is effective against ... GPTQ, AWQ, AutoRound, and GGUF i-quant."
  - §5.3 Gaussian Noise Defense: "In Figure 2, we show that this defense does not apply to our attack. As the noise level increases, utility begins to drop clearly from σ = 10⁻³, while the attack remains effective. Further increasing σ reduces ASR only together with a complete utility collapse (σ = 5·10⁻³). ... prior attacks rely on tightly constrained weight intervals, whereas ours is driven by large inserted outliers that are unaffected by small noise." (Qwen2.5-7B-Instruct, jailbreak)
  - §5.3 Table 3 (Llama3.1-8B-Instruct, jailbreak, GPTQ 4/8-bit): rescaling or zeroing detected outliers cuts ASR "from > 95% to < 2%" at attacker scale c = 512 but "fail[s]" at c = 4096 (ASR ≈ 44–47% even in BF16); authors conclude "a model with such detected anomalies should be treated as untrusted, rather than repaired through rescaling." Detection via a statistical test on the weight distribution (§C.2).
- **Correction:** "Zhan et al.은 outlier 주입으로 AWQ/GPTQ/AutoRound/GGUF i-quant까지 ([2605.15152, ICML 2026 workshop (AIWILD/AI4GOOD); ICLR 2027 under review])". Keep the Gaussian-noise sentence. For the report's dither-as-defense idea: the paper's own explanation — the attack is "driven by large inserted outliers that are unaffected by small noise" — implies dither (small additive noise before rounding) would face the same limitation against outlier injection, while it may still be relevant against interval-constrained attacks (2405.18137 / 2505.23786). The paper does not evaluate dither.
