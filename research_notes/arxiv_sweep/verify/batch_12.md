# Citation verification — batch 12 (2510.11696, 2601.18150, 2601.14243, 2602.13953)

All four IDs are cited in ONE sentence of the report (idea on quantized-rollout RL):

> QeRL(NVIDIA, ICLR 2026)은 NVFP4+LoRA 롤아웃에서 "양자화 노이즈가 정책 엔트로피를 올려 탐색을 향상"한다고 주장하고 IS 보정은 쓰지 않는다 ([2510.11696] [V: README]). FP8-RL([2601.18150] [S]), Jet-RL([2601.14243] [S]), QuRL([2602.13953] [S]), HiFloat4/Rollout-ResQ(...), AIS(...)가 서로 다른 처방을 낸다. **엔트로피, KL(μ‖π), 보상을 한 런에서 함께 보고한 논문이 없다.**

Claims checked: (a) QeRL = NVIDIA, ICLR 2026, NVFP4+LoRA, entropy/exploration claim, no IS correction; (b) the other three give *different prescriptions*; (c) negative claim: no paper reports entropy + KL(μ‖π) + reward together in one run.

Summary: (a) confirmed; (b) confirmed; (c) **incorrect as written** — FP8-RL Fig. 11 reports exactly those three (plus accuracy, response length, grad-norm) in one 600-step run.

---

## 2510.11696 — QeRL: Beyond Efficiency – Quantization-enhanced Reinforcement Learning for LLMs

- Exists: yes. First author Wei Huang; 14 authors, affiliations NVIDIA (1), MIT (2), HKU (3), THU (4). Published 2025-10-13 (v1 only).
- Venue: **ICLR 2026** (poster page iclr.cc/virtual/2026/poster/10006405; GitHub README "About": "[ICLR 2026] QeRL enables RL for 32B LLMs on a single H100 GPU"). arXiv comment only says "Code is available at https://github.com/NVlabs/QeRL".
- Verdict: **confirmed**.

Checks:
- NVFP4 + LoRA: abstract — "QeRL addresses these issues by combining NVFP4 quantization with Low-Rank Adaptation (LoRA), accelerating rollout phase of RL".
- Entropy/exploration claim: abstract — "our findings show that quantization noise increases policy entropy, enhancing exploration, and enabling the discovery of better strategies during RL." Sec. 3 ("Quantization encourages exploration"): "We discover that quantization inherently increases the sampling entropy ... of the policy during deployment (shown in Fig.5)". Fig. 5 compares entropy of NVFP4/MXFP4/NF4/FP16-LoRA (r32) on GSM8K; App. H Fig. 14 Qwen2.5-14B-Instruct entropy "remains consistently higher than that of LoRA throughout the RL process". This is an EMPIRICAL claim (entropy curves), motivated by analogy to parameter-noise exploration (Eq. 5: Q(θ) − θ = Δε); no theorem.
- No IS (mismatch) correction: confirmed by design. Intro: "FlashRL ... uses quantized rollout models ... However, precision mismatches between the rollout model and logits model (e.g., 8-bit vs. 16-bit) require importance sampling to correct discrepancies, necessitating both 8-bit and 16-bit models to run simultaneously, which increases memory usage. To overcome these limitations, we focus on lower-bit quantization while avoiding duplicate models in memory." Sec. 2/Fig. 2: "QeRL uses NVFP4 quantization for LLM weights and integrates a Marlin-based approach in both rollout and prefilling stages" — i.e. rollout and log-prob forward share the same NVFP4 weights, so no TIS/MIS is applied. The only other "importance sampling" mention (App. J) is the ordinary PPO/GRPO ratio on the training side.
- What it reports per run: reward (Fig. 4, DAPO/GRPO) and entropy (Fig. 5, Fig. 14). It does NOT report any KL(rollout‖train) — the word KL appears only in the GRPO/DAPO background and in Tab. 4 setup ("without using entropy or KL losses"). So the report's negative claim holds for QeRL.
- Other numbers: >1.5× rollout speedup; 32B RL on one H100 80GB; GSM8K 90.8%, MATH500 77.4% (7B, matching full-param FT); AQN noise schedule Z_start=1e-2, Z_end=5e-4; LR 1e-5 (QeRL/QLoRA) vs 5e-6 (BF16 LoRA, "can not be larger ... or it will collapse"); policy updates per rollout 4 (GSM8K, off-policy) / 1 (BigMath, on-policy). Weight-only AWQ calibration on 256×2048 OpenThoughts sequences.
- Limitations (App. "Limitation analysis"): 3B–32B only, no >70B; math-reasoning benchmarks only, no code/general tasks.

## 2601.18150 — FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning

- Exists: yes. First author Zhaopeng Qiu; 7 authors, all NVIDIA (Beijing). Published 2026-01-26, updated 2026-04-10 (v2, "Added more FP8 end2end experiments"). Self-described "report"; implemented in verl (docs: verl.readthedocs.io/en/latest/low_precision/fp8.html).
- Venue: preprint / technical report (no venue stated, none found).
- Verdict: **partially_confirmed** — correct that its prescription differs (token-level TIS/MIS importance sampling, C=2), but it **contradicts** the report's "no paper reports entropy, KL(μ‖π) and reward in one run".

Checks:
- Prescription: abstract — "(iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants)"; Sec. 2.1.3 "w_TIS(a|s) = clip(w(a|s), C)", C=2 for all FP8 configs; "it is recommended to pair FP8 rollout with importance-sampling-based rollout correction (e.g., token-level TIS)". For MoE it additionally recommends end-to-end FP8 (train+rollout) because BF16-train+FP8-rollout with TIS alone "crashed" (Sec. 2.4.2).
- Metric definition (Sec. 2.2.1): "(iv) mismatch KL, the KL divergence D_KL(π_θ^FP8 ‖ π_θ) between the rollout policy π_θ^FP8 (inference) and training policy π_θ" — this is KL(rollout ‖ train) = the report's KL(μ‖π).
- **Contradicting passage** (Sec. 2.4.3, Qwen3-30B-A3B-Base, FP8 recipe ablation): "Figure 11 presents training dynamics across six metrics. The hybrid recipe (orange) closely tracks the BF16 baseline (purple) across all metrics—validation accuracy, reward, response length, entropy, and mismatch KL—throughout the full 600-step training run. In contrast, the pure E4M3 recipe (green) trains stably for the first ~500 steps but then collapses: the gradient norm spikes sharply (from ~0.3 to ~1.75), accompanied by a surge in entropy and mismatch KL (>0.25)." → entropy, KL(μ‖π) and reward are reported together in one run. Caveat: it is an FP8-training-recipe ablation (E4M3 vs hybrid) under end-to-end FP8, not a study relating rollout-noise entropy to exploration; the main figures (Figs. 2, 4, 10, 15) list accuracy/reward/response length/mismatch KL without entropy.
- Other numbers: dense Qwen3-8B-Base rollout speedup ~10–20%, MoE Qwen3-30B-A3B ~30–50% (abstract: "up to 44%"); FP8 W8A8 without TIS "exhibits noticeable accuracy degradation" (Fig. 2); MoE mismatch KL ordering "FP8 rollout-only > FP8 end-to-end > BF16", rollout-only spike "above 5" around step 700 crashing the run; FP8 training gave no significant training-side speedup on 30B MoE ("primary benefit ... is the reduction in train–inference mismatch"); E4M3 collapse traced to MoE fc1 grad tiles exceeding range (5% avg, 21% layer-0, p99 26%→41% at steps 571–576).

Suggested correction to the report: replace "엔트로피, KL(μ‖π), 보상을 한 런에서 함께 보고한 논문이 없다" with something like "FP8-RL의 레시피 ablation(Fig. 11, Qwen3-30B-A3B, 600 step)만 엔트로피·mismatch KL·보상을 한 런에서 함께 그리지만 E4M3 vs hybrid 학습 레시피 비교일 뿐, 롤아웃 양자화 노이즈의 엔트로피 효과와 KL(μ‖π) 편향을 같은 런에서 연결한 논문은 없다."

## 2601.14243 — Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow

- Exists: yes. First author Haocheng Xi; 10 authors, NVIDIA (1), MIT (2), UC Berkeley (3), independent (4). Published 2026-01-20, v2 2026-01-24. Comment "11 pages, 6 figures, 4 tables". Text: "We will release our code and pre-trained models when less anoynomous" (under review).
- Venue: preprint/unknown (no OpenReview or acceptance found by web search).
- Verdict: **confirmed** (cited only as giving a different prescription).

Checks:
- Prescription: unified FP8 precision flow for training AND rollout, i.e. make RL on-policy; no IS correction, no inter-step calibration. Sec. 4: "the root cause of BF16-train-FP8-rollout training failure is the inconsistency of the precision flow between training and rollout, making the RL training effectively off-policy ... we propose enforcing an unified FP8 precision flow between training and rollout". Sec. 3.3 on TIS: "Though in some cases and some studies like Truncated Importance Sampling (TIS), BF16-train-FP8-rollout does not result in much performance degradation, we find that this conclusion is very fragile and depends on specific dataset and model settings."
- Failure modes claimed (empirical): Qwen2.5-7B/MATH/GRPO accuracy drops once rollout length exceeds 8K; "When we further increase the rollout length to 16K, FP8 rollout's accuracy collapses after only 20 steps"; Qwen3-8B on GSM8K fine but Qwen3-8B-Base on harder tasks diverges (Fig. 4). Also notes "For shorter generation lengths, these discrepancies ... may even be partially beneficial for training" (a hedge in the direction of QeRL's claim).
- Metrics reported: accuracy/reward curves only; no entropy curve ("entropy" appears once, describing DAPO Clip-Higher) and no mismatch-KL curve ("KL" only as GRPO loss coefficient 1e-3). Consistent with the report's negative claim.
- Numbers: up to 33% rollout speedup, up to 41% training speedup, 16% end-to-end over BF16; PTQ calibration "can take tens of minutes even for small 8B LLMs". Cites QeRL as [57] and TIS as [47].

## 2602.13953 — QuRL: Efficient Reinforcement Learning with Quantized Rollout

- Exists: yes. First author Yuhang Li (Yale, intern at NVIDIA Research); NVIDIA Research / Yale / USC. Published 2026-02-15.
- Venue: **ICLR 2026** (arXiv comment "Accepted to ICLR 2026"; text header "Published as a conference paper at ICLR 2026").
- Verdict: **confirmed** (cited only as giving a different prescription).

Checks:
- Prescription: quantized old actor as behavior policy in decoupled PPO + TIS, plus **Adaptive Clipping Range (ACR)** ("dynamically adjusts the clipping ratio based on the policy ratio between the full-precision actor and the quantized actor, which is essential for mitigating long-term training collapse"; shrinks lower bound by r_{i,t}, keeps fixed upper (1+ε)), plus **Update-Aware Quantization (UAQ)** via invariant scaling to make weight updates exceed quantization granularity. Distinct from QeRL (no correction), FP8-RL (TIS only), Jet-RL (unified precision).
- Reported dynamics: training reward and token clipped fraction (Fig. 2), KL(π_behav ‖ π_prox) and max proximal-to-behavior ratio (Fig. 3): "at long training steps (e.g., > 1000 steps) ... the KL divergence increases from 0.002 to 0.025, which is 12× higher"; max ratio "up to 10^5". No entropy curves anywhere ("entropy" has zero hits). So the negative claim holds for QuRL.
- Table 2 (AIME 2024, Qwen2.5-7B-Math, DAPO 200 steps): BF16 Avg@1 33.33 / Avg@32 31.67; INT8: RL 0.00/0.001, FlashRL 26.66/30.29, QuRL w/o UAQ 33.33/30.63, QuRL w/ UAQ 33.33/31.25; FP8: RL 0.00/0.003, FlashRL 30.00/32.60, QuRL w/o UAQ 36.66/33.12, QuRL w/ UAQ 33.33/33.27. Table 1 (GSM8K PPO, Qwen2.5-0.5B): QuRL gap ~2% INT8, ~1% FP8 vs FlashRL 4% INT8. Rollout 20–80% faster (INT8/FP8).
- Note: the paper says TIS "works well under 500 steps" but fails >1000 steps — a useful boundary for the report's discussion.
