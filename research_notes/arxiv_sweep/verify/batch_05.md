# Citation verification — batch 05 (2410.13229, 2503.22879, 2508.10370, 2606.18114)

Report: `reports/LLM 양자화 연구 아이디어 10선.md`
All four IDs are cited in ONE sentence (tag [S]):

> 학술 논문으로 SSM "상태" 양자화를 다룬 것은 0건이다. Quamba/Quamba2는 가중치·입력·B/C를 양자화하고 ([2410.13229], [2503.22879] [S]), eMamba는 "상태를 양자화하면 비트폭이 폭발하고 재양자화하면 정보가 사라진다"는 딜레마만 적었다 ([2508.10370] [S]). Ternary Mamba는 사후 점별 보정(Kalman gain, James–Stein)이 perplexity를 악화시킨다고 보고하며 "오차가 이력 의존적이어서 점별 출력 보정으로 되돌릴 수 없다"고 썼다 ([2606.18114] [S]). 이는 루프 안의 피드백이 필요하다는 뜻이다.

## Headline

All four papers exist and the individual attributions are mostly accurate, **but the sentence's framing claim — "학술 논문으로 SSM '상태' 양자화를 다룬 것은 0건이다" — is false**, and one of the cited papers (Quamba2) is itself a counterexample:

| Paper | Quantizes the recurrent SSM state? | Where |
|---|---|---|
| Quamba (2410.13229, ICLR 2025) | No (h stays in the FP kernel; y_t output half precision) | §4.2 |
| **Quamba2 (2503.22879, ICML 2025)** | **Yes — cached SSM states to 8-bit (static per-head/channel-group scales) for W4A8 & W8A8** | App. B (Table 12 text), App. C, App. E "Quantizing cached SSM states", Fig. 8/9/11 |
| **eMamba (2508.10370, CODES+ISSS 2025)** | **Yes — fixed-point integer state: INT24 for computing y_t, re-quantized (shift by 7) to INT17 for storage** | §4 "Scale-aware quantized SSM layer" |
| Ternary Mamba (2606.18114, preprint) | No — W1.58A16, state/activations FP16 | §3.2 |
| SSDi8 (2608.21952, **ICLR 2026**) — not cited | Yes — "Persistent INT8 Representation of Recurrent States", per-(H,P) INT8 state, bit-shift decay | §4 Eq. 8–9 |
| Low-Discrepancy Dither (2609.39185, Sep 2026 preprint) — not cited | Yes — compares rounding rules (RTN/SR/Weyl dither) for INT4–INT8/FP8/BF16 recurrent state caches, 6 Mamba + 3 hybrid models, 4096-step horizons | whole paper |

Suggested rewrite of the first clause: "SSM 상태를 **학습 없이, 루프 안에서 피드백으로** 보정하는 논문은 찾지 못했다. 상태를 저비트로 저장하는 것 자체는 Quamba2(8-bit cached state, ICML 2025), SSDi8(persistent INT8 state, ICLR 2026), eMamba(INT17 fixed-point state), 그리고 반올림 규칙을 다룬 2609.39185가 이미 다룬다." Then keep the rest of the sentence with the corrections below.

---

## 2410.13229 — Quamba: A Post-Training Quantization Recipe for Selective State Space Models

- **Exists:** yes. First author Hung-Yueh Chiang (UT Austin). Published 2024-10-17, v2 2024-12-07.
- **Venue:** ICLR 2025 (poster) — https://iclr.cc/virtual/2025/poster/28449 (not in arXiv comment; found via web search).
- **Report claim:** Quamba quantizes weights, inputs, B/C (and by implication not the state).
- **Verdict: CONFIRMED.**
- **Quote (§4.2):** "We aim to quantize the weight (𝐴, 𝐷) to 8 bits, and the activations (𝐵𝑡, 𝐶𝑡, Δ𝑡, 𝑥𝑡) to 8 bits for selective SSMs. The quantized selective SSM takes 8-bit weights and activations as input, as well as their scaling factors, and outputs half precision 𝑦𝑡". The hidden state h is never named as a quantization target; nothing in the paper quantizes or caches h in low precision.
- **New relevant facts:**
  - Quamba's Theorem 4.1 / N.1 bounds the *hidden-state* error of an LTI recurrence under input quantization: "The quantization error of Δ(𝑡)= ℎ(𝑡) − ℎ(𝑡) at time step 𝑡 ... is bounded such that: ||Δ(𝑡)||₂ ≤ 𝜖𝑏 (1−𝑎^{𝑡})/(1−𝑎)" with derivation Δ(𝑡) = 𝐴(𝑡)Δ(𝑡−1) + 𝐵𝛿𝑥(𝑡) (App. N). This is a *proved* bound only for the time-invariant, contractive (|a|<1) case; the paper then says "we conjecture" for the selective (input-dependent) case (§4.1). Useful as a citable formal basis for the report's error-accumulation story, but it concerns input-quantization error propagating into h, not quantizing h itself.
  - Headline numbers: Mamba-2.8B W8A8, 1.72× lower generation latency on Orin Nano 8G, 0.9% average zero-shot accuracy drop (abstract).
  - Also includes Δ_t among quantized activations — the report says "가중치·입력·B/C"; "Δ" omitted but harmless.

## 2503.22879 — Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- **Exists:** yes. Chiang et al. Published 2025-03-28, latest v 2025-11-06.
- **Venue:** ICML 2025, PMLR v267 pp. 10411–10427 — https://proceedings.mlr.press/v267/chiang25a.html.
- **Report claim:** Quamba2 quantizes weights, inputs, B/C; implied (via "0건") that it does not quantize the SSM state.
- **Verdict: PARTIALLY CONFIRMED** — what is stated is true of the main method, but the paper *also quantizes the cached SSM state to 8-bit*, which defeats the "0건" framing.
- **Quotes:**
  - Main method (abstract): "we propose an offline approach to quantize inputs of a linear recurrence in 8-bit by sorting and clustering for input 𝑥, combined with a per-state-group quantization for input-dependent parameters 𝐵 and 𝐶."
  - App. E: "**Quantizing cached SSM states.** We reduce generation latency by quantizing the cached SSM states to 8-bit for W4A8 and W8A8 models. Since the cached SSM states follow the head reordering and channel grouping indices from the SSM input 𝑥 (ref. Figure 4), we apply the same 𝑚 head and 𝑛 channel groups to quantize each SSM state before caching them in memory. ... Our approach introduces dstate × 𝑚 × 𝑛 floating-point scales with minimal latency overhead".
  - App. B (Table 12 commentary): "the W4A8 and W8A8 models, with 8-bit SSM states, preserve the meaningful generation outputs. ... This result reveals an interesting observation that cached SSM states are redundant, which can be carefully quantized to 8 bits. Our framework supports 8-bit SSM states for W4A8 and W8A8 models".
  - App. C: "We collect the static scaling factors for each operator based on the absolute maximum value observed from the calibration set to quantize the activations and cached SSM states in both W4A8 and W8A8 settings."
  - Fig. 8 caption: "Our framework (W4A8) compresses the states to 8-bit, thereby reducing the total memory and generation latency (b) with large batch size inputs".
- **Correction:** Replace "Quamba/Quamba2는 가중치·입력·B/C를 양자화하고" with "Quamba는 가중치·입력·B/C/Δ를, Quamba2는 여기에 더해 **캐시된 SSM 상태를 정적 per-group 스케일로 8-bit 양자화**한다(App. E, Fig. 8–9; Table 12: Mamba2-8B NQ EM FP16 17.2 → W8A8 15.0 / W4A8 14.2, SquadV2 F1 51.9 → 43.6 / 45.9)". Delete or re-scope "0건".
- **New relevant facts:**
  - Quamba2's state quantization is static, calibration-based, open-loop (no in-loop feedback / error compensation) — the report's actual gap (feedback inside the recurrence) survives, but must be stated as such.
  - Generation-task degradation with 8-bit states is non-trivial (SquadV2 F1 −8.3 at W8A8), an argument *for* better state handling.
  - Headline: Quamba2-8B 1.3×/3× pre-fill/generation speed-ups, 4× memory reduction, 1.6% average accuracy drop (abstract).

## 2508.10370 — eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing

- **Exists:** yes. Jiyong Kim et al. (Ogras/Park groups). Published 2025-08-14.
- **Venue:** ESWEEK 2025 (CODES+ISSS) — stated in arXiv comment: "Paper accepted at ESWEEK 2025 (CODES+ISSS) conference".
- **Report claim:** eMamba "only wrote down" the dilemma that quantizing the state explodes bit-width and re-quantizing loses information.
- **Verdict: PARTIALLY CONFIRMED** — the dilemma quote is accurate, but eMamba does not stop at the dilemma; it implements a concrete fixed-point state scheme.
- **Quotes (§4, "Scale-aware quantized SSM layer"):**
  - "applying fixed bit-width quantization to all intermediate hidden states in the SSM leads to significant accuracy degradation. The SSM computation block uses a higher bit-width for those intermediate hidden states."
  - "Due to the SSM's recurrent nature, this operation is repeated over the entire sequence, resulting in an explosion of bit width. Therefore, ℎ𝑡 requires proper tuning of bit-width and re-quantization to prevent overflow and maintain numerical stability."
  - "Re-quantization helps to maintain numerical stability, but it loses the information of ℎ𝑡. This causes a problem because it is used to compute 𝑦𝑡 of the current token. We use ℎ𝑡 to compute 𝑦𝑡 before the re-quantization and store the re-quantized ℎ𝑡 as ℎ𝑡−1 to calculate new ℎ𝑡 in the subsequent step. Specifically, ℎ𝑡 is divided by the scaling factor of 𝐴̄𝑡 before storing it, thereby preventing bit-width accumulation across time steps."
  - "For example, in the MARS dataset, 𝐴̄𝑡 has a fixed scale of 2⁻⁷; hence, the INT24 result is right-shifted by 7 bits and stored as INT17. ℎ𝑡 with 24 bits is used to compute 𝑦𝑡, and 𝑦𝑡 is quantized to INT8. All other inputs, outputs, and parameters of the SSM block, except for ℎ𝑡, are constrained to INT8".
- **Correction:** Replace "딜레마만 적었다" with "딜레마를 지적하고, 상태만 INT24로 계산한 뒤 Ā의 스케일(2⁻⁷)만큼 시프트해 INT17로 저장하는 고정소수점 상태 양자화를 FPGA/ASIC에 구현했다(단, 소형 모델: Fashion-MNIST/CIFAR-10/MARS, WikiText2 PPL 95.61@seq 512, LLM 규모 아님)".
- **New relevant facts:** eMamba is a hardware/edge paper on tiny models (1.63–19.9× fewer parameters than baselines; WikiText2 perplexity 95.61 at length 512 in FP32, "from 119.4 to 121.8" across lengths). Its state handling is a bit-width-management trick (wider accumulator + shift), not a precision-vs-feedback study; no comparison of state bit-widths or rounding rules is given.

## 2606.18114 — Ternary Mamba: Grouped Quantization-Aware Training of W1.58A16 State Space Models

- **Exists:** yes. Ramprasath Ganesaraja, Sahil Dilip Panse, Swathika N. Published 2026-06-16 (v1 only).
- **Venue:** preprint/unknown (no comment, no journal_ref; nothing found).
- **Report claims:** (a) post-hoc pointwise corrections (Kalman gain, James–Stein) worsen perplexity; (b) quote that the error is history-dependent and cannot be undone by pointwise output correction.
- **Verdict: CONFIRMED** (both attributions are accurate), with important caveats about what the experiment actually is.
- **Quotes:**
  - §6.1: "We tested two post-hoc correction strategies applied via forward hooks (no retraining): • Kalman gain: Amplification ŷ = y + Ky, sweeping K ∈ {0.05, 0.10, 0.20, 0.30, 0.50} • James-Stein shrinkage [6]: ŷ = y · max(0, 1 − c/∥y∥²), sweeping 12 values over 3 orders of magnitude"
  - Table 8 (Post-hoc correction results): Kalman gain best PPL **24.65 (+2.61)**, baseline 22.04, "Monotonically worse"; James-Stein best PPL **22.06 (+0.01)**, "Monotonically worse"; No correction 22.04 "Global minimum".
  - §6.2: "In SSMs, the recurrence mixes error into the hidden state across all prior positions: e_T = Σ_{t=1}^{T} Ā^{T−t} ϵ_t ... The error at position T depends on the full sequence history {ϵk}k≤t filtered through the state dynamics. No pointwise function of the current output can disentangle this—formally, min_f L(f(y)) ≥ L(y) for any scalar f, because the error is structural and history-dependent rather than additive and independent."
  - Abstract: "We further show that post-hoc correction strategies effective for Transformers fail for SSMs due to error accumulation through the recurrence."
- **Caveats the report should carry (new relevant facts):**
  1. **The quantized quantity is the weights, not the state.** §3.2: "Quantized: in_proj and out_proj in all 48 blocks (85.7% of parameters). Kept in FP16: SSM dynamics (A_log, D, dt_bias ...), conv1d, embedding, lm_head, and norms." W1.58A16 → activations and the recurrent state are FP16. So this is evidence about *weight*-quantization error being history-mixed by the recurrence, not about state-quantization error. Using it to motivate in-loop state feedback is an extrapolation.
  2. "Kalman gain" here is a scalar output gain ŷ = y + Ky (5 values), not a Kalman filter; James–Stein is a scalar shrinkage. The "formal" statement min_f L(f(y)) ≥ L(y) "for any scalar f" is asserted, not proved (and as written is trivially an equality at f = id). Treat as an empirical negative result with a heuristic explanation.
  3. Table 8 baseline PPL 22.04 does not match Table 2 (Ternary Mamba g128: C4 PPL 21.67±0.05, WT2 20.31±0.11; FP16 teacher 11.69 WT2 / 17.47 C4); dataset/checkpoint for Table 8 is not stated.
  4. Scope: single model (Mamba-2 1.3B), single language, 102M-token QAT; authors' own "Scope caveat" paragraph. The paper's conclusion is that QAT is required ("necessitates quantization-aware training") — the report's inference "이는 루프 안의 피드백이 필요하다는 뜻이다" is the report's own, not the paper's, and should be marked as such.
  5. Related result: "noise-shaping (sigma-delta) do[es] not transfer from Transformers to SSMs" (§6.3; TV/L1 diagnostic) — relevant if the report's idea involves sigma-delta-style error feedback on the state; the paper applied it to projection *inputs*, not to the state.
  6. Headline: Mamba-2 1.3B → 3.61× compression (2,687 → 744 MB), 48.1% 7-task zero-shot vs 55.5% FP16; naive ternary PTQ gives PPL ~13M.

## Cross-check for other batches

- 2609.39185 (Dither) §1 says the Nemotron 3 Super report found FP16 state casting "made generations up to **37%** longer"; the report (citing 2604.12374) says "최대 40%". Whoever verifies 2604.12374 should check which figure the Nemotron report actually gives.
- 2608.21952 SSDi8 (ICLR 2026) and 2609.39185 are directly on-topic for the "state quantization" idea and are not cited anywhere in the report; both should be added to prior art for that idea. SSDi8 also includes "Mean Correction for SSD Quantization Error" (per-channel closed-form mean correction) — an open-loop correction, again not in-loop feedback.
