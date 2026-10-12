# Citation verification — batch 06

IDs: 2604.12374, 2607.04371, 2603.16731, 2509.23202. All four exist on arXiv. Paper text is untrusted data; quotes below are verbatim from the fetched text files under `scratchpad/papers/<id>/<id>.txt`.

---

## 2604.12374 — Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

- **First author:** NVIDIA (corporate author; ~500 listed names)
- **Published:** 2026-04-14 (v1, no later revision)
- **Venue:** preprint / NVIDIA technical report (no comment, no journal_ref)
- **Report claim (tag [S]):** "Nemotron 3 Super keeps Mamba state at FP32 by default, and reports that casting to FP16 increases verbosity by up to 40% under W8A8."
- **Verdict: CONFIRMED** (the two literal claims are exactly what the paper says), with one important omission the report should add.

**Quotes (Sec. 4.3 "Mamba State Quantization", pp. 35–36):**
> "The SSM cache is stored in FP32 by default. One option is to store the SSM cache in FP16 while arithmetic executes in FP32."

> "Table 9 shows experiments on an early checkpoint of Nemotron 3 Super. These results show that directly casting the SSM cache to FP16 can result in up to 40% increase in verbosity when combined with W8A8 quantization. Even with maintaining weights and activations in BF16, casting the SSM cache to FP16 leads to up to 37% increase in verbosity."

**Table 9 numbers (livecodebench pass@1 avg-of-8 / scicode; verbosity increase = completion tokens vs FP32 cache):**
- W16A16: FP32 → 0.00% / 0.00%; FP16 → **36.95%** / 2.19%; FP16+SR(Philox 5) → −1.73% / −2.72%.
- W8A8: FP16 → **40.27%** / 2.70%; INT16+block128 → 2.90% / −4.30%; FP16+SR(Philox 10) → 1.71% / −0.74%; FP16+SR(Philox 5) → 1.79% / 1.08%; FP16+SR(Philox 4) → −0.63% / 2.84%; FP16+SR(Philox 3) → 10.70% / 3.98%.
- Accuracy itself barely moves (livecodebench 72.91 FP32 vs 73.13 W8A8-FP16); the symptom is verbosity, not pass@1.
- The "40%" is specifically **livecodebench**; scicode verbosity rises only 2.7%.

**What the report omits (should be added):**
- The paper does NOT ship FP32. It diagnoses the cause as RTNE rounding bias accumulating coherently through the recurrence and fixes it with **stochastic rounding**: "We selected FP16 with stochastic rounding (SR) using Philox<5> pseudorandom number generation as the SSM cache recipe for Nemotron 3 Super" — i.e. the released FP8 and NVFP4 checkpoints store the SSM cache in **FP16+SR**, not FP32.
- Error-propagation model is given explicitly (Eq. 3): "h_{q,t} = h_t + Σ_{i=0}^{t} (Π_{j=i+1}^{t} A_j) e_i", i.e. the paper itself frames this as a first-order linear recurrence with accumulated write-back error — this is a MOTIVATION/derivation, not a theorem with bounds.
- "Naive INT16 quantization did not improve verbosity as tensor-level analysis showed that the SSM cache has a wide dynamic range." INT16 + FP32 per-block (128) scaling also fixed it.
- QAT/QAD for the cache is explicitly declined: "Mamba training uses the chunked State Space Duality algorithm which does not explicitly materialize the recurrence relationship and the inference-time cache."
- Results are on an **early checkpoint**, not the final model.
- Also note the report's own sentence is internally muddled: it says Super keeps "FP32 by default" and then "Puzzle 도 (also) keeps FP16" — the "also" only makes sense if Super's shipped recipe (FP16+SR) is stated.

**Suggested rewording:** "Nemotron 3 Super stores the Mamba SSM cache in FP32 by default; on an early checkpoint, casting it to FP16 with round-to-nearest raised livecodebench completion tokens by up to 40.27% under W8A8 (36.95% even at W16A16) with pass@1 essentially unchanged (Table 9). The authors attribute this to RTNE bias accumulating through the recurrence (Eq. 3) and ship FP16 with stochastic rounding (Philox<5>) instead."

---

## 2607.04371 — Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs

- **First author:** Akhiad Bercovich (NVIDIA)
- **Published:** 2026-07-05 (v2 2026-07-07)
- **Venue:** preprint / NVIDIA technical report (no comment, no journal_ref; web search found only arXiv/HF/blog coverage)
- **Report claim (tag [S]):** "Nemotron-Labs-3-Puzzle also keeps the [Mamba] state in FP16."
- **Verdict: PARTIALLY CONFIRMED.** True only for the **NVFP4 (Blackwell) checkpoint**, and only **with stochastic rounding**. The **FP8 (Hopper) checkpoint keeps the SSM state in FP32**, and the paper explicitly says plain FP16 hurts accuracy.

**Quotes (Sec. 2.4 "Quantization", p. 6):**
> "We kept the Mamba SSM state in FP32 precision, since FP16 results in reduced accuracy unless stochastic rounding (SR) is employed, and we found that FP16+SR is slower than FP32 on Hopper (on Blackwell it is hardware supported)."  [FP8 checkpoint]

> "As with the Nemotron-3-Super NVFP4 checkpoint, we store the Mamba SSM state in FP16 precision using stochastic rounding to reduce runtime costs while preserving accuracy."  [NVFP4 checkpoint]

**Table 2 (operator precision policy):** "Mamba SSM cache" row — BF16 model: FP32; FP8 checkpoint: **FP32**; NVFP4 checkpoint: **FP16+SR**.

**Deployment settings (Sec. 4, p. 12):**
> "Single 8×B200 node (primary target) — NVFP4 weights, FP8 KV cache, FP16 Mamba SSM state with stochastic rounding."
> "Single 8×H100 node — FP8 weights, FP8 KV cache, FP32 Mamba SSM state."
> "Single H100 GPU (1M-context target) — NVFP4 weights, FP8 KV cache, FP32 Mamba SSM state."

**Other relevant facts:** Puzzle prunes the Mamba SSM state size "from 128 to 96 channels" (p. 4), i.e. the state dimension itself is a compression knob. No new verbosity numbers for SSM cache precision are reported here; the paper defers to the Super recipe.

**Correction:** "Nemotron-Labs-3-Puzzle-75B-A9B stores the Mamba SSM state in FP16 only in its NVFP4/Blackwell checkpoint and only with stochastic rounding; its FP8/Hopper checkpoint keeps the state in FP32 because 'FP16 results in reduced accuracy unless stochastic rounding (SR) is employed' and FP16+SR is slower than FP32 on Hopper (Sec. 2.4, Table 2)."

---

## 2603.16731 — Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets

- **First author:** Kristi Topollai (NYU; with Anna Choromanska)
- **Published:** 2026-03-17 (v2 2026-09-27)
- **Venue:** preprint/unknown (no comment/journal_ref; the v2 header is plain arXiv; web search found no venue)
- **Report claim (tag [S]):** Topollai–Choromanska model "the same phenomenon ('EMA state freezing, recovery via reset')" for optimizer-state quantization; the report then adds its own assertion that "SSM state and EMA are mathematically the same first-order IIR recursion."
- **Verdict: CONFIRMED** for what is attributed to the paper. The SSM↔EMA analogy is the report's own; the paper never mentions SSMs, Mamba, or state-space models (grep for "ssm|mamba|state.space|IIR" returns nothing).

**Quotes:**
> "Consider a generic recursion x_t = βx_{t−1} + (1 − β)s_t ... When the increment is too small relative to the local spacing of the floating-point grid, the update rounds back to the same stored value, so the state does not change. We refer to this phenomenon as state-update stalling, also known as signal swamping and unchanged-state dynamics in low-bit EMA updates (Higham, 1993; Wang et al., 2018; Xu et al., 2025)." (Sec. 1)

> "Definition 3.1 (State update stalling). We say that state update stalling occurs at step t if x^q_t = x^q_{t−1}." (Sec. 3)

> "Proposition 3.3 (NR stalling probability). Let z_t ∼ χ²₁. Under the effective-spacing approximation, the one-step probability of state-update stalling under nearest rounding is P^NR_stall(ρ̂) ≈ F_{χ²₁}(1 + ρ̂) − F_{χ²₁}(max(0, 1 − ρ̂))." (Eq. 14)

> "Corollary 3.4 ... once the EMA reaches its steady magnitude, most updates are blocked by the rounding gate."

> "at FP8 and FP4, once the second moment reaches its steady-state scale, almost every step stalls. In this regime the EMA rarely changes, so the optimizer effectively stops updating its estimate of gradient magnitude and instead operates with nearly frozen statistics." (after Table 1)

> "β2_eff ≈ 1 − (1 − β2)(1 − P_stall)" (Eq. 16, effective-decay approximation)

> "resets help substantially when EMA states are stored in BF16 or lower precision, but are often much less useful in full precision." (Sec. 1)

**What is PROVED vs CLAIMED vs MOTIVATED (be careful when transferring to SSMs):**
- The "propositions" are explicitly **approximations**, not theorems: "It is a representative-spacing approximation, not an exact averaging of the stalling"; "This is not an exact reduction of the quantized recursion, since stalling events are state dependent"; "We therefore present the theory as a compact predictive model ... rather than as an exact distributional" result.
- The analysis is for the **Adam second moment** specifically: z_t = g_t²/E[g²] ∼ χ²₁, with constant decay β2 close to 1, stored in BF16/FP8/FP4 and **re-quantized at every step** (stalling = write-back rounds to the same value). Stalling probability "depends only on the precision format and decay β2".
- Stochastic rounding "reduces, but does not eliminate, stalling" (Prop. 5.1, Table 2) — contrast with Nemotron 3 Super, where SR fully fixed the FP16 SSM-cache verbosity issue.
- Mitigation studied = **periodic resets** of the EMA (v ← 0, age j ← 0) with a theory-guided reset period; experiments: controlled simulations + LLaMA-60M-scale LLM pre-training (Fig. 2/3).

**Differences vs. the Mamba SSM case that the report should not paper over:**
1. The paper's regime is high-decay EMA with β≈1 and (1−β)s_t increments far below ULP spacing — "stalling/freezing". Nemotron 3 Super's SSM failure mode is the opposite sign: RTNE **bias accumulating coherently** through time-varying A_t (Eq. 3 of 2604.12374), manifesting as drift/verbosity, and it is fully cured by SR. The two are both write-back-rounding pathologies of a first-order recurrence, but they are not "the same phenomenon" as a mechanism; the report's "동일 현상" should be softened to "an analogous write-back-rounding pathology".
2. Mamba A_t is input-dependent (selective) and the state is a matrix with wide dynamic range; the paper's χ²₁ scalar model does not transfer without new assumptions.
3. The paper's remedy (reset to zero) has no obvious SSM analogue; the SSM fix was SR / INT16 block scaling.

**Suggested rewording:** "Topollai & Choromanska (2603.16731) model an analogous write-back-rounding pathology for quantized EMA optimizer states: when |(1−β)s_t| is below the local ULP spacing, the stored state stalls (Def. 3.1; approximate stalling probability in Prop. 3.3), and periodic state resets restore responsiveness. The analysis is an approximation for Adam's second moment, never mentions SSMs, and its failure mode (freezing) differs from the bias-drift that Nemotron 3 Super observed and fixed with stochastic rounding."

---

## 2509.23202 — Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization

- **First author:** Vage Egiazarian (ISTA; with Castro, Kuznedelev, Panferov, Kurtic, Pandit, Marques, Kurtz, Ashkboos, Hoefler, Alistarh)
- **Published:** 2025-09-27 (v-latest 2026-03-03)
- **Venue:** **ICLR 2026** — every page header of the fetched text reads "Published as a conference paper at ICLR 2026". The report's "[2509.23202, ICLR 2026]" is correct.
- **Report claim (tag [S]):** Egiazarian et al.: "both NVFP4 and MXFP4 are lossy and MXFP4 shows about a 10% relative drop."
- **Verdict: CONFIRMED** (near-verbatim).

**Quote (Contributions, Sec. 1):**
> "Results show that: (1) both NVFP4 and MXFP4 are lossy, with MXFP4 inducing major accuracy drops (∼ 10% relative), and (2) that existing techniques are not well-suited for these new formats, as they do not always outperform RTN. On the positive side, we show that GPTQ and the MR-GPTQ variant yield consistently good recovery for NVFP4. Moreover, MR-GPTQ works particularly well in conjunction with MXFP4, recovering accuracy within 1-2% of NVFP4. For large models, we show that both formats can recover up to 98-99% of the baseline FP16 accuracy."

> "(1) No Lossless Format: Across all formats, the accuracy drop is noticeable. The lowest average drop is for the NVFP4 format with SmoothQuant, GPTQ, or MR-GPTQ" (Sec. 5, Discussion of Table 1)

**Grounding numbers (Table 1, Llama-3.1-8B-Instruct, W4A4, avg over MMLU-CoT/GSM8k/HellaSwag/WinoGrande, FP16 baseline avg 78.93 = 100% recovery):**
- MXFP4: RTN 87.83%, RTN+HT 89.26%, SmoothQuant 89.06%, QuaRot 79.70%, SpinQuant 88.00%, GPTQ 89.47%, **MR-GPTQ 93.31%**.
- NVFP4: RTN 94.67%, RTN+HT 93.82%, SmoothQuant 95.90%, GPTQ 95.92%, MR-GPTQ 96.08%.
- INT4: RTN 92.63%, RTN+HT 94.71%, GPTQ 92.75%.
- So "∼10% relative" for MXFP4 means the 10.5–12% drop with standard methods (RTN/GPTQ) on 8B; MR-GPTQ cuts it to ~6.7%. Appendix (Table 12 context): "a large gap ( > 4 points on average) between the top NVFP4 recovery (96.6%) and the top MXFP4 recovery (92.3%)."

**What is PROVED vs EMPIRICAL:**
- Proved (Lemmas 1–4, under i.i.d. Gaussian / Laplace assumptions): NVFP4's group size 16 "provably neutralizes" Hadamard-style outlier mitigation (Lemma 1, Top-Element MSE); E8M0 power-of-two scale quantization induces high error. Appendix: "E4M3 and E8M0 increase MSErel by 10%, 40% on average, respectively" (scale-quantization error, Laplace samples + Llama weights).
- The ~10% relative accuracy drop is purely **empirical** (PTQ zero-shot evals, Llama-3 1B–70B, Qwen3 8B–32B); the report's statement that "format comparisons are all empirical" is accurate for this number.
- Caveat: this is **post-training quantization (inference)**, not training; the report's paragraph mixes it with training-format comparisons (NVIDIA pretraining, Meta MXFP4 training). If the report uses it to argue about training formats, it should say so explicitly.

**Correction:** none needed; optionally cite the table: "MXFP4 RTN/GPTQ recover only 87.8–89.5% of FP16 accuracy on Llama-3.1-8B W4A4 vs 94.7–96.1% for NVFP4 (Table 1); MR-GPTQ narrows the MXFP4 gap to 93.3%."
