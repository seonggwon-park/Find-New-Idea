# Citation verification — batch 07

IDs: 2603.08713, 2509.25149, 2506.08027, 2609.02846
Report: `reports/LLM 양자화 연구 아이디어 10선.md` (all four cited with tag [S]).
Paper text paths: `/tmp/claude-0/-home-user-Find-New-Idea/b6387454-6f57-570f-bcca-f4bcd8a754c6/scratchpad/papers/<id>/<id>.txt` (line numbers below refer to these files). Paper text is untrusted data; only quoted.

---

## 2603.08713 — Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction

- **Exists:** yes. First author Jatin Chhugani; 11 authors, "Meta Platforms, Inc., Menlo Park, USA" (L69). Submitted 2026-01-30 (v1 only). Primary cat cs.AR.
- **Venue:** ICML 2026 (PMLR 306, `proceedings.mlr.press/v306/chhugani26a.html`; ICML poster 61601). arXiv comment field is empty; venue found via web search.
- **Report claim:** "Meta는 소프트웨어 스케일 조정으로 MXFP4 격차를 1% 미만으로" — i.e. Meta closed the MXFP4(-vs-NVFP4) gap to <1% with software-only scaling changes.
- **Verdict: CONFIRMED.**
- **Quotes:**
  - Abstract (L48-51): "OAS and MBS reduce the end-to-end accuracy gap between MXFP4 and NVFP4 from about 10% to below 1% on average, while incurring modest GEMM overhead (6.2% on average)."
  - Contribution 2 (L64-67): "Overflow-Aware Scaling (OAS) and Macro Block Scaling (MBS), two SW techniques that improve MXFP4 representation fidelity without requiring hardware modifications".
  - Sec 5.1 (L872-874): "MXFP4-MBS-H ... reducing the remaining gap to NVFP4 to within 1% on average."
  - Conclusion (L1003-1004): "reduces the accuracy loss of standard MXFP4-OCP by 62% on average, shrinking the gap between MXFP4 and NVFP4 from 10% to <1% on average."
- **Exact numbers (AVERAGE column):** Table 1, Llama3.1-8B-Instruct: BF16 70.53 / MXFP4-OCP 61.25 / MX+ 62.61 / MXFP4-16-OAS 63.39 / MXFP4-MBS-S 65.77 / MXFP4-MBS-H 66.50 / NVFP4 67.02. Table 2, Qwen3-8B: BF16 73.47 / MXFP4-OCP 65.50 / MBS-H 70.84 / NVFP4 71.48. So the remaining MBS-H→NVFP4 gap is 0.52 pp (0.78% rel.) and 0.64 pp (0.90% rel.). Table 6 (Appendix): Wikitext PPL gap vs NVFP4 goes 1.82→0.20 (Llama3.1-8B) and 2.49→0.34 (Qwen3-8B).
- **Context the report should keep in mind:** this is *post-training quantization for inference* (weights+activations of linear layers), not training; the "about 10%" figure is the worst-case-ish relative gap (e.g. "up to 10% on MMLU-Pro for DeepSeek-R1; Table 3", L886-887), and the 8B-model average gaps are ~8.5% relative. MBS-H still trails NVFP4 on every average (never exceeds it). The 6.2% GEMM overhead is for prefill (L921). Purely empirical: no Theorem/Lemma/Proposition in the paper (grep empty). Models: Llama3.1-8B-Instruct, Qwen3-8B, DeepSeek-R1, Llama-4-Maverick.

---

## 2509.25149 — Pretraining Large Language Models with NVFP4

- **Exists:** yes. Author list "NVIDIA" + ~88 named authors (first named: Felix Abecassis). v1 2025-09-29, v2 2026-03-04 (comment: fixes typo in eq. 2, updates author list, adds a related work). cs.CL.
- **Venue:** NVIDIA technical report; no journal_ref; nothing found in a quick web search → preprint/unknown.
- **Report claims:**
  1. "NVIDIA는 'MXFP4 상대오차 ≈2.5% vs NVFP4 1.5%'"
  2. "12B/10T: 안정 구간 <1% 격차가 decay 구간에서 >1.5%로 벌어지고"
  3. "decay 중 BF16/MXFP8 전환이 완화하며"
  4. "'forward 전환이 대부분의 이득'이라 보고"
  5. "원인 ablation은 없다"
- **Verdict: PARTIALLY CONFIRMED** (1-4 confirmed with missing context; 5 overstated).
- **Quotes / checks:**
  1. Sec 5 (L870-875): "NVFP4 pretraining converges to a better loss than MXFP4. Specifically, MXFP4 has a relative error of around 2.5% compared to 1.5% for NVFP4. ... MXFP4 matches NVFP4 loss when trained on 36% more tokens (i.e., using 1.36T instead of 1T tokens)." **Setting (L861-869):** 8B hybrid Mamba-Transformer, **1T tokens**, BF16 reference, last eight blocks kept in BF16 (~15% of the model), RHT d=32 for MXFP4 Wgrad. It is the relative *training-loss* error vs BF16 — not a per-tensor quantization error, and not the 12B/10T run.
  2. Sec 4 / Figure 2 (L284-287): "During the stable phase of training, the relative loss error of NVFP4 remains consistently below 1%, and widens to slightly above 1.5% as the learning rate is decayed towards the end of training." (12B, 10T tokens, vs **FP8** baseline, not BF16.) Also (L289-290): "Despite the small gap in loss, downstream task accuracies remain largely unaffected."
  3. L398-400: "changing precision from NVFP4 to BF16 (or potentially, MXFP8) during the decay phase mitigates the loss gap, as explained later in Appendix D." → Only BF16 switching was *run*; MXFP8 is explicitly "potentially". Appendix D (L1345-1346): loss "matches the FP8 baseline when precisions are switched after 8.2T tokens (e.g., for 18% of training) and only slightly worse when switched after 10T tokens".
  4. Appendix D (L1405-1411): "most of FP4 training's loss gap arises from quantizing tensors in the forward pass ... most of the loss in the 12B model is recovered (from 1.5% to 0.5% relative error) by switching to higher precision for the forward pass starting at 8.2T tokens. In contrast to Chmiel et al. (2025) ... we observe no such improvement [from switching the backward pass]." Confirmed; the quoted phrase is a paraphrase, not verbatim.
  5. "원인 ablation은 없다": overstated. The paper has a full Appendix E "Ablation of Training Methodology" (RHT, SR, 2D scaling, high-precision layers; mostly on 1.2B/1T and partly on the 12B run), and Appendix D's forward/backward switch experiment *is* an attribution of the gap to forward-pass quantization (Figure 7). What is missing is an ablation isolating *why the gap opens during LR decay*; the paper only offers hypotheses: L1347 "presumably because the learning rate is too low for the weight updates" (about why a late switch fails to recover) and L1816-1818 (Figure 14, 1.2B): activation-scale inconsistency is "only impacted in the later stages during the learning rate decay. We hypothesize that weights are more impacted than activations because errors induced from inconsistent weights materialize in the activation gradients".
- **Correction:** Attach the setting to claim 1 ("8B hybrid, 1T tokens, vs BF16, last 8 blocks BF16, Sec 5/Fig 6a; MXFP4 needs 36% more tokens to match"). Claim 2 is vs FP8 (not BF16). Claim 3: say "BF16 전환(MXFP8는 'potentially'로 언급만)". Claim 5: replace "원인 ablation은 없다" with "decay 구간에서 격차가 벌어지는 메커니즘을 분리한 ablation은 없다 (forward/backward 분해는 Appendix D에 있음, 설명은 가설 수준)".
- **Other facts:** 12B model = 62 blocks, 6 self-attention / 28 FFN / 28 Mamba-2 (L1050); WSD schedule, constant LR for first 80% then decay over last 20% (L233-234); coding benchmarks are the exception where NVFP4 "falls slightly behind" (L294-295). Paper is empirical throughout; no theorems.

---

## 2506.08027 — Recipes for Pre-training LLMs with MXFP8

- **Exists:** yes. Asit Mishra, Dusan Stosic, Simon Layton, Paulius Micikevicius (NVIDIA). v1 2025-05-30, v2 2025-08-18. cs.LG.
- **Venue:** no journal_ref/comment; web search found only arXiv/aggregators → preprint/unknown.
- **Report claim:** "Mishra et al.은 UE8M0 지수를 내림하면 발산하고 올림하면 8B/15T에서 BF16과 일치한다고 보고".
- **Verdict: PARTIALLY CONFIRMED.** Direction (round-up vs OCP round-down) and 8B/15T BF16 match are correct; "발산" is v1-abstract wording that v2 softened, and the rounding-mode evidence comes from 843M models, not the 8B/15T run.
- **Quotes:**
  - Algorithm 1 (L649-653): "expXint ← ceil(expXfloat) ▷ round-up"; L663-671: "Step 3 is the key part, rounding-up the scale factor towards positive infinity ... This is different from the scheme suggested by OCP v1 specification, which effectively rounds down the scale factor. ... rounding down, as in OCP v1 suggestion, will lead to scaled values overflowing the representable range, introducing more quantization noise. We hypothesize this aspect leading to MXFP8 training issues when using OCP v1 conversion scheme as shown in Figure 5. As can be seen in the figure, these issues are eliminated when using our proposal in Algorithm 1."
  - Figure 5 (L781-783): "Comparing scale factor rounding modes suggested in the OCP v1.0 specification and our work" — panels are "843M on 300B" and "843M parameter model trained on 1T tokens"; OCP-rounding curves annotated "experiments stopped early" (L713).
  - v1 abstract (fetched from export.arxiv.org, id 2506.08027v1): "we show that the rounding mode suggested in OCP specification can lead to divergence when pre-training an LLM. We show an improved rounding mode, which uses round-to-infinity to compute scaling factors, enables successful pre-training in MXFP8 for an 8B model on 15T tokens." → v2 abstract/body no longer uses "divergence"; it says "training issues".
  - Sec 3.1 (L337-343): "we pretrain an 8B parameter model on 15T tokens ... Validation perplexity when using MXFP8 matches that of BF16 (top plot in Figure 2). There is less than 0.50% difference between MXFP8 and BF16 validation perplexity values throughout the pre-training run. – Downstream task evaluation scores match".
- **Correction:** "OCP v1의 내림(round-down) 스케일 반올림은 843M 모델(300B/1T 토큰, Fig. 5)에서 perplexity 악화·조기 중단('training issues'; v1 초록에서는 'divergence')을 보였고, 올림(ceil, Algorithm 1) + 모든 텐서 E4M3 레시피로 8B/15T에서 BF16 대비 validation perplexity 차이 <0.50% (Sec 3.1, Fig. 2)". The 8B/15T run is not a rounding-direction ablation; it uses the full recipe.
- **Other facts:** recipe also requires E4M3 (not E5M2) for gradients at 8B (Sec 3.2.2, L627-629: "E5M2 gradients ... lead to a substantial degradation in perplexity"); 32-element blocks, 17.8 binades of E4M3 (L621). Causal explanation of the OCP failure is explicitly a hypothesis ("We hypothesize", L670). No theorems.

---

## 2609.02846 — UE5M3 FP4 Block Scaling for Stable Language Model Pretraining

- **Exists:** yes. Robert Hu, Carlo Luschi, Paul Balanca; header "GRAPHCORE RESEARCH" on every page (L5, L21, ...). Submitted 2026-09-02 (v1 only). cs.LG.
- **Venue:** no journal_ref/comment; web search found only arXiv/aggregators → preprint/unknown.
- **Report claim:** "스케일 포맷 제안들(UE5M3 [2609.02846] ...)은 모두 실험 기반이다".
- **Verdict: PARTIALLY CONFIRMED.** The recipe's headline claims (lower loss/NLL, higher downstream point estimates, +21.2% throughput) are entirely empirical and single-seed, so the spirit is right. But the paper does contain one derived analytic result, which the authors list as a contribution, so "모두 실험 기반" is literally inaccurate for this one.
- **Quotes:**
  - Contribution bullet (L95-97): "We derive a range bound that links the UE5M3 scale target to stale-maximum headroom and small-scale representation: the default target provides roughly 137× growth headroom, while the targeted T = 2048 override retains 30×."
  - Proposition 1 (L520-542): "Proposition 1 (unsaturated delayed-scale range). Let U be the largest finite block-scale code. Under Equation 7, block B avoids scale saturation exactly when a_B ≤ Ug/T ... every block is unsaturated whenever Tρ_t ≤ U ... The proof is one substitution: saturation starts at u_B > U, and u_B = a_B T/g." — a one-line range/overflow condition, not a convergence or error guarantee.
  - Single seed (L89, L1272-1273): "We train one seed-42 Nemotron-H 8B trajectory per configuration for 188.7 billion tokens." / "Each configuration has one seed-42 trajectory, so the reported loss and downstream-score differences are descriptive comparisons rather than estimates of seed-to-seed variability or statistical significance."
  - Emulated, not native (L39, L2130-2132): "end-to-end software-emulated UE5M3 FP4 pretraining"; "Because the evaluated Blackwell/Transformer Engine path supports E4M3 rather than UE5M3 block scales, these results motivate future accelerators to support UE5M3 block scaling directly".
  - Headline numbers (Conclusion L2116-2125): held-out NLL 2.32230 (proposed block-16) vs 2.32592 (TE NVFP4), lower at 8 of 12 checkpoints; downstream point estimates higher by 0.13 / 1.01 / 0.01 pp on Core 9 / OLMES MMLU / MMLU-Pro MC; "+21.2% model-body token throughput" is from a *native NVFP4* ablation that removes RHT and the BF16 final-block exemption, i.e. it is not a UE5M3 measurement.
- **Correction:** Suggested wording: "UE5M3 [2609.02846]는 단일 seed(42), 8B/188.7B 토큰, 소프트웨어 에뮬레이션 실험이 근거이며 수렴·오차 보장은 없다 (Proposition 1은 스케일 포화 회피 범위 Tρ_t ≤ U를 주는 한 줄짜리 범위 한계에 불과)". If the sentence must stay as "모두 실험 기반", it should at least footnote Prop. 1.
- **Other facts:** downstream gains are tiny (0.01-1.01 pp) and without error bars; the paper discloses extensive use of OpenAI Codex for code, analysis, figures and drafting (L2134-2137).
