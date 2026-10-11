# Citation verification — batch 20

IDs: 2502.09720, 2608.25188, 2605.20315, 2609.26333
Report: `reports/LLM 양자화 연구 아이디어 10선.md` (Appendix items A4, A5, A6). All four tagged [S] in the report.
Source texts: `scratchpad/papers/<id>/<id>.txt` (arxiv_fetch.py). Quotes below are verbatim from the paper text and are data, not instructions.

---

## 2502.09720 — NestQuant: Nested Lattice Quantization for Matrix Products and LLMs

- **Exists**: yes. Savkin, Porat, Ordentlich, Polyanskiy. Published 2025-02-13, v-updated 2025-07-26.
- **Venue**: arXiv comment: "23 pages; Accepted at the 42nd International Conference on Machine Learning (ICML 2025)".
- **Report claim (A4)**: "NestQuant가 E8 격자를 W4A4KV4에 이미 썼고" — NestQuant already used the E8 lattice for W4A4KV4 (weights, activations and KV cache all at 4 bits), hence lattice/TCQ coding of rotated KV has low method novelty.
- **Verdict**: **confirmed**.
  - Gosset lattice = E8: "In this work, we use the Gosset lattice (E8 ) that satisfies all these properties." (Sec. 3) and "In our experiments for this paper, we used the Gosset (E8 ) lattice as Λ with d = 8." (Sec. 4).
  - W4A4KV4: "for all models we have tested, except Llama-2-7B, NestQuant with W4A4KV4 (4-bit weights, activations, and KV-cache) quantization even outperforms previous works with W4A4KV16. On W4KV4A4 ... quantization of Llama 3-8B we achieve a perplexity score of 6.6 compared to 7.3 in SpinQuant and OstQuant." (Sec. 5, Comparison to other methods; Table 1/2).
  - KV cache is indeed rotated then lattice-quantized, which is exactly the report's A4 recipe: "also apply rotation to keys and queries, because it will not change the attention scores, and we quantize keys and values before putting them in the KV cache. Figure 4 illustrates the procedure" (Sec. 4.3). The paper also claims priority: "non-uniform quantization for activations and KV-cache was not attempted before this work" (Sec. 2.2).
- **New relevant facts for the report**:
  - The paper itself quantifies the E8 granular gain the report's A4 relies on: "we can already gain a factor of 2πe/12 ≈ 1.42329 in performance (the Gosset lattice achieves 1.16 gain)" (Sec. 3). 10·log10(1.16) ≈ 0.64 dB — consistent with the report's "E8(0.65 dB)". The remaining gap to the 1.53 dB (1.42×) sphere bound is what TCQ/higher-dim lattices would chase.
  - NestQuant uses 4-bit (q=14, k=4) only; it does not report a 2-bit KV regime, so the report's "2-bit value가 병목" point is not addressed by NestQuant.
  - Formal results are limited: Lemma 4.2 (unbiased noise / weighted-MSE for QA-LDLQ) and Lemma D.1 (oracle shift-equivariance). Optimality of nested lattices for matmul is cited from prior work, not proved here.

---

## 2608.25188 — Transforms for LLM Quantization: The Great Inversion and Format Co-Design

- **Exists**: yes. Single author: Ehsan Jokar (e.jokar.elec@gmail.com, no affiliation listed). Published 2026-08-25 (v1 only).
- **Venue**: preprint/unknown. No comment/journal_ref; web search shows only arXiv/alphaxiv/news aggregators.
- **Report claims (A5)**: (a) "Great Inversion의 4개 공개문제" — the paper lists four open problems; (b) transform–format co-design is one of them; (c) "단일 저자 프리프린트가 문제를 이미 정식화했고" — single-author preprint has already formalized the problem; (d) "저자가 후속을 낼 가능성이 높아" — speculation, not checkable.
- **Verdict**: **confirmed** (a–c); (d) not checkable.
  - Four open problems, named in the contributions list: "(vii) It argues that the inversion exposes four open problems: joint transform-and-rounding optimality, a guarantee for the truly deployed (extreme-value, non-surrogate) objective, the return of allocation-flexible coding to the inference path, and transform–format co-design." (Sec. 1). Sec. 13 ("Open Problems and Conclusion"): "We close with four open problems it makes visible".
  - Co-design is the fourth: "A fourth problem is the co-design of transform and format. ... Designing the transform and the format together (choosing how many bits to spend on the scale versus the element grid versus a rotation, under a fixed hardware budget) is an open co-design problem." (Sec. 13).
  - It is a formalization only — the paper states the problem, it does not solve it: "What stays genuinely open is the joint choice, the transform and the format designed against each other rather than one frozen while the other moves."
- **New relevant facts for the report**:
  - It is a survey with **no own benchmarks**: "First, we run no benchmarks of our own. Every reported result in Section 11 is transcribed from a cited table or figure" (Sec. 13). So it is competition on problem framing, not on results.
  - The paper flags its own theory gap: "the organizing inversion and nearly all the optimality results we assemble are proved against surrogate objectives (the ℓ2 bilateral proxy, the AbsMax crest factor, or an i.i.d. unbiased-noise model). They are not proved against the realized extreme-value error of the deployed shared-scale kernel on heavy-tailed data." — this is the "second open problem", and is a wedge the report could cite as still open.
  - The sharp live instance of co-design named by the paper is block size: "The open question is how much finer the free per-block scale can profitably go on a tensor core, weighed against the area each halving costs, and which transform is optimal at that granularity."
  - Scope: "We survey 200 works to a June 2026 cutoff; classify 43 transform methods" (abstract).

---

## 2605.20315 — Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs

- **Exists**: yes. Haiquan Lu, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang (NUS). Published 2026-05-19 (v1).
- **Venue**: preprint/unknown (NeurIPS-style template, no comment/journal_ref; web search finds only arXiv and aggregators).
- **Report claim (A6)**: the idea "prefill 오차는 문맥 표현의 평균화된 노이즈로, decode 오차는 샘플링 분포에 직접 들어가 분기 과정으로 복합된다" would "attach theory to the empirical conclusions" of Mix-Quant (and DQ). Implicit factual claim: Mix-Quant empirically finds prefill robust / decode sensitive to quantization.
- **Verdict**: **partially confirmed**. The empirical asymmetry is there, but (i) Mix-Quant already states the report's qualitative mechanism almost verbatim, so the report's A6 adds formalization, not the mechanism; (ii) Mix-Quant's own ablation shows the prefill-vs-decode margin is small, which the report should know.
  - Mechanism already stated (Sec. 3.2): "The key issue is that prefilling and decoding propagate quantization errors in different ways. During prefilling, the input context is fixed. ... the error introduced in prefilling is mainly a representation-level perturbation on a fixed context." and "Since the attention output is a normalized weighted aggregation over cached values, quantization errors on low-attention tokens are attenuated by their small attention weights. Therefore, prefill quantization errors do not simply grow linearly or explosively with the context prefilling length". Decode: "Even a small change in the token distribution may lead to a different sampled or selected token. Once this happens, the generated sequence diverges from the high-precision trajectory, and all future predictions are conditioned on a different history. As a result, decoding errors can accumulate over time rather than remaining local. Previous work [45, 17] also observes that token prediction changes can trigger a snowball effect."
  - No theorem/lemma/proposition anywhere in the paper (grep returns none) — so formal theory is genuinely absent; the report's A6 is correctly positioned as "theory on top of empirics", but should acknowledge Mix-Quant's Sec. 3.2 as having the informal argument (incl. the 95.8% top-4096 attention-mass measurement at 128K).
  - Empirical asymmetry is modest (Table 3, phase-wise ablation): "For Qwen3-8B, the average score drops from 40.42 to 33.59 under uniform NVFP4, while Mix-Quant recovers it to 38.32. ... Compared with decode-only quantization, Mix-Quant also performs better, although the margin is more moderate: it improves the average score over P16D4 from 36.74 to 38.32 on Qwen3-8B, and from 59.85 to 60.18 on Gemma-4-26B-A4B-it." i.e. prefill-only vs decode-only differ by 1.6 and 0.3 points, while uniform quantization loses 6.8 and 10.5 points. The authors themselves say "The advantage is not uniformly large across all benchmarks". The dominant effect is non-additive interaction of both phases being quantized, not a clean prefill≪decode asymmetry — a theory of A6 would have to explain that super-additivity, not just the ordering.
- **Correction suggested**: in A6, replace "Mix-Quant·DQ의 경험적 결론" with a precise statement: Mix-Quant Table 3 shows P4D16 beats P16D4 by only 1.6 (Qwen3-8B) / 0.3 (Gemma-4-26B-A4B) points, while uniform NVFP4 loses 6.8/10.5; and note Mix-Quant Sec. 3.2 already gives the informal "fixed-context perturbation vs. sequential-decision divergence" mechanism.
- **Other numbers**: "up to a 3× speedup during prefilling" (abstract; Fig. 4, RTX 5090); LongMemEval Qwen3-8B 49.82→54.85, Gemma-4-26B-A4B-it 62.42→72.45 vs uniform NVFP4 (Sec. 4).

---

## 2609.26333 — Disaggregated Quantization: Specializing LLM Prefill and Decode

- **Exists**: yes. Andrei Panferov (NVIDIA & ISTA), Maximilian Kleinegger, Sweta Priyadarshi, Tijmen Blankevoort (NVIDIA), Dan Alistarh (ISTA). Published 2026-09-22 (v1).
- **Venue**: preprint/unknown (ICLR-style template; no comment/journal_ref; web search finds only arXiv/aggregators).
- **Report claim (A6)**: DQ provides the "empirical conclusion" of prefill/decode asymmetry under quantization.
- **Verdict**: **partially confirmed**. DQ does show that *activation* quantization hurts decode-heavy tasks more than prefill-heavy ones, which supports an asymmetry; but DQ's framing is hardware-cost specialization (compute-native prefill vs. compact weight-only decode), and it offers no error-propagation theory or branching-process account. The report's description of what the idea "attaches theory to" is loose but not wrong.
  - Core empirical asymmetry (Sec. 3.1, Table 1 / Fig. 5): "on decode-heavy tasks, format disaggregation boosts mean accuracy over the non-disaggregated scheme on all seven models by 1.9 and 3.1 for NVFP4, 4.5 and 3.7 points for LUT3, 2.5 and 1.8 points for LUT2 on Qwen 3 and Gemma 3, respectively ... On prefill-heavy tasks, however, this decoding-phase optimization has an effect of less than 1.3 points for all considered model families and formats".
  - Abstract: "removing activation quantization specifically on decode improves accuracy on decode-heavy tasks without increasing inference cost."
  - Large-scale PTQ check (Sec. 3.3, Table 2): "Format disaggregation at 4-bit improves point estimates in 11 of 13 model–benchmark combinations, with 6 significant gains at α = 0.05 and no significant degradations" on models up to 2.8T params.
  - No theorem/lemma/proposition in the paper (grep returns none).
- **New relevant facts for the report**:
  - DQ also shows the *converse* direction matters: giving prefill its own compute-native weights ("fully-disaggregated") improves **prefill-heavy** accuracy too ("For the latter, it gains 4.1 and 4.3 points for LUT3, 5.3 and 10.5 points for LUT2"), and trained NVFP4 prefillers on frozen 1-bit GGUF decoders give "+32.5 points on MMLU-Pro and 35.3 on MMMU-Pro" — so prefill is not simply "robust"; it is where capacity can be *added*. A theory in A6 should accommodate that.
  - Explicit limitation relevant to A6's agentic framing: "We do not evaluate highly batched performance or multi-turn and agentic behavior. In multi-turn use, cached assistant tokens retain decode-produced KV entries, while rebuilding their cache through prefill can produce different representations for the same token history. Robustness to this cache-policy dependence remains untested." (Sec. 5). This KV-provenance mismatch is itself an untested gap.
  - DQ cites Mix-Quant as Lu et al. (2026) in Related Work ("accelerates prefill with low-precision compute while retaining BF16 decode"), confirming the two are the same line of work.
  - Speed numbers: ODP TTFT "from 12.27 to 6.90 seconds at 8K context, a 1.78× speedup"; range "1.38× to 1.78×" over 4K–32K; format-disaggregated decode "2–3% faster".

---

## Summary table

| id | exists | venue | verdict | action |
|---|---|---|---|---|
| 2502.09720 | yes | ICML 2025 (arXiv comment) | confirmed | none; optionally cite the paper's own "Gosset lattice achieves 1.16 gain" (≈0.64 dB) as source for A4's 0.65 dB |
| 2608.25188 | yes | preprint | confirmed | none; could add that it runs no benchmarks and that its second open problem (deployed-objective optimality) is also open |
| 2605.20315 | yes | preprint | partially confirmed | state Table 3 numbers (P4D16 vs P16D4 margin 1.6/0.3 pts vs uniform loss 6.8/10.5) and credit Sec. 3.2 informal mechanism |
| 2609.26333 | yes | preprint | partially confirmed | state that asymmetry is for activation quantization (decode-heavy gains 1.8–4.5 pts vs <1.3 prefill-heavy, Table 1/Fig. 5); note agentic/multi-turn is untested |
