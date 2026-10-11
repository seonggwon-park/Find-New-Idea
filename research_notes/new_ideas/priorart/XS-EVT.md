# XS-EVT: prior-art check (A3 + S05 merged)
Extreme-value law for long-context retrieval under KV quantization: ln N*_q = ln N*_FP/(1+r), with r = sigma_q^2/sigma_d^2, so bits b(N) ~ 1/2 log2 ln N.

Checked on 2026-10-11 with roughly 30 tool calls.

**Verdict: survives_narrowed. Novelty 2/5.** The math is already published. 2609.22101 proves it with no quantization term, and it is textbook information theory. Two of the gap-claim premises are misattributed. What remains is an empirical test: whether the critical length measured on real retrieval heads follows the variance-substitution law across bit widths.

## Closest work (full text read)
| id | date | what it contains | overlap |
|---|---|---|---|
| 2609.22101 *Context Poisoning as Extreme-Value Attention Interference* | 2026-08-12 | Thm 1 (txt L251-311): one item with S* <= gamma, N iid N(0, sigma^2) distractors, A_N <= a0+(1-a0)[Phi((gamma+tau log((1-rho)/rho))/sigma)]^N; it requires gamma >= sigma sqrt(2 log N)+O(1) (L306-308). Cor 1 covers a random needle logit. Prop 1 shows that gating to K replaces N by K. L1296-1324: it tests the iid-Gaussian assumption on Gemma-3-4B and calls Thm 1 "a diagnostic baseline rather than a calibrated quantitative bound". | **The whole math core.** Adding independent N(0, sigma_q^2) noise to every logit only replaces sigma^2 with sigma_d^2+sigma_q^2 in Thm 1. The candidate's "Theorem" is a one-line corollary. Its own text calls it "corollary-level". |
| 2608.14191 *KV Cache Compression Through the Lens of Transform Coding* (AATC) | 2026-08-14 | Proves that under a white-noise quantization model the attention distortion splits additively over key and value contributions. Fig. 1 (L1179-1215) plots RULER against context length at 2 and 2.5 bits: "AATC var-only degrades sharply as the context grows", while full AATC does not. | Empirical: KV-quantization damage grows with length. The paper gives no law. |
| 2609.04098 *Why Gated DeltaNet Survives 4-Bit Quantization* | 2026-09-03 | L495: "The FP8-KV cost behaves in exactly the opposite way: it is small, rises with position". L552: the weight cost is "largest at the start, falling to zero and below by the end of the window". The KV cost is removed with calibrated scales (§7). | Already reports S05 consequence (2), weight gap shrinking and KV gap growing with position, as an observation. The paper attributes the KV part to an attention-path scale effect. |
| 2607.28699 *WitCert* | 2026-07-30 | Raw fp8 scores 22.8 and gating restores 79.7 (L38, L1877-1886). Setting: "the six hard RULER-4096 tasks" (L1869-1873), Qwen2.5-7B. Fig. 3 (L1864) shows a "broken fp8 KV cache", fixed by scaling, outlier bypass and K=int8. | Overlap: none. This is a premise problem; see below. |
| 2606.24033 *Block-GTQ* | 2026-06-23 | Lemma 2 (L3076, L3128) bounds each query-key pair, so it has no length term. | Premise is correct. |
| 2605.20868 *Runtime-Certified Bounded-Error Quantized Attention* | 2026-05-20 | L900: the TV bound is "independent of sequence length N". With naive INT8K/INT4V, NIAH falls to 5-10% against 39-66% dense, and RULER loses 44-56pp "across all context lengths" (L1844-1850), so the drop is not length-graded. | Evidence against the claim that length is the driver. |
| 2605.26266 *Quantized Keys Steal Attention* (Jensen bias) | 2026-05-25 | Corrects the sum-of-mass bias from noise. | The candidate already excludes this. |
| 2608.28911 *SemKV* | 2026-08-28 | Quality cliff in (2.0, 2.322] bits. The cliff "depends on the model-quantizer pair under a specified evaluation protocol" (L25). Storage is measured from 4K to 128K, but the cliff is not studied as a function of length. | Adjacent: an empirical bit cliff with no length law. |
| 1307.6609 / 1312.2063, Ingber, Courtade, Weissman, *Compression for (Quadratic) Similarity Queries* | 2013 | Fundamental trade-off between compression rate and how reliably queries can be answered on a compressed database. L102-107 cite the biometric-identification-capacity line, where the number of entries that can be identified reliably is limited by the compression rate. | Classical information-theory analogue of "bits trade against log N". The scalar EVT version is the textbook M-ary orthogonal-signalling and union-bound result: reliable iff ln N < SNR/2. |
| vLLM blog "State of FP8 KV-Cache" (2026-04-22) | 2026-04 | "On a 128k needle-in-a-haystack task, FP8 accuracy dropped from 91% (BF16 baseline) to just 13% — a regression traced to imprecise FP32 accumulation in the Tensor Cores". It was fixed by two-level accumulation. | The best-known long-context FP8 collapse came from a kernel accumulation bug, not from noise in the stored KV values. |

## Premise check
1. "WitCert raw fp8 22.8 on hard RULER; nothing explains this collapse with a length-dependent theory". **The numbers are right, but they do not support the framing.** The test uses 4096-token RULER, so it is not long context. The paper calls the cache "broken" and repairs it by scaling, outlier bypass and K=int8, which points to a range or scale failure. The candidate's own law gives r = sigma_q^2/sigma_d^2 of about 1e-3 for e4m3, a relative step near 2^-4. That predicts a change in ln N* of about 0.1%, which is not a collapse. So the law cannot explain the event that motivates it.
2. "Block-GTQ Lemma 2 has no length term". **True**, because the bound is per query-key pair.
3. "2609.22101 Thm 1 gives Omega(sqrt(log N)) and has no quantization". **True.** But its sigma already lumps all logit noise together, so quantization enters only by redefining sigma.
4. "2609.04098 NVFP4 gap shrinks with position while KVarN accumulates; the sign difference is unexplained". **Partly false.** 2609.04098 itself separates the two costs (weight cost falls with position, FP8-KV cost rises) and explains the KV part as a calibration and scale effect. KVarN's accumulation comes from wrong token scales in pseudo-decode (abstract), not from extreme-value dilution.
5. "Queries for a length-versus-bits quantization law returned 0". **Confirmed.** None of my 20+ formulations found a paper that states ln N*(b) or b(N) for KV quantization.

## What remains
- The law ln N*_q = ln N*_FP/(1+r) and the bit requirement b(N) have not been stated or fitted for KV quantization. The fitting protocol is also new: estimate m, sigma_d and sigma_q per retrieval head from short contexts, then predict where needle accuracy fails at long context and at each bit width.
- A per-head bit allocation that targets a given context length is new. Block-GTQ allocates by RoPE band, RateQuant and RDKV by rate-distortion, and DuoAttention and RazorAttention keep retrieval heads in full precision; none of them depends on length.
- Weaknesses a reviewer would raise. The theory has no new content. Distractor logits in real heads are not iid Gaussian, and 2609.22101 L1296-1324 already shows the violation. Real KV-quantization failures (WitCert, 2605.20868, the vLLM accumulation bug, SemKV's 2-bit cliff) look like scale, outlier or kernel failures that hit all lengths, not EVT dilution. The temperature and SSMax "predictions" hold by definition for argmax readout.

## Sharpened hypothesis
For each retrieval head h, estimate (m_h, sigma_{d,h}, sigma_{q,h}(b)) from contexts of 4K tokens or less. Then the Gaussian-EVT corollary ln N*_h(b) = m_h^2 / (2(sigma_{d,h}^2 + sigma_{q,h}(b)^2)) should predict the context length where needle accuracy fails at 2-4 bits, out to 128K, within a stated tolerance. It should do better than two baselines: length-agnostic logit-MAE (Block-GTQ-style) and the measured-cliff heuristic (SemKV). Allocating bits to the heads that limit length should then extend usable context at a fixed budget. Falsifier: if failure onset under well-scaled 2-3 bit KV does not shift with length as predicted, for example if the accuracy drop is flat across 8K-128K as 2605.20868 L1844-1850 suggests for naive INT4V, the law is wrong for real models. Present it as an empirical law tested on 2609.22101's model, not as a theorem.

## Significance
For a main track, the theory alone would be rejected as a trivial corollary of 2609.22101 and of textbook extreme-value and signalling results. The paper would need strong quantitative prediction across 3 or more model families, and a length-targeted allocator that beats Block-GTQ, KVTuner, RateQuant or SemKV at equal bits, with the gain growing with context length. It would also need to control for scale and outlier failures, which seem to dominate observed collapses. Without these it is a workshop paper.

## Queries run (arxiv_search.py unless noted)
1. abs:quantization AND abs:"needle" AND abs:("context length" OR "sequence length") AND abs:(theory OR theoretical OR analysis OR bound) [date, >=2025] -> 0
2. abs:("KV cache" OR "key-value cache") AND abs:(quantization OR quantized OR "low-bit") AND abs:(signal-to-noise OR SNR OR "noise") AND abs:(length OR "long context") [date, >=2025]
3. abs:attention AND abs:("extreme value" OR Gumbel OR "order statistics") AND abs:(quantization OR precision OR "low-bit" OR noise) [>=2024]
4. abs:("retrieval heads" OR "retrieval head") AND abs:(quantization OR quantized OR bits OR precision) [>=2025]
5. KV quantization AND ("bit allocation" OR mixed-precision) AND (context length / length-aware) [>=2025]
6. (quantization OR low-bit) AND long context AND (cliff OR collapse OR "phase transition" OR threshold OR "critical length") [>=2026]
7. (attention OR softmax) AND (SNR OR margin) AND ("log n" OR logarithmic OR "extreme value") AND (length OR context) [>=2025]
8. (quantization OR FP8 OR INT4) AND (RULER OR needle-in-a-haystack) AND (theory OR law OR predict OR explain) [>=2026]
9. (softmax OR attention) AND (argmax OR top-k OR "hard retrieval") AND (noise OR "logit noise") AND (context length OR "number of keys") AND (Gaussian OR Gumbel) [>=2024] -> 0
10. ("associative memory" OR Hopfield) AND (quantization OR precision OR bits) AND capacity AND (attention OR transformer)
11. ("finite precision" OR "bounded precision" OR "low precision") AND (attention OR transformer) AND (context length) AND (retrieval OR copying OR recall)
12. quantization AND (KV cache OR attention) AND (context length) AND ("grows with" OR "scales with" OR "degrades with") [>=2025]
13. ("attention dilution" OR "attention fading" OR "score dilution") AND (quantization OR noise)
14. ti:(quantization OR low-bit OR "KV cache") AND ti:(length OR long context OR position) [>=2026]
15. ("orthogonal signaling" OR "M-ary" OR "union bound") AND (attention OR transformer OR language model)
16. quantization AND ("scaling law") AND (context length OR long context) [>=2025] -> 0
17. (key quantization OR key cache) AND (ranking OR top-k OR argmax) AND (error OR noise) [>=2025]
18. ("similarity queries" OR "similarity identification" OR "identification rate") AND (compression OR quantization)
19. (inner product search OR nearest neighbor) AND quantization AND (database size) AND (recall OR extreme value OR theory) -> 0
20. ("modern Hopfield" OR "dense associative memory") AND noise AND capacity [>=2023]
21. (attention OR softmax) AND (noise OR quantization) AND (needle OR retrieval) AND ("effective context" OR "usable context" OR "critical length") [>=2025]
22. (quantization) AND attention AND ("number of tokens" OR "number of keys") AND (bound OR theorem) [>=2026] -> 0
23. quantization AND (attention OR KV cache) AND (top-1 OR argmax OR ranking OR rank flip) AND (length OR tokens) [>=2025-06]
24. (needle OR NIAH OR RULER) AND (quantization OR KV cache compression) AND (retrieval head OR induction head) [>=2025]
25. WebSearch: KV cache quantization accuracy degrades with context length needle retrieval theory bits log context length
26. WebSearch: vLLM / llama.cpp FP8 KV cache long context needle accuracy drop (this found the vLLM blog; fetched with curl)
