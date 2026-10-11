# XS-V prior-art check: attention as a reconstruction filter for V (subtractive dither + token-axis sigma-delta / error feedback)

Date of check: 2026-10-11. Verdict: **survives_narrowed**. Novelty 3/5.

## What was searched (~20 arXiv API queries + 3 web searches)
- KV/value cache + {dither, dithered, stochastic rounding} (2025+); KV/value cache + {error diffusion, noise shaping, sigma-delta, delta-sigma, error feedback} (2025+) -> **0 hits**
- KV cache + {correlated error, systematic error, bias}; attention + quantization noise + {averag, cancel}; KV + error compensation + {sequential, token-wise}
- {sigma-delta, noise shaping, error diffusion} + {transformer, LLM, attention} (2025+, and all-years by relevance)
- {error feedback, error carry} + {KV, cache, activation, recurrent state}; {subtractive dither, dithered quantizer} + {neural, transformer, LLM}
- 2026-only: KV + {GPTQ, Hessian}; KV + {token axis, across tokens, inter-token}; KV + {deadzone, zero-point, constant channel}; {matrix product, inner product} + {dither, noise shaping}; {blue noise, low-pass} + quantization + attention; KV + error accumulation; {sigma-delta, error feedback, dither} + {Mamba, recurrent state}
- KV + {DPCM, predictive/delta coding}; title:{stochastic rounding, dither, noise shaping} + {LLM, transformer}; GPFQ / sigma-delta NN quantization (all years)
- Web: "KV cache error feedback noise shaping token axis"; GitHub/vLLM/SGLang/llama.cpp dither/SR in KV; "noise shaping|error diffusion|sigma-delta KV cache 2026" (extended). No implementation or blog found.

## Papers read (full text grepped)
| id | title | date | relation |
|---|---|---|---|
| 2607.28699 | WitCert: Sound Runtime Risk Observability and Gating for KV-Cache Quantization | 2026-07-30 | Implements zero-storage Philox-keyed subtractive dither for KV (INT8) and **measures it as quality neutral on real V** |
| 2609.39185 | Low-Discrepancy Dither for Quantized Recurrent State Caches | 2026-09-30 | Dither for SSM state; dismisses sigma-delta for storage reasons; says KV errors "are not fed back" |
| 2608.14191 | KV Cache Compression Through the Lens of Transform Coding (AATC) | 2026-08-14 | Assumes token-white errors; explicitly says exploiting token dependence is "intractable" |
| 2607.23390 | When Can Depth Replace Precision? | 2026-07-25 | Error feedback with carry across depth/microsteps, not KV |
| 2609.05641 | Nuclear-Norm Lower Bound for Dithered Scalar Quantization of Matrix Products | 2026-09-04 | Dithered quantization of matrix products under independent noise; no shaping, no KV |
| 2610.01889 | Stochastic Rounding in Low-Precision Transformer Inference (GPT-2) | 2026-10-01 | SR vs RN at arithmetic sites; sqrt(n) vs n reduction error; not KV storage |
| 2606.03458 | KVarN | 2026-06-02 | Error accumulation across decode steps from token scales; no correlation/shaping analysis |
| 2609.26425 | QuantWM (2-bit KV, video world models) | 2026-09-22 | Low-rank K compensation; no token-axis feedback |
| 2605.13810 | Provable Quantization with RHT | 2026-05-13 | Dithered RHT quantizer unbiasedness/MSE; no attention averaging |
| 2605.26266, 2609.15810, 2606.02288, 2609.04490, 2602.08005 | (abstract/grep level) | 2026 | K Jensen-bias correction; V block-mean smoothing; sink bias vectors; recurrent write-back error feedback; residual KV |

## Exact overlapping statements
- WitCert 2607.28699 l.264-266: "We quantize with x̂ = s (round(x/s + ξ) − ξ) where the dither ξ ∼ U[−1/2, 1/2) is regenerated deterministically at read time." l.866: dither "drawn by Philox4x3210 from a five-tuple counter (request, layer, KV head, token, channel)"; l.631: "dither reconstructed from Philox and therefore stored at zero cost". **So zero-storage hash-seeded subtractive dither for the KV cache is already published and implemented in a fused kernel.**
- WitCert l.1240-1268 (Table 9, Qwen2.5-7B, 8k, 8-bit K+V, real V on real attention output): Rel. output error RTN 2.751e-2, dither 2.717e-2, RTN+outlier 1.995e-2, dither+outlier 1.951e-2. "the gain from dither itself is only 1.01×, i.e. quality neutral"; an earlier "1.7–1.8× better than RTN" claim "is retracted". **This is a published empirical null for the averaging-gain hypothesis (a) at 8 bits.** It does not test 2-4 bits.
- 2609.39185 l.594-597: sigma-delta "need[s] the previous rounding error to be stored, which doubles the size of a state cache". l.603-604: for attention models "key–value cache quantization ..., where errors are not fed back." (premise of the gap claim confirmed verbatim)
- AATC 2608.14191 l.409-412: Assumption 1, errors "mutually independent across tokens i and channels c"; l.437-441: "cross-token independence is an idealization: neighboring KV representations are correlated... Exploiting this dependence, however, would require a computationally intractable token varying bit allocation." (premise confirmed; AATC names the gap but proposes nothing for it)
- 2607.23390 l.639-644: classical sigma-delta (Daubechies-DeVore 2003, Güntürk 2003), sigma-delta NNs (O'Connor & Welling 2017), SDQ-LLM; "a carry state makes the accumulated rounding error telescope". Telescoping carry is classical; nothing on KV.
- 2610.01889 abstract: SR error envelope O(sqrt(n) u) vs O(n u) for RN in reductions. Classical numerical-analysis version of "dither averages down, RTN can be coherent".

## Premise check
1. "WitCert uses dither only as a certification device; the averaging gain is never analyzed": **half true.** It is not analyzed theoretically, but WitCert measured it on real V and found 1.01x (Table 9). The candidate has to cite this and explain it (high rate, 8 bits, makes RTN errors effectively white already) instead of presenting the gap as unexamined.
2. "2605.13810 covers only unbiasedness/MSE of dithered RHT": consistent with the abstract and grep (no attention or averaging analysis).
3. "2606.02288 reports near-constant bias vectors in keys": **stretched.** The abstract says spike-carrying (sink) tokens converge to constant vectors. That is a few tokens, not near-constant channels across all tokens. It is not direct evidence for coherent per-channel V error.
4. "2609.39185 storage objection does not apply to write-once KV": correct. The carry exists only at write time, is O(d) per head, and is discarded.
5. "AATC assumes white token-wise noise": confirmed (Assumption 1).
6. **Deadzone mechanism (b) is quantizer-dependent.** With the per-token asymmetric min-max V quantizer (KIVI-style, zero-point), a constant channel c is not mapped to 0. Its error depends on the per-token (scale, zero) pair, which varies across tokens. So error coherence must be measured, not assumed. It holds for symmetric/absmax and shared-scale formats (e.g. FP8 per-tensor, NVFP4 block scale with small blocks).
7. **Hidden cost of sigma-delta not in the hypothesis: range headroom.** v_t - u_{t-1} exceeds the range of v_t by up to Δ/2 on each side. At 2 bits with per-token min-max (3 intervals), Δ grows by about 4/3, so per-element variance grows by about 1.78x. That comes on top of the 2x from differencing. In my toy simulation (scratchpad sd_check.py), per-element V MSE at 2 bits was RTN 0.304, dither 0.298, SD 0.763 (2.5x). At 4 bits it was 1.21e-2, 1.23e-2 and 2.50e-2. The "at most 2x worse for spiky heads" claim is therefore optimistic at 2 bits: the worst case is about 3.5x, and the selection threshold on R must include the headroom factor. Under diffuse attention the gain still dominates. Toy output error at 2 bits: uniform-p RTN 0.52, dither 1.7e-2, SD 1.3e-5; last-256 local p RTN 0.69, dither 0.13, SD 1.7e-3; sink+diffuse about equal.
8. Math core (c) is correct: e_t = ε_t - ε_{t-1} gives an output error of Σ ε_t (p_t - p_{t+1}) + boundary, with variance σ² Σ(Δp)². One-hot p gives 2σ² vs σ² (factor 2, before headroom). Uniform p over N gives about 2σ²/N² vs σ²/N.

## Verdict: survives_narrowed
- **Preempted / weak:** part (a), zero-storage hash-seeded subtractive dither for KV (WitCert, implemented, including the kernel). Its 1/N_eff averaging is textbook dither theory (Schuchman 1964; Gray & Stockham 1993) and SR theory (Connolly-Higham-Mary). The quality benefit has a published empirical null at INT8 (WitCert Table 9). Part (b), the deadzone/N_eff threshold, is a small corollary and depends on the quantizer.
- **Remains novel (no paper found):** part (c), first-order (or order-L) **token-axis error feedback / sigma-delta at KV write time**. Attention probabilities act as the reconstruction low-pass filter, with output error σ² Σ(Δp)² instead of σ² Σ p². Also new: the zero-persistent-storage argument that refutes the 2609.39185 objection for write-once caches, the per-head R = Σ(Δp)²/Σp² selection rule, and the V-only prediction. AATC explicitly calls exploiting token correlation "intractable". This is a cheap way to exploit token structure of the *error*, not of the signal.

## Significance (main-track reviewer view)
Interesting and cheap, with a clean one-line theory. A reviewer would ask:
1. Do real LLM attention rows have small Σ(Δp)² relative to Σ p²? Sinks, retrieval heads and RoPE-induced oscillatory attention may break smoothness. The method's value is an empirical question about attention spectra.
2. Does it beat Hadamard-rotated 2-bit V (KIVI/KVarN/TurboQuant) at equal bits on long-context and reasoning tasks, not just output MSE?
3. Prefill writes are parallel, but the carry is sequential along tokens, so a scan is needed in the write kernel. Interaction with paged/chunked prefill and eviction: evicting a token breaks the telescoping and leaves one residual of size ε.
4. WitCert's null result on dither must be engaged with directly.

Without downstream gains at 2 bits it is a workshop paper. With a 0.5-1 bit equivalent V gain on real benchmarks it is a credible main-track contribution.

## Sharpened hypothesis
For write-once KV caches, first-order token-axis error feedback on V (q_t = Q(v_t - u_{t-1}), with the carry kept only in the write kernel, zero persistent bytes) turns attention-output error from σ'² Σ p_t² into σ'² Σ (p_t - p_{t+1})². Here σ'² includes the range-headroom inflation (about 1.78x at 2-bit per-token min-max). Prediction: on heads where measured R·(σ'²/σ²) < 1 (diffuse or local heads; sink token kept in high precision), shaped 2-bit V matches unshaped 3-bit V in attention-output MSE and downstream accuracy. The gain is additive to Hadamard rotation and does not come from dither, which WitCert shows is neutral at 8 bits. Applying it to K gives no gain.
