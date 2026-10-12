# Idea 2 — ΣΔ (error-feedback) quantization of SSM/hybrid recurrent state: keyword + adjacent-field sweep

Date: 2026-10-10. Scope: arXiv 2023-01 → 2026-10 (date-sorted pass from 2025-06), plus web/GitHub/ACL for non-arXiv items.
Idea under test (report §아이디어 2): (i) RTN re-quantization of h_t = a_t h_{t-1} + u_t freezes slow channels (deadband); (ii) stochastic rounding (SR) random-walks with Var ≈ Δ²t/12; (iii) first-order error feedback (ΣΔ) gives a deterministic bound |d_t| ≤ Δ uniform in t and a ∈ [0,1) ("정리 2"); (iv) b_r-bit residual; (v) decay-rate bit allocation b_i = ½log₂(σ²/((1−a_i²)θ)); (vi) joint KV+state water-filling for hybrids. Report claims "학술 논문으로 SSM 상태 양자화를 다룬 것은 0건".

## Bottom line

* The "0 papers on SSM/recurrent **state** quantization" claim is **false as of 2026-10**. At least seven arXiv papers (2608.27513 DAMP, 2609.30950, 2609.38166 LeapQuant, 2609.38169 STEPQuant, 2609.39185 Weyl dither, 2609.04490 write-back, 2609.04098) plus Q-Mamba (ACL 2025 Findings, W8A8**H4** = 4-bit state cache) quantize the recurrent state directly. All but 2609.04490 are Aug–Sep 2026.
* **Claims (i) and (ii) are already proved (idealized) and measured.** 2609.39185 Lemma 1 ("write erasure") = deadband; Corollary 1(a) RTN error grows linearly sT·min(f,1−f); (b) SR error ∝ s√(T f(1−f)); (c) Weyl dither O(s log T). 2609.04490 coins "write deadband" and "recurrent write margin". 2603.16731 and 2505.00347 (SOLO Thm 3.1 "signal swamping") prove the same freeze condition for EMA optimizer states.
* **Claim (v) decay-rate bit allocation is preempted.** 2609.30950 Proposition 1 derives the observability-Gramian weight Γ_i = ρ_i/(C·expm1(2s_i)) (= ρ_i a²/(C(1−a²)) for a = e^{−s}) and the high-rate rule b_i = b̄ + ½log₂(c_i/GM(c)) — the same reverse water-filling. DAMP uses w = 1/(1−a²) (capped); STEPQuant uses L_u = Σ_j exp(2jℓ_u).
* **Claim (iii)/(iv) — the ΣΔ bound and its b_r-bit residual analysis — is NOT in any paper.** 2609.39185 explicitly declines error feedback ("they need the previous rounding error to be stored, which doubles the size of a state cache; at equal storage only the choice of threshold remains"). 2609.04490 *does* implement error feedback and k∈{2,3,4}-bit "quantized residual memory" post-training, but on a 32-unit GRU for fluorescence-lifetime imaging, with no bound, no decay analysis, no SSM/LLM. A vLLM contributor (issue #55196, comment 2026-09-15) self-reports INT8 state + error feedback on Qwen3.5-4B GDN (GSM8K 81→41→81) and an INT8+int4-residual variant — exactly configuration (d) of the report — with no paper and no theory.
* **New baseline to beat:** the Weyl (golden-ratio) dither of 2609.39185, not SR — it beats SR by 11–49% KL (median 31%, ≈ +0.25 bit) at zero storage cost. Also "fewer re-quantizations" alternatives: Nemotron 3 Ultra periodic checkpointing (C=8) and LeapQuant per-window quantization (window 16 → 39–100× lower state MSE at 64K) must be baselines.
* **Possible contradiction for delta-rule states:** 2609.04098 shows, for GDN, state error from W4A4 noise plateaus (relS 12.96% @256 → 12.31% @32K) and a 1% impulse decays to 1/e within 80–1,382 steps although decay horizons are 44K–62K, because every write erases error along the current key. The idea's a→1 random-walk/deadband story holds cleanly for diagonal SSM without erasure (Mamba-1/2); for GDN/KDA the erasure term must be in the model (2609.30950 note: errors orthogonal to repeated keys are never erased, so the plateau is not universal).

## Candidates (ranked)

### 1. 2609.39185 — Low-Discrepancy Dither for Quantized Recurrent State Caches (Khilar, 2026-09-30) — partially_preempts, P1
* Setup: h_t = a_t h_{t−1} + w_t; stored ĥ_t = Q_{u_t}(a_t ĥ_{t−1} + w_t); compares RTN / SR / Weyl dither on 6 Mamba (130M–1.4B) + 3 hybrids (Granite 4.0-H 350M/1B, Falcon-H1 0.5B), INT4/6/7/8, FP8, BF16; single T4.
* **Lemma 1 (Write erasure)**: if |δ_t| = |(a_t−1)ĥ_{t−1}+w_t| < s/2, RTN leaves the stored value unchanged — this is the report's deadband condition verbatim.
* **Proposition 1 / Corollary 1** (a_t = 1, constant write): |e_T| ≤ s·T·D*_T (star discrepancy); RTN |e_T| = sT·min(f,1−f) (linear); SR typical size s√(T f(1−f)); Weyl ≤ C s(1+log T). **Lemma 2**: per-write MSE s²/12 (RTN) vs s²/6 (SR/Weyl) when f uniform.
* Appendix: general e_t = a_t e_{t−1} + r_t, summation by parts |e_T| ≤ max_m |R_m| (tail residual sums) — the same telescoping skeleton the report uses for g_t; "we do not claim a bound on the windowed residual sums in the general case".
* Numbers (Table 1, decode regime): Weyl < SR in all 15 cells, 11–49% lower KL (median 31%); RTN worst on all Mamba cells, 1.4–17× worse than Weyl; over 4096 steps on hybrids RTN loses by up to 400×; Weyl@INTk ≈ SR@INT(k+0.25).
* Related-work §: "Noise shaping and sigma–delta modulation … are the classical answer to quantization inside a feedback loop, but they need the previous rounding error to be stored, which doubles the size of a state cache" → ΣΔ left open; the report's b_r-bit residual is the direct answer. A "memory-gated rule" (RTN for τ<τ*, dither otherwise) was tried and failed pre-registered tests.
* Threat: preempts (i), (ii) and the 1/(1−a) time-constant framing; leaves (iii) open and sets its cost objection.

### 2. 2609.04490 — When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference (Erbas, Intes, Pandey, 2026-09-03) — partially_preempts, P1
* Defines "recurrent-state write-back", "recurrent write margin" M_{t,j} = 2|δ_{t,j}|/Δ_B, "write deadband" (M<1), deadband fraction.
* Post-training interventions on a fixed 32-unit GRU (Seq2SeqLite, 6,627 params, FLI τ1/τ2 estimation) and an LSTM: continuous, SR, **error feedback**, **k-bit quantized residual memory (k=2,3,4)**, direction memory (k-bit sign counter, new).
* Numbers: deterministic 4-bit state → ~70× / ~300× error on τ1/τ2; at P2F error feedback 0.36/0.37 ns, 2-bit residual 0.34/0.40, 3-bit direction memory 0.34/0.34 ns. 8-bit reference forced to 4-bit: 1.89/2.60 ns, deadband fraction 99.40%; EF recovers 0.34/0.46, 4-bit residual 0.34/0.45, 4-bit direction memory 0.49/0.58 (native 8-bit 0.20/0.22). LSTM: cell state far more sensitive than hidden state.
* No theorem/lemma/bound (grep: Theorem/Proposition/Lemma = 0), no decay-rate analysis, no SSM/LLM, no T-growth curves. Shows the mechanism (d) empirically works on a toy recurrent model.

### 3. 2609.30950 — Low-Bit Recurrent States in Hybrid Language Models (Chen & He, 2026-09-25) — partially_preempts (bit allocation), P1
* Model S_t = (I − k_t a_tᵀ)D_t S_{t−1} + k_t(w_t⊙v_t)ᵀ; write-back every C tokens.
* **Proposition 1 (time-averaged readout distortion)**: under zero-mean noise uncorrelated across rows/write-backs, steady-state readout MSE = Σ_i Γ_i σ_i², Γ_i = (1/C)Σ_{ℓ≥1}E[(Φ_ℓᵀq)_i²]; constant diagonal decay without erasure → Γ_i = ρ_i/(C·expm1(2s_i)) (= ρ_i a_i²/(C(1−a_i²))). Allocation: minimize Σ c_i 2^{−2b_i} → **b_i = b̄ + ½log₂(c_i/GM(c))** — the report's reverse water-filling rule with the +½log₂(1/(1−a²)) term, calibration-free.
* Also: "Deterministic rounding at C = 1 can suppress updates below half a quantization step … lag-one error correlations up to 0.23 for s_i ≤ 10⁻³ at three to five bits … readout error 0.17–2.4× that of independent noise … model overestimates the gain over uniform four-bit by 1.7–2.6×" — i.e., they know the Gaussian-noise model breaks in the deadband regime and leave it as a heuristic. This is the gap the ΣΔ analysis could close (ΣΔ makes the noise model valid).
* Results: 4-bit mean payload cuts excess NLL 3.3–27.9× vs best of 7 baselines on 3 hybrids (Kimi-Linear-REAP-35B-A3B, Qwen3.x, Nemotron-3-Nano-4B in tables); 6 bits within 0.005 nats of FP32; log-quantized decay rates; gains diminish with less frequent write-back. No error feedback (count 0).

### 4. vLLM issue #55196 "[RFC]: fp8 KV cache gives little to no memory benefit on Mamba/GDN hybrid models" (shivasathishs-rp, opened 2026-09-03) — partially_preempts (engineering), P1
* Capacity ratios (not speedups): Qwen2.5-3B 2.00×; Falcon-H1-1.5B 1.00× @512, 1.00× @1024, 1.51× @2048, 1.76× @8192, 1.84× @32768 (603,229 bf16 vs 1,111,381 fp8 tokens at 32k). `--mamba-cache-dtype` only accepts auto/float32/float16/bfloat16.
* Comment tomalmog (2026-09-15): INT8 + error-feedback for GDN path on Qwen3.5-4B: "INT8 uniform state collapses GSM8K 81% → 41%"; error feedback "recovers all 40 points back to 81%"; INT8 state + int4 residual: "1.28× memory vs bf16" (direction ambiguous in the page), 37/50 vs 38/50 GSM8K; kernel moves 22% less traffic. zz774ever (09-14) proposes an FP8 state path. Maintainer (09-25) invited PRs.
* No paper, no theory, GDN not Mamba. Confirms the report's quoted RFC; the exact (INT8 + 4-bit residual) recipe is already community-implemented.

### 5. 2608.27513 — DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization (Zhang et al., Meituan/SCUT, 2026-08-27, v2 09-30) — partially_preempts, P2
* "first study of post-training recurrent-state quantization for GDN and KDA". Uniform INT8/FP8 degrade reasoning; INT4/NVFP4 collapse to ~0. Decay-based error-retention weight w = 1/(1−a_eff²) (capped), a_eff = exp(E_t log a_t); keeps high-risk key channels FP16, rest INT8 (9.9 bits/value) → near-FP32 on Qwen3.6-35B, Kimi-Linear-48B, Kimi-K3 (6 benchmarks); SGLang −69.1% state storage, 2.59× update kernel, −19.0% TPOT. Slow-decay heads/channels stable across tasks (supports offline allocation). No EF/SR/dither.

### 6. 2609.38169 — STEPQuant (Yao et al., 2026-09-29) — partially_preempts, P2
* **Proposition 1**: E_t = (I − β_t k_t k_tᵀ)D_t E_{t−1} + ε_t = A_t E_{t−1} + ε_t, ŷ_t − y_t = E_{t−1}ᵀA_tᵀq_t; ‖A_t‖₂ ≤ ‖D_t‖₂ ≤ 1. Lifetime weight L_u = Σ_{j<H} exp(2jℓ_u) (≈ 1/(1−a²)); Spearman ρ ≈ 0.80 between head half-life and accumulated error (Qwen3.8-27B, INT6, C4). Spatial: key-row × value-column scale fitting. 6-bit ≈ FP32-state; 4-bit > uniform INT8; SGLang kernels, >5× compression, −68.7% serving memory. No EF/SR.

### 7. 2609.38166 — LeapQuant (Pan, Xi, … Han, Keutzer, Stoica, 2026-09-29) — supports / competing mechanism, P2
* Figure 3 (Qwen3.5-9B, PG-19, 64K decoded tokens): per-step quantization error "grows steadily with the context length, even for BF16. Stochastic rounding helps little, as it only lowers the BF16 error at 64K by 3× and makes INT8 diverge." Quantizing once per 16-token window lowers error at 64K by 100× (BF16) and 39× (INT8). Compensator tokens for outliers; residual smoothing. Near-lossless INT8; 2.05–3.70× kernel, 1.47× E2E on B200/RTX PRO 6000/5090.
* Supports claim (ii) (SR diverges at INT8). Offers a non-EF fix (fewer re-quantizations) that must be a baseline.

### 8. 2606.15007 — Nemotron 3 Ultra (NVIDIA, 2026-06-12) §4.5 + Table 16 — supports, P2
* Table 16 (Nemotron 3 Super NVFP4, emulated; mean accuracy drop / verbosity increase vs FP32 cache): FP16 RTN 1.07% / +9.98%; FP16 SR −0.26% / +0.36%; INT8 RTN (block 128) 1.42% / +9.91%; INT8 SR −0.15% / +1.28%; FP8 RTN 4.68% / +25.17%; FP8 SR 0.61% / +3.12%. With checkpointing C=8: FP16 RTN 0.03% / +3.07%; FP16 SR 0.19% / +0.40%; INT8 RTN 0.30% / +4.16%; INT8 SR −0.42% / +0.60%; FP8 RTN 0.46% / +3.69%; FP8 SR 0.29% / +1.41%.
* "mantissa precision and stochastic rounding are key ingredients"; FP8 E4M3 degrades; proposes periodic cache checkpointing with activation replay; 32-bit Mamba cache > FP8 KV cache up to 64K tokens (Fig. 14). Ships FP16+SR.
* Supports the deadband diagnosis (RTN verbosity +10%, SR fixes) but NVIDIA's explanation is "RTNE bias accumulates coherently"; no deadband/EF.

### 9. 2604.12374 — Nemotron 3 Super (NVIDIA, 2026-04-14) §4.3 Table 9 — supports, P3
* FP16 cast → up to +40% verbosity with W8A8, +37% with W16A16; INT16 with FP32 per-block (128) scales fixes it; FP16 + SR (Philox<5>) fixes it; Blackwell PTX SR instruction. (Report already cites; numbers confirmed.)

### 10. Q-Mamba: Towards more efficient Mamba models via post-training quantization (ACL 2025 Findings, aclanthology 2025.findings-acl.551; NOT on arXiv) — partially_preempts (existence), P2
* Abstract: state caches "contribute significantly to memory consumption"; Decoupled Scale Quantization (separate scales on state and channel dims) + Efficient Selectivity Reconstruction (block-wise reconstruction with parallel scan); Mamba2-2.7B **W8A8H4** (4-bit state caches): 50% memory reduction, 2.13% avg zero-shot degradation. Falsifies "0 papers". No temporal/EF analysis visible in abstract.

### 11. 2609.04098 — Why Gated DeltaNet Survives 4-Bit Quantization (Kozyrev & Maiboroda, 2026-09-03) — contradicts (for delta-rule states), P2
* NVFP4 W4A4 on all 496 linear layers of Qwen3.8-27B incl. GDN; matches BF16 within seed noise (5-task avg −0.52). Mechanism (iii): "the delta-rule recurrence bounds injected quantization noise at a flat plateau over 32K tokens" — relS 12.96% @256 vs 12.31% @32,768 (plateau 12.6%, max 14.9%); 1% impulse at t=1024 → 1/e in 80–1,382 steps, 1/10 in ~2,200–2,900, vs decay-implied horizons 1/(1−α) of 44K–62K; "every write overwrites the state along the current key direction". Per-position NLL gap +0.081 nats first half → +0.011 second half.
* Caveat: noise enters through quantized projections, not state re-quantization; still, it argues against "a→1 ⇒ random walk" for GDN/KDA. Idea should restrict/extend the model with the erasure term.

### 12. 2603.16731 — Topollai & Choromanska, Quantization of Optimizer States: State Staleness and Resets (2026-03-17, v3 09-27) — supports, P2
* Predictive model of one-step stalling probability for quantized EMA, buildup over time; SR lowers stalled fraction; theory-guided reset periods. (Already cited by the report.) Same first-order IIR; no EF.

### 13. 2505.00347 — SOLO: Pushing the Limits of Low-Bit Optimizers: A Focus on EMA Dynamics (Xu et al., 2025-05-01) — supports, P3
* **Theorem 3.1 (Signal Swamping)**: quantized EMA state remains unchanged if quantizer radius r ≥ (1−β)·|…| — the deadband inequality with β ≡ a; worse as β→1; fix = logarithmic levels for unsigned EMA + precision-specific momentum; **Theorem 3.6** on SR max radius; 2–3-bit Adam states. Not cited by the report; it is the closest prior "deadband theorem".

### 14. 2506.01404 — Zheng et al., Quantitative Error Feedback for Quantization Noise Reduction of Filtering over Graphs (IEEE TSP, v2 2026-05-11) — supports (classical theory source), P3
* Error-spectrum-shaping from state-space digital filters applied to quantized graph filtering; **Theorem 2**: exact noise reduction for deterministic IIR graph filtering over an infinite horizon with closed-form optimal error-feedback coefficients; extensions to random graphs and asynchronous updates. The report's "정리 2" is a scalar first-order special case of this family (classical ΣΔ/IIR theory); worth citing to be honest about novelty of the bound itself.

### 15. 2608.30386 — DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving (Yu et al., 2026-08-31) — tangential, P4
* "Retention horizons" per head/channel derived from weights; omit short-horizon units from prefix checkpoints (2.63× KDA checkpoint compression; TTFT −42.6%). Same decay structure, no quantization theory.

Also seen (not ranked): 2608.15533 DeltaLog (deferred materialization: base state + update log, −7.83× write traffic) and 2609.33477 suffix-replay prefix caching — both reduce the number of state write-backs; 2610.12444 (AdamW 4-bit, preconditioner-space SR, 2026-10-08) — optimizer-state recurrence; 2608.21952 SSDi8 — persistent INT8 path for SSD with per-channel error-correction term (activations/weights, not state); 2609.30820 looped transformers "feedback exposure"; 2506.12480 small-scale S4D (state x and A matrix most sensitive, PTQ 40%→QAT 96% sMNIST); 2507.06079 QS4D; 2606.18114 Ternary Mamba (report already cites). Searches for "sigma-delta"/"noise shaping"/"error diffusion" × recurrent/Mamba returned only halftoning, MIMO ADC, spiking and graph-filter papers — no ML paper applies ΣΔ to a recurrent state cache.

## What survives for Idea 2 (revised)

1. **Theorem**: the t- and a-uniform |d_t| ≤ Δ bound for first-order error feedback on a diagonal gated recurrence, with the b_r-bit-residual corollary (worst case Δ2^{−(b_r+1)}/(1−a), stochastic Δ2^{−b_r}/√(24(1−a))). Nobody states this for recurrent-state caches; 2609.39185 provides the exact summation-by-parts skeleton and explicitly stops short. Must position relative to classical ΣΔ/IIR error-spectrum shaping (Schreier–Temes; Zheng 2506.01404).
2. **Equal-storage comparison** that 2609.39185 says is the only fair one: INT(k) + b_r-bit residual vs INT(k+b_r) Weyl/SR/RTN, vs checkpointing C=8 (Nemotron Ultra) and per-window (LeapQuant). Baseline (e) in the report must include Weyl dither.
3. **Noise-model repair for allocation**: 2609.30950 admits its Gaussian model misses deadband correlations (readout error 0.17–2.4× model; gain overestimated 1.7–2.6×). ΣΔ makes the residual noise white-ish (first-difference shaped), so decay-weighted water-filling (their rule) becomes valid — a joint "ΣΔ + Gramian allocation" story is new.
4. **Mamba-style (no erasure) vs delta-rule (erasure)**: predict the ΣΔ gain is large for Mamba-1/2 and Nemotron-H and small for GDN/KDA (2609.04098 plateau); that split is itself a testable, unclaimed hypothesis.
5. Hybrid joint KV+state water-filling (vi) — not found in any paper; the vLLM RFC page-size coupling is the practical motivation.

Drop/rewrite: the "0 papers" sentence; "deadband is new" framing; the standalone decay-rate bit-allocation contribution (now a known rule).

## Queries run

arXiv API (python3 -I scripts/arxiv_search.py; ranges 20230101–20261031 unless noted):
1. `abs:Mamba AND abs:quantization AND (abs:state OR abs:"hidden state" OR abs:recurrent)`
2. `abs:"state space" AND abs:quantization AND abs:"error feedback"`
3. `abs:"sigma-delta" AND (abs:"neural network" OR abs:recurrent OR abs:quantization)`
4. `abs:"noise shaping" AND abs:quantization AND (abs:recurrent OR abs:"neural network" OR abs:LLM OR abs:"language model")`
5. `abs:"error feedback" AND abs:quantization AND (abs:RNN OR abs:recurrent OR abs:LSTM OR abs:"state space" OR abs:Mamba)`
6. `abs:hybrid AND abs:Mamba AND abs:"KV cache" AND (abs:quantization OR abs:quantized OR abs:FP8 OR abs:INT8)`
7. `(abs:stale OR abs:staleness OR abs:freeze OR abs:frozen) AND abs:quantization AND (abs:"exponential moving average" OR abs:EMA OR abs:"optimizer state")`
8. `abs:"linear attention" AND (abs:quantization OR abs:quantized OR abs:"low-bit") AND abs:state`
9. `(abs:"recurrent state" OR abs:"SSM state" OR abs:"state cache" OR abs:"hidden state") AND (abs:quantization OR abs:quantized OR abs:quantizing OR abs:"low-precision") AND (abs:Mamba OR abs:"state space" OR abs:"linear attention" OR abs:DeltaNet OR abs:recurrent)` --sort date --from 20250601
10. `(abs:Mamba OR abs:"state space model" OR abs:SSM) AND (abs:quantization OR abs:quantized OR abs:INT8 OR abs:FP8 OR abs:INT4 OR abs:"low-bit")` --sort date --from 20250601
11. `(abs:"error feedback" OR abs:"error-feedback" OR abs:"error diffusion" OR abs:"sigma-delta" OR abs:"noise shaping" OR abs:dither OR abs:"stochastic rounding") AND (abs:recurrent OR abs:Mamba OR abs:"state space" OR abs:"linear attention" OR abs:RNN OR abs:"delta rule" OR abs:DeltaNet)` --sort date --from 20250601
12. `(abs:hybrid OR abs:Nemotron OR abs:Falcon-H1 OR abs:"Qwen3-Next" OR abs:Jamba) AND (abs:Mamba OR abs:"state space" OR abs:"linear attention" OR abs:DeltaNet) AND (abs:quantization OR abs:quantized OR abs:FP8 OR abs:INT8)` --sort date --from 20250601
13. `(abs:deadband OR abs:"dead band" OR abs:"dead-band" OR abs:"limit cycle" OR abs:"limit cycles") AND (abs:quantization OR abs:quantized OR abs:"finite precision" OR abs:"fixed-point") AND (abs:recursive OR abs:IIR OR abs:filter OR abs:"state space")`
14. `(abs:"stochastic rounding") AND (abs:"random walk" OR abs:accumulate OR abs:accumulation OR abs:recursive OR abs:recurrence OR abs:"moving average")`
15. `(abs:"error feedback" OR abs:"error diffusion" OR abs:"sigma delta" OR abs:"sigma-delta" OR abs:"noise shaping") AND (abs:"uniform bound" OR abs:"uniformly bounded" OR abs:"bounded error" OR abs:"error bound" OR abs:stability OR abs:telescoping)` --sort relevance
16. `(abs:"quantized state" OR abs:"state quantization" OR abs:"quantized states") AND (abs:recursive OR abs:recurrence OR abs:dynamics OR abs:"exponential decay" OR abs:"error accumulation")`
17. `(abs:"water-filling" OR abs:"waterfilling" OR abs:"bit allocation") AND (abs:"Gauss-Markov" OR abs:"AR(1)" OR abs:autoregressive OR abs:"first-order" OR abs:"decay rate" OR abs:"forgetting") AND (abs:quantization OR abs:"rate-distortion" OR abs:"rate distortion")`
18. `(abs:dither OR abs:dithering OR abs:"dithered quantization") AND (abs:"exponential moving average" OR abs:EMA OR abs:recurrent OR abs:"recurrent state" OR abs:"state space") AND NOT abs:image`
19. `(abs:"error feedback" OR abs:"error compensation") AND (abs:"momentum" OR abs:"optimizer state" OR abs:"exponential moving average") AND (abs:quantization OR abs:"low-precision" OR abs:compression)`
20. `(abs:"quantized" AND abs:"error feedback") AND (abs:consensus OR abs:"linear system" OR abs:"linear systems" OR abs:"control") AND (abs:"bounded" OR abs:"input-to-state")`
21. `(abs:"write-back" OR abs:"writeback" OR abs:"deferred write" OR abs:checkpointing OR abs:"activation replay") AND (abs:"recurrent state" OR abs:"SSM state" OR abs:"Mamba" OR abs:"linear attention") AND (abs:quantization OR abs:quantized OR abs:cache)` --sort date --from 20250101
22. `(abs:"leaky integrator" OR abs:"exponential smoothing" OR abs:"first-order IIR" OR abs:"first-order recursive") AND (abs:quantization OR abs:quantized OR abs:"finite precision" OR abs:"fixed-point") AND (abs:"error feedback" OR abs:"sigma-delta" OR abs:dither OR abs:"limit cycle" OR abs:bound)`
23. `cat:cs.IT AND (abs:"sigma-delta" OR abs:"sigma delta" OR abs:"noise shaping") AND (abs:"first-order" OR abs:"first order" OR abs:stability OR abs:"error bound")`
24. `(abs:"low-discrepancy" OR abs:"Weyl" OR abs:"quasi-random") AND (abs:rounding OR abs:quantization) AND (abs:"neural" OR abs:recurrent OR abs:"deep learning")`
25. `(abs:"optimizer state" OR abs:"optimizer states" OR abs:momentum) AND (abs:quantization OR abs:quantized OR abs:"low-bit" OR abs:"8-bit" OR abs:"4-bit") AND (abs:"error feedback" OR abs:residual OR abs:stale OR abs:"random walk" OR abs:accumulat)` --sort date --from 20240101
26. `(cat:math.OC OR cat:eess.SY OR cat:cs.SY) AND abs:"error feedback" AND abs:quantiz AND (abs:"uniformly bounded" OR abs:"ultimately bounded" OR abs:"bounded error" OR abs:"input-to-state")` (0 hits)
27. `(abs:"KV cache" OR abs:"KV-cache") AND (abs:"recurrent state" OR abs:"SSM state" OR abs:"Mamba state" OR abs:"state cache") AND (abs:budget OR abs:allocation OR abs:"memory") AND (abs:quantization OR abs:quantized OR abs:FP8)` --sort date --from 20250101
28. `(abs:"rate-distortion" OR abs:"rate distortion") AND (abs:"recurrent" OR abs:"state space" OR abs:"Gauss-Markov" OR abs:"Gauss Markov" OR abs:"AR(1)") AND (abs:quantization OR abs:"bit allocation" OR abs:"bits")`
29. `ti:"Q-Mamba" OR (abs:"Q-Mamba" AND abs:quantization)`; `ti:"more efficient Mamba models" OR (ti:Mamba AND ti:"post-training quantization" AND abs:"state cache") OR (abs:Mamba AND abs:"state cache" AND abs:quantization)` (Q-Mamba not on arXiv)

Web / other:
30. WebSearch `"error feedback" OR "sigma-delta" quantization "recurrent state" Mamba hybrid LLM state cache 2026`
31. WebSearch `vllm issue 55196 mamba cache dtype fp8 int8 ssm state quantization error feedback`
32. WebSearch `SGLang mamba ssm state cache quantization stochastic rounding int8 fp16 Nemotron recurrent state`
33. WebSearch `openreview "recurrent state" quantization "error feedback" OR "noise shaping" OR "dither" linear attention Gated DeltaNet state`
34. WebFetch https://github.com/vllm-project/vllm/issues/55196 ; curl https://aclanthology.org/2025.findings-acl.551/
Full texts fetched (scratchpad/papers/<id>/): 2609.39185, 2609.38169, 2609.38166, 2609.30950, 2609.04490, 2609.04098, 2608.27513, 2608.21952, 2610.11214, 2609.30820, 2604.13440, 2610.12444, 2506.12480, 2506.01404, 2603.16731, 2606.18114, 2505.00347, 2605.06675, 2609.33477, 2606.15007, 2604.12374, 2608.30386, 2608.15533.
