# Low-Precision / Quantized PRE-TRAINING of LLMs: State of the Art and Open Problems (as of 2026-10-10)

**Provenance note for the report writer.** In this session the fetch tool could not resolve any host (arXiv, OpenReview, HF, Semantic Scholar all failed DNS) and the shared web-search budget ran out after 30 queries. Every finding below therefore comes from (i) search-engine extended summaries/snippets of the primary papers (URLs given), or (ii) the researcher's prior knowledge, which is explicitly tagged **[PK: not re-verified this session]**. Numbers tagged PK should be spot-checked against the arXiv PDF before being quoted in a final report. No citation below is invented: where an arXiv ID is uncertain it is marked "ID unverified".

Citation key: arXiv IDs of the form 25xx/26xx are 2025/2026 preprints. "Gap" = question left open by the literature or where papers disagree.

---

## KQ1. FP8 pretraining: instabilities, fixes, and what remains unexplained

### Takeaway
FP8 (E4M3 for all GEMM operands + fine-grained block scaling + FP32/high-precision accumulation) is now production-grade at frontier scale (DeepSeek-V3, NVIDIA 8B/15T MXFP8), but the instability literature is thin and mechanistic only in one case (SwiGLU weight-alignment outlier amplification after ~200B tokens). Scale-factor rounding direction, accumulation precision, and which modules stay in BF16 remain recipe-level knowledge rather than theory, and 2026 reports still show late-training divergences that even a BF16 switch does not cure.

### Cited Findings

**Formats and the Transformer Engine baseline**
- The FP8 formats used by all of the below are E4M3 (3 mantissa bits, max 448) and E5M2 (max 57,344), proposed in Micikevicius et al., "FP8 Formats for Deep Learning" (arXiv 2209.05433, 2022) **[PK: not re-verified]**; the original NVIDIA Transformer Engine recipe used E4M3 for weights/activations, E5M2 for gradients, and per-tensor "delayed scaling" from an amax history — [TE docs](https://docs.nvidia.com/deeplearning/transformer-engine/) **[PK]**.
- Fishman et al. note that FP8 has a much smaller dynamic range than BF16 and that delayed scaling assumes stable statistics across batches, so outliers lead to underflow or overflow when combined with delayed scaling — [arXiv 2409.12517](https://arxiv.org/abs/2409.12517).

**FP8-LM (Peng et al., Microsoft, arXiv 2310.18313, Oct 2023)**
- Proposes an FP8 automatic mixed-precision framework with 8-bit gradients, optimizer states and distributed communication added in stages, three levels of FP8 utilization, open-sourced as MS-AMP — [arXiv 2310.18313](https://arxiv.org/pdf/2310.18313); [emergentmind](https://www.emergentmind.com/papers/2310.18313).
- Reported results differ between versions: v2 abstract cites ~39% lower real memory and 75% faster training than Megatron-LM BF16 on GPT-175B/H100; the HF page (v1) lists 42% memory savings, 64% speedup, 17% over Transformer Engine — [arXiv PDF](https://arxiv.org/pdf/2310.18313); [HF paper page](https://huggingface.co/papers/2310.18313).
- Key techniques (PK): "precision decoupling" (FP8 first moment, FP16 second moment, FP16 master weights with tensor scaling) and "automatic scaling" of gradient all-reduce to avoid under/overflow — a survey describes precision decoupling and automatic scaling as designed to solve FP8's underflow/overflow issues — [arXiv 2407.20018](https://arxiv.org/pdf/2407.20018).
- Stated limitation/critique of prior work: end-to-end speed-up, memory and communication savings are "very limited" when master weights and gradients stay in high precision — [arXiv 2310.18313](https://arxiv.org/pdf/2310.18313). No limitations section surfaced in search.
- Critical note: GPT-175B results are throughput runs, not converged-quality runs; the quality claims are for ≤13B-ish models over ~100B tokens **[PK]**.

**DeepSeek-V3 FP8 recipe (DeepSeek-AI, arXiv 2412.19437, Dec 2024) [ID PK]**
- Activations quantized in 1×128 tiles, weights in 128×128 blocks, with online (per-step) max-abs computed per block; E4M3 for both forward and backward GEMMs — [Insights into DeepSeek-V3 hardware paper, arXiv 2505.09343](https://arxiv.org/pdf/2505.09343); [survey arXiv 2505.01043](https://arxiv.org/pdf/2505.01043).
- Accumulation promoted to FP32 on CUDA cores every 4 WGMMA instructions (i.e., every 128 K-elements) to compensate for the limited (~14-bit) accumulation precision of H800 FP8 tensor cores — [arXiv 2505.09343](https://arxiv.org/pdf/2505.09343) (14-bit figure is **PK**).
- Embedding, output head, MoE gating, normalization, attention operators remain BF16/FP32; optimizer moments in BF16; master weights and weight gradients in FP32 — [fireworks.ai summary](https://fireworks.ai/blog/deepseek-model-architecture); [arXiv 2507.09955](https://arxiv.org/pdf/2507.09955).
- The widely quoted "<0.25% relative loss error vs BF16" was measured in ablations on 16B and 230B DeepSeek-V2 models, NOT on the 671B V3 run itself — [arXiv 2505.09343](https://arxiv.org/pdf/2505.09343); [medium summary](https://medium.com/@prashantsahdev/deepseek-v3-blog-5-low-precision-training-the-fp8-revolution-in-large-scale-ai-29fc4b14761e). Critical: there is no public BF16 control for V3, so the headline claim is an extrapolation.
- Stated hardware wish-list (future work): tensor cores with higher accumulation precision and native fine-grained (tile/block) scaling support — [arXiv 2505.09343](https://arxiv.org/pdf/2505.09343).
- (PK) V3 also stores some activations in FP8 for the backward pass (with E5M6 for attention-input activations), and uses FP8 for MoE dispatch all-to-all communication — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) **[PK: verify]**.

**Scaling FP8 training to trillion-token LLMs (Fishman, Chmiel, Banner, Soudry; Intel Habana; arXiv 2409.12517, Sep 2024; ICLR 2025)**
- Trained FP8 on up to 2T tokens (~20× prior FP8 runs) and found instabilities invisible in shorter runs; Llama2-7B FP8 diverged after ~220B tokens — [arXiv 2409.12517](https://arxiv.org/abs/2409.12517); [OpenReview PDF](https://openreview.net/pdf?id=E1EHO0imOb).
- Mechanism: outlier amplification by SwiGLU, because w1 and w2 of SwiGLU, uncorrelated at init, become correlated/aligned during training; the analysis assumes ℓ2 regularization (weight decay) and sufficiently large inputs — [Graphcore research blog summary](https://graphcore-research.github.io/fp8_smooth_swiglu/); [arXiv 2409.12517](https://arxiv.org/abs/2409.12517).
- Fix: Smooth-SwiGLU (per-channel scaling that does not alter the function) plus FP8 quantization of BOTH Adam moments; 7B on 256 Gaudi2 matched BF16 with ~34% throughput gain — [arXiv 2409.12517](https://arxiv.org/abs/2409.12517).
- Theoretical claim: an analysis of weight alignment under ℓ2 regularization predicting quadratic (product) growth of the SwiGLU output outlier — [Graphcore blog](https://graphcore-research.github.io/fp8_smooth_swiglu/). Assumptions: weight decay, large-magnitude inputs; no proof that this is the only late-training outlier mechanism.
- Critical: single architecture (Llama2-7B), single hardware (Gaudi2, delayed scaling); no independent replication found in search.

**COAT (Xi et al., NVIDIA/MIT/Berkeley; arXiv 2410.19313; ICLR 2025)**
- Argues existing FP8 frameworks apply FP8 only to linear layers and leave optimizer states and activations in higher precision — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6ac807c9b296964409b277369e55621a-Paper-Conference.pdf).
- Dynamic Range Expansion: applies f(x)=sign(x)|x|^k with per-group k chosen on the fly before FP8 quantization of optimizer states, motivated by E4M3 range ~2e5 vs first-moment range ~1e3 — [GitHub NVlabs/COAT](https://github.com/NVlabs/COAT); [Lacuna summary](https://lacuna.tiptreesystems.com/work/coat-compressing-optimizer-states-and-activations-for-memory-efficient-fp8/wrk_ffcfe7d810576602171f41812f09f4ce).
- Mixed-Granularity Activation Quantization (per-tensor for linear, per-group for non-linear ops); third-party summary: ~1.54× memory reduction, ~1.43× speedup vs BF16, "nearly zero" accuracy loss — [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6ac807c9b296964409b277369e55621a-Paper-Conference.pdf). No stated-limitations text surfaced.

**MXFP8 recipe (Mishra, Stosic, Layton, Micikevicius; NVIDIA; arXiv 2506.08027, Jun 2025)**
- Central finding: the scale-factor rounding mode in the OCP MX spec (effectively round-down of the exponent) can cause divergence in LLM pretraining; rounding the E8M0 scale exponent UP (round-to-infinity) fixes it and makes MXFP8 match BF16 for an 8B model on 15T tokens — [alphaxiv](https://www.alphaxiv.org/abs/2506.08027v1); [Pith review](https://pith.science/paper/2506.08027).
- Recipe: E4M3 for weights, activations AND activation gradients (not E5M2 for grads), blocks of 32 sharing a UE8M0 power-of-two scale — [Pith review](https://pith.science/paper/2506.08027). Later FP8 recipes adopt the same round-up rule — [arXiv 2509.22536](https://arxiv.org/html/2509.22536v5).

**Other FP8 stability work (titles surfaced; details limited)**
- "To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability" (arXiv 2405.18710, May 2024) — [arXiv](https://arxiv.org/pdf/2405.18710). (PK: argues FP8 narrows the stable learning-rate range and proposes LR-sensitivity as a stability metric.)
- Nemotron 3 Ultra (NVIDIA, arXiv 2606.15007, Jun 2026): reportedly found that switching all tensors to BF16 did NOT resolve a training divergence in its (low-precision) setting — [arXiv 2606.15007](https://arxiv.org/pdf/2606.15007) (from search summary; verify).
- "A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models" (arXiv 2509.22536, Sep 2025) — [arXiv](https://arxiv.org/html/2509.22536v5).
- "Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training" (arXiv 2602.18851, Feb 2026) — title only — [arXiv](https://arxiv.org/pdf/2602.18851).
- "MoR: Mixture of Representations for Mixed-Precision Training" (arXiv 2512.22804) and "SNIP: An Adaptive Mixed Precision Framework for Subbyte LLM Training" (arXiv 2602.01410) — titles only — [MoR](https://arxiv.org/pdf/2512.22804); [SNIP](https://arxiv.org/pdf/2602.01410).
- Survey: "Low-Precision Training of LLMs: Methods, Challenges, and Opportunities" (arXiv 2505.01043, May 2025) — [arXiv](https://arxiv.org/html/2505.01043v1).

### Inferences
- The field converged on E4M3-everywhere + block/tile scaling + FP32 accumulation, abandoning E5M2 gradients and delayed per-tensor scaling; both DeepSeek-V3 and NVIDIA MXFP8 do this independently. The remaining degrees of freedom (block shape 1×128 vs 32, scale format E8M0 vs FP32, rounding direction of the scale) are determined by hardware, not by theory.
- The only mechanistic instability story (SwiGLU w1/w2 alignment under weight decay) predicts a time-of-onset that scales with tokens, which would explain why 100B-token studies miss it; but no paper has derived the onset time as a function of LR, weight decay, width or depth.
- "Scale rounding direction matters" (Mishra et al.) implies saturation/clipping is more damaging than underflow for MXFP8 training; this contradicts the folk assumption (from FP8-LM/TE) that underflow of small gradients is the primary FP8 failure.

### Gaps
- No public BF16 control exists for any ≥100B-parameter FP8 run; the "<0.25%" figure is from ≤230B-parameter DeepSeek-V2 ablations, and Fishman et al. is 7B. Whether FP8 loss gaps grow, shrink or stay flat with model size over trillions of tokens is unmeasured.
- Fishman et al.'s SwiGLU mechanism has not been reconciled with the Dissecting-Outlier-Dynamics finding (KQ2) that outliers evolve from transient early spikes to persistent "hot channels"; it is unknown whether these are the same phenomenon.
- Nemotron 3 Ultra's report that a BF16 switch did not fix divergence (if accurate) contradicts the NVFP4 paper's finding that a late BF16 switch closes the gap; no paper explains when precision switching works.
- No theory links accumulation precision (14-bit vs FP32), block size and the tolerable token count; DeepSeek's choice of promotion every 128 elements is empirical.
- COAT's and FP8-LM's stated limitations could not be retrieved (fetch unavailable).

---

## KQ2. FP4 / MXFP4 / NVFP4 pretraining: techniques, residual gaps vs BF16, scaling claims

### Takeaway
By mid-2026 the standard FP4 pretraining stack is: Random Hadamard Transform (RHT) on gradient-GEMM inputs, stochastic rounding (SR) on gradients only, 2D (16×16) weight scaling so forward and backward see the same quantized weights, a tail of BF16 "sensitive" layers, and optionally a late-phase BF16/MXFP8 switch. The longest run (NVIDIA, 12B/10T tokens) still shows a ~1–1.5% loss gap that widens during LR decay; 2026 work attributes residual gaps to (a) biased gradients (element-wise SR variance, E2M1 "shrinkage bias"), (b) persistent hot-channel outliers in post-QK and SwiGLU paths, (c) weight oscillation at bin boundaries, and (d) un-quantized optimizer/attention components. No FP4 paper trains >13B dense parameters end-to-end, and ablations are typically ≤3B / ≤100B tokens.

### Cited Findings

**Microsoft FP4 (Wang et al., arXiv 2501.17116, Jan 2025; ICML 2025)**
- Techniques: Differentiable Gradient Estimator (DGE) for weight quantization (replaces STE with a smooth surrogate derivative), Outlier Clamping and Compensation (OCC) for activations (clamp at a quantile, compensate residual via sparse matrix), vector-wise quantization, mixed precision — [ICML proceedings](https://proceedings.mlr.press/v267/wang25ae.html); [arXiv](https://arxiv.org/pdf/2501.17116).
- Scale: models 400M, 1.3B, 7B, 13B; 13B on up to 100B tokens, "accuracy close to BF16 and FP8" — [ICML](https://proceedings.mlr.press/v267/wang25ae.html); [author blog](https://mr-philo.github.io/posts/2025/05/analysis-fp4/).
- Limitation (stated): FP4 was SIMULATED on H-series FP8 tensor cores because FP4 tensor cores were unavailable; no real speedup measured — [arXiv HTML](https://arxiv.org/html/2501.17116v2). Critical: 100B tokens is ~8 tokens/param for 13B, far below Chinchilla-optimal, so the result says little about long-horizon FP4 behavior.

**Quartet (Castro, Panferov, et al.; ISTA/Red Hat; arXiv 2505.14669, May 2025; venue unverified)**
- Forward: Hadamard rotation + RMSE/MSE-optimal clipping (QuEST-style); backward: SR with randomized Hadamard transform; "fully MXFP4" linear layers with Blackwell kernels — [themoonlight review](https://www.themoonlight.io/en/review/quartet-native-fp4-training-can-be-optimal-for-large-language-models); [arXiv](https://arxiv.org/pdf/2505.14669).
- Theoretical/scaling claim: a low-precision scaling law with two efficiency parameters effN, effD ∈ (0,1] multiplying N and D in a Chinchilla-type law; fitted effN≈0.64, effD≈0.94 for Quartet vs effN=0.50, effD=0.15 for prior LUQ-INT4 — [themoonlight](https://www.themoonlight.io/en/review/quartet-native-fp4-training-can-be-optimal-for-large-language-models); [alphaxiv](https://www.alphaxiv.org/abs/2505.14669). Mechanistic reading: effN tracks forward-pass compression error, effD tracks gradient-estimator bias — [liner review](https://liner.com/review/quartet-native-fp4-training-can-be-optimal-for-large-language).
- CONTRADICTION in secondary sources on the exact functional form: one summary writes L = [A/(N·effN)^α + B/(D·effD)^β]^γ + E, another writes (A(N·effN)^α + B(D·effD)^β)^γ + E; the OpenReview snippet is garbled — [OpenReview PDF](https://openreview.net/attachment?id=XMzxZ6h68o&name=pdf). Treat the reciprocal (Chinchilla-like) form as the likely correct one; verify Eq. 1.
- Assumption: effN, effD are constants independent of N, D (fit on small models), which is the key untested assumption behind "FP4 can be optimal".
- Hardware limitation (stated): SR relies on specialized operations that have hardware support for MXFP4 but may be lacking for other formats — [themoonlight](https://www.themoonlight.io/en/review/quartet-native-fp4-training-can-be-optimal-for-large-language-models).

**Training LLMs with MXFP4 (Tseng, Yu, Park; Cornell/AWS; arXiv 2502.20586; AISTATS 2025)**
- SR gives unbiased gradient estimates but applied directly to MXFP4 has high variance from block-level outliers; RHT is used to theoretically bound the variance of SR — [arXiv](https://arxiv.org/pdf/2502.20586); [OpenReview](https://openreview.net/forum?id=a8z5Q0WSPL).
- GPT models up to 6.7B, "minimal degradation" over BF16 mixed precision; estimated >1.3× speedup over FP8 and >1.7× over BF16 during backpropagation (MXFP4 applied to backward GEMMs) — [Amazon Science](https://www.amazon.science/publications/training-llms-with-mxfp4); [GitHub](https://github.com/amazon-science/mxfp4-llm).
- Theory: variance bound of SR-after-RHT (exact theorem statement not retrieved); assumption is that the Hadamard makes coordinates approximately Gaussian/sub-Gaussian so per-block max shrinks to O(sqrt(log d)) **[PK]**.

**TetraJet (Chen et al., Tsinghua; arXiv 2502.20853, Feb 2025) and TetraJet-v2 (arXiv 2510.27527, Oct 2025)**
- TetraJet (MXFP4, Vision Transformers): identifies biased gradients and forward weight oscillation near quantization thresholds as primary degradation sources; proposes Q-EMA (round using an EMA of historical weights) and Q-Ramping; >50% reduction of accuracy degradation on ViT — [arXiv 2502.20853](https://arxiv.org/html/2502.20853v1); [GitHub](https://github.com/thu-ml/TetraJet-MXFP4Training). Critical: ViT/DeiT, not LLM.
- TetraJet-v2 (NVFP4, LLMs): weight oscillation = quantized weights flip between two bins while master weights barely move; "double-block quantization" for unbiased gradient estimation given NVFP4's scale-factor numerics; OsciReset (reset unstable master weights to bin centers); outlier control; claim: "consistently outperform prior work" — [arXiv 2510.27527](https://arxiv.org/abs/2510.27527); [GitHub README](https://github.com/thu-ml/TetraJet-v2-NVFP4Training/blob/master/README.md). Model sizes/numbers not retrieved.

**Pretraining LLMs with NVFP4 (NVIDIA, arXiv 2509.25149, Sep 2025, rev. Mar 2026)**
- 12B hybrid Mamba-Transformer on 10T tokens; "longest publicly documented 4-bit run"; techniques: Random Hadamard transforms, two-dimensional (16×16) weight quantization so forward and backward share quantized weights, stochastic rounding on gradients, selective high-precision (BF16) layers (~last 15% of linear layers, PK) — [arXiv](https://arxiv.org/html/2509.25149v1); [emergentmind](https://www.emergentmind.com/papers/2509.25149).
- Loss gap vs FP8 baseline: <1% during the stable LR phase, widening to slightly above 1.5% during LR decay — [arXiv](https://arxiv.org/html/2509.25149v1).
- Switching to BF16 (or possibly MXFP8) during the decay phase mitigates the gap; most of the benefit comes from switching forward-pass tensors — [liner review](https://liner.com/review/pretraining-large-language-models-with-nvfp4); switch does not eliminate the gap — [emergentmind](https://www.emergentmind.com/topics/nvfp4-pre-training).
- (PK) Downstream: MMLU-Pro 5-shot ≈62.6% for both NVFP4 and FP8 **[PK: verify]**.
- Critical: baseline is FP8, not BF16; a single run; ablations of individual techniques are on smaller models; hybrid Mamba architecture means transformer-only generalization is unproven.

**Quartet II (Panferov et al., arXiv 2601.22813, Jan 2026, rev. Jun 2026)**
- MS-EDEN: an unbiased quantizer that moves randomness from individual FP4 values onto group scale factors (SR on micro-scales), after a Randomized Hadamard rotation, retaining provable unbiasedness in expectation; >2× lower quantization error than element-wise SR (synthetic MSE 9.8 vs 17.5 ×10⁻³ on N(0,1)) — [arXiv](https://arxiv.org/html/2601.22813v1); [emergentmind MS-EDEN](https://www.emergentmind.com/topics/ms-eden-quantization-operator).
- Forward: RTN with native NVFP4 scaling + Four-over-Six adaptive block scaling heuristic (arXiv 2512.02010, Dec 2025); fully-NVFP4 linear-layer graph — [liner review](https://liner.com/review/quartet-ii-accurate-llm-pretraining-in-nvfp4-by-improved-unbiased); [Four Over Six](https://arxiv.org/pdf/2512.02010).
- Loss-gap result is Fig. 4 (C4 val-loss gap vs BF16 across tokens-per-parameter) for Llama-2-like models; a third-party summary says gaps shrink ≥20%; up to 4.2× speedup claim (third-party) — [Lacuna](https://lacuna.tiptreesystems.com/work/quartet-ii-accurate-llm-pre-training-in-nvfp4-by-improved-unbiased-gradient/wrk_11746737aca4e06dfcba1a290998e2be). Exact numbers unverified.

**Dissecting Outlier Dynamics in LLM NVFP4 Pretraining (Dong et al., arXiv 2602.02047, Feb 2026)**
- Longitudinal study: outliers originate from Softmax in attention, gating in linear attention, and SwiGLU in the FFN; "post-QK" operations most quantization-sensitive; outliers shift from transient early spikes to a small set of persistent "hot channels" late in training — [arXiv](https://arxiv.org/abs/2602.02047).
- Linear attention reduces per-tensor heavy tails vs softmax attention but still shows persistent block-level spikes under block quantization — [arXiv](https://arxiv.org/abs/2602.02047).
- Fix: Hot-Channel Patch (online residual re-injection for hot channels) + post-QK protection = CHON recipe; on GLA-1.3B / 60B tokens reduces loss gap to BF16 from 0.94% to 0.58% — [arXiv](https://arxiv.org/abs/2602.02047). Critical: 1.3B, 60B tokens, linear-attention architecture.

**Rethinking Shrinkage Bias in LLM FP4 Pretraining / UFP4 (Zhao et al., Ant Group Ling Team, arXiv 2606.20381, Jun 2026)**
- Theoretical claim: non-uniform formats such as E2M1 have a negative expected rounding error under round-to-nearest-even ("Shrinkage Bias") caused by geometric asymmetry of representable bins; the bias accumulates multiplicatively across layers and is AMPLIFIED by the Random Hadamard Transform, offered as a unified explanation of instability in E2M1-based recipes — [arXiv](https://arxiv.org/html/2606.20381v1); [Pith](https://pith.science/paper/2606.20381).
- UFP4 recipe: uniform-grid E1M2 (or INT4) elements instead of E2M1; RHT on every training GEMM but SR restricted to the dY term; results on Dense 1.5B, MoE 7.9B and MoE 124B long-run pretraining, with E1M2 "significantly outperforming" E2M1 baselines in BF16-relative loss degradation — [Pith](https://pith.science/paper/2606.20381).
- Critique (automated Pith review): experiments compare full recipes, so the bin-uniformity effect is not isolated from changes to the rounding schedule and RHT; no independent replication — [Pith](https://pith.science/paper/2606.20381).
- CONTRADICTION: this paper says RHT amplifies a bias; Tseng et al./Quartet/NVIDIA use RHT precisely to reduce variance and claim unbiasedness with SR. Reconciliation hinges on RTN vs SR: shrinkage bias is an RTN phenomenon, so recipes using RTN forward + RHT are the ones exposed.

**Pretraining LLMs with MXFP4 on Native FP4 Hardware (arXiv 2605.09825, May 2026)**
- On Llama-3.1-8B, stochastic rounding and randomized Hadamard rotations reportedly FAIL to stabilize training once the weight-gradient (Wgrad) GEMM is quantized, whereas deterministic Hadamard rotations consistently restore stable optimization — [arXiv](https://arxiv.org/html/2605.09825v3) (from search summary; verify).
- CONTRADICTION with Tseng et al. (2502.20586), Quartet, and NVIDIA NVFP4, all of which rely on SR+RHT for the gradient GEMMs. Possible explanations: MXFP4 (E8M0 scales, 32-blocks) vs NVFP4 (E4M3 scales, 16-blocks); 8B scale; which GEMMs are quantized.

**Full-Stack FP4 (Ding et al., arXiv 2607.04422, Jul 2026)**
- Claims first NVFP4 framework covering linear layers, optimizer states/arithmetic and attention; each module fails numerically in its own way — [arXiv](https://arxiv.org/abs/2607.04422).
- AdamW second moments are non-negative, heavy-tailed and denominator-sensitive; they are "reshaped" to be storable in NVFP4; Muon's Newton–Schulz iterations run natively in NVFP4; attention Q/K/V and backward dS quantized with sensitive paths in BF16; a LoRA-SVD decomposition cuts the linear-only loss gap from 1.40% to 0.61% — [arXiv HTML](https://arxiv.org/html/2607.04422v1).
- Overall: 1.47% loss gap vs BF16 at 3B params / 64B tokens with all three blocks in NVFP4 — [arXiv](https://arxiv.org/abs/2607.04422). Notes that most end-to-end FP4 stacks leave optimizer states in FP32 and that 4-bit Muon is feasible but needs mixed-precision stabilization — [arXiv HTML](https://arxiv.org/html/2607.04422v1).
- Critical: 1.47% at 3B/64B tokens is WORSE than NVIDIA's 1–1.5% at 12B/10T despite more quantization; the full-stack cost is real.

**UE5M3 FP4 Block Scaling for Stable LM Pretraining (arXiv 2609.02846, Sep 2026)**
- Proposes a UE5M3 scale format (5 exponent, 3 mantissa bits, unsigned) for FP4 block scaling as a middle ground between NVFP4's E4M3 scale (precise, narrow range) and MX's UE8M0 (wide range, power-of-two only); reports loss spikes with other formats — [arXiv](https://arxiv.org/pdf/2609.02846). Details not retrieved.

**Other 2025–2026 FP4 titles (details not retrievable this session)**
- "FP4 All the Way: Fully Quantized Training of LLMs" (Chmiel et al., Intel Habana; OpenReview kuzye4EPLR; arXiv 2505.19115 **[ID PK, unverified]**) — [OpenReview](https://openreview.net/pdf?id=kuzye4EPLR). (PK: proposes SR for backward/RTN forward and a criterion relating gradient norm to quantization noise for when FP4 lags BF16.)
- "Elucidating the Design Space of FP4 Training" (arXiv 2509.17791, Sep 2025) — [arXiv](https://arxiv.org/html/2509.17791v1).
- "The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training" (arXiv 2603.10444, Mar 2026) — [awesomepapers](https://awesomepapers.io/llm-papers/papers/2603.10444).
- "HiFloat4 Format for LM Pre-training on Ascend NPUs" (Huawei, arXiv 2604.08826, Apr 2026) — [arXiv](https://arxiv.org/html/2604.08826v1).
- "Format-Aware Fusion for Fast FP4 Pretraining" (arXiv 2610.00053, Oct 2026) — [arXiv](https://arxiv.org/html/2610.00053).
- "Puro-2B: ... Trained on RTX 5090" (arXiv 2608.27370) — [arXiv](https://arxiv.org/pdf/2608.27370).
- "Scaling Laws for Floating Point Quantization Training" (Sun et al., Tencent Hunyuan, arXiv 2501.02423, Jan 2025): unified law in exponent bits, mantissa bits and block size; (PK) predicts a data cap beyond which more tokens hurt at fixed precision and that cost-optimal FP precision lies between 4 and 8 bits — [arXiv](https://arxiv.org/pdf/2501.02423) **[PK on content]**.

### Inferences
- The residual FP4 gap decomposes into additive sources that different 2026 groups each claim to be "the" bottleneck: gradient bias/variance (Quartet II, UFP4), hot-channel outliers (Dong et al.), weight oscillation (TetraJet-v2), un-quantized components (Full-Stack). No paper combines all fixes, so the achievable floor is unknown.
- The gap widening during LR decay (NVIDIA) is consistent with the "effective step size vs quantization noise" picture: as LR shrinks, SR noise and bin-flip oscillation stop averaging out and dominate updates; this is also why late BF16 switching helps and why weight-oscillation fixes matter most late. This is an inference, not a published derivation.
- The SR+RHT success (NVFP4, 16-blocks, E4M3 scale) vs failure (MXFP4, 32-blocks, E8M0 scale, 8B) suggests the scale format, not the element format, determines whether SR variance is controllable; UE5M3 and Four-over-Six are both attempts to fix the scale, supporting this reading.

### Gaps
- No FP4 pretraining result exists for a dense model >13B or for >10T tokens; no FP4 run has a BF16 (rather than FP8) control at ≥12B.
- Direct contradiction (SR+RHT works vs fails for Wgrad) between 2605.09825 and 2502.20586/2505.14669/2509.25149 is unresolved.
- Direct contradiction on RHT (variance reducer vs bias amplifier) between UFP4 and the rest; no paper measures both bias and variance of the same gradient estimator at scale.
- Scaling of effN/effD with N and D is assumed constant; Quartet II's Fig. 4 across tokens-per-parameter is the only test and its numbers were not retrievable.
- Whether the late-decay gap is caused by oscillation, by SR noise, or by outlier growth is untested because no paper ablates them at the same LR schedule point.
- Full-stack FP4 costs (1.47% at 3B) have not been scaled; optimizer-state FP4 (AdamW second moments) is tested at a single scale.

---

## KQ3. Scaling laws for precision (training and QAT): functional forms, predictions, limitations

### Takeaway
Three families of law exist: (1) Kumar et al.'s "effective parameter count" law, N_eff = N(1 − e^{−P/γ}), with a PTQ-degradation term growing in D/N; (2) Ouyang et al.'s PTQ-degradation law ∆qLoss ∝ D^0.53 / (N^0.23 P^5.5) fitted on Pythia+GPTQ; (3) QAT/low-bit-training laws (Chen et al. 2505.14302; Quartet effN/effD; Spectra 1.1; Sun et al. FP law) that all agree quantization error falls with N and RISES with D. The headline predictions — compute-optimal training precision ~7–8 bits (integer type) and "more tokens make low-bit worse" — rest on ≤1.7B-parameter fits, integer (not FP block-scaled) quantization, and an independence assumption for noise; no 2026 law yet incorporates the two-phase (FP then QAT) regime or the block-scale format.

### Cited Findings

**Scaling Laws for Precision (Kumar, Ankner, et al.; arXiv 2411.04330, Nov 2024; ICLR 2025)**
- Core: training in lower precision reduces the model's "effective parameter count"; N_eff is a saturating exponential in precision whose returns plateau around 6–8 bits — [alphaxiv overview](https://www.alphaxiv.org/overview/2411.04330); [arXiv](https://arxiv.org/html/2411.04330). (PK form: N_eff(P) = N(1 − e^{−P/γ}), with separate γ for weights, activations and KV cache; L = A/N_eff^α + B/D^β + E.)
- PTQ degradation model: δ_PTQ = C_T · (D^{γ_D}/N^{γ_N}) · e^{−P_post/γ_post} with fitted constants; positive exponent on D means overtrained models are MORE sensitive to post-training quantization, and eventually additional pretraining data can be actively harmful for quantized inference — [liner review](https://liner.com/review/scaling-laws-for-precision); [ICLR 2025 PDF](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf).
- Compute-optimal precision "around 7–8 bits" when fitting on integer-type quantization; "the race to low-precision training may have to stop before going below 4-bits" because lower precision forces much larger models — [arXiv PDF](https://arxiv.org/pdf/2411.04330). Also: when model size is fixed, training larger models in lower precision can be compute-optimal — [alphaxiv](https://www.alphaxiv.org/overview/2411.04330).
- Evidence base: 465 pretraining runs, models ≤1.7B params, ≤26B tokens — [emergentmind](https://www.emergentmind.com/papers/2411.04330).
- Stated/inferred limitations: integer-type fits are strong only down to 4 bits and diverge from FP-type below that, so cross-type laws are "suggested, not established"; the law treats low-precision effects as independent multiplicative noise; frontier-scale extrapolation untested; no hardware cost model beyond cost ∝ bits — [ICLR PDF](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf); [paper notes](https://github.com/AkihikoWatanabe/paper_notes/issues/1512).

**Low-Bit Quantization Favors Undertrained LLMs (Ouyang et al., arXiv 2411.17691, Nov 2024; ACL 2025)**
- >1500 quantized checkpoints; QiD (quantization-induced degradation) grows with training tokens D, shrinks with N and with bits P; used to predict quantization behavior of models trained on 100T tokens, concluding low-bit quantization of such models "may NOT be desirable" — [arXiv](https://arxiv.org/abs/2411.17691); [ACL Anthology](https://aclanthology.org/2025.acl-long.1555.pdf).
- Fitted form (Pythia + GPTQ): ∆qLoss(N,D,P) = 0.017 · D^0.5251 / (N^0.2261 · P^5.4967); a review quotes the token exponent as ≈0.5316 (version difference) — [ResearchGate](https://www.researchgate.net/publication/386529634_Low-Bit_Quantization_Favors_Undertrained_LLMsScaling_Laws_for_Quantized_LLMs_with_100T_Training_Tokens); [liner](https://liner.com/review/lowbit-quantization-favors-undertrained-llms-scaling-laws-for-quantized-llms). Total: Loss_q = Loss_16bit + ∆qLoss — [themoonlight](https://www.themoonlight.io/de/review/low-bit-quantization-favors-undertrained-llms-scaling-laws-for-quantized-llms-with-100t-training-tokens).
- Limitations: PTQ (GPTQ) on Pythia only; 100T-token prediction is an extrapolation by >100× in D; no independent critique found — [arXiv](https://arxiv.org/abs/2411.17691). Scope note: this is a PTQ law, relevant here only because it shares the D-exponent prediction with the training laws.

**Scaling Law for Quantization-Aware Training (Chen et al., arXiv 2505.14302, May 2025; ICML 2026 poster)**
- 268 QAT experiments; quantization error decreases with model size, increases with training tokens, increases with coarser quantization granularity (logarithmic in group size G); law = Chinchilla loss + additive quantization-error term — [arXiv](https://arxiv.org/abs/2505.14302); [alphaxiv](https://www.alphaxiv.org/abs/2505.14302); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62561).
- Weight quantization error grows faster with tokens; activation quantization error in the FC2 (down-projection) layer caused by outliers is the primary W4A4 bottleneck; mixed precision on FC2 makes weight and activation errors converge to similar levels — [HF paper page](https://huggingface.co/papers/2505.14302).
- Limitation identified by follow-up: the law lacks a two-phase (full-precision then QAT) training process — "Compute-Optimal Quantization-Aware Training" (arXiv 2509.22935, Sep 2025) — [alphaxiv](https://www.alphaxiv.org/abs/2509.22935). The follow-up builds on ParetoQ's quantizers (Elastic Binarization 1-bit, SEQ 2-bit, LSQ ≥3-bit) — [arXiv](https://arxiv.org/pdf/2509.22935); its optimal QAT-fraction result was not retrieved.

**Quartet effN/effD law** — see KQ2; effN≈0.64, effD≈0.94 for MXFP4 Quartet; constants assumed scale-independent — [themoonlight](https://www.themoonlight.io/en/review/quartet-native-fp4-training-can-be-optimal-for-large-language-models).

**Spectra 1.1 (arXiv 2506.23025, Jun 2025)**
- Ternary (TriLM) scaling law to 1T tokens: more tokens reduce validation loss more than more parameters; stated limitation: the law does not account for the number of bits b — [arXiv](https://arxiv.org/html/2506.23025).

**Scaling Laws for Floating Point Quantization Training (Sun et al., Tencent, arXiv 2501.02423, Jan 2025)** — [arXiv](https://arxiv.org/pdf/2501.02423). Content PK: law in (N, D, E, M, B) with a D^β/N^α · f(E,M,B) interaction term; predicts an optimal exponent/mantissa split, a "critical data size" beyond which more data hurts at fixed precision, and 4–8 bit as cost-optimal **[PK: verify]**.

**Depth Laws for the Precision Floor (Tarawneh, arXiv 2609.32060, Sep 2026)**
- Single-author preprint: noise-aware training roughly doubles tolerable noise in shallow networks but the gain decays with depth; frames a "QAT paradox"; residual scaling as mitigation — [arXiv](https://arxiv.org/abs/2609.32060). Unverified beyond abstract.

**How Numerical Precision Affects Arithmetical Reasoning (arXiv 2410.13857)** — theory that low precision limits arithmetic circuits in transformers — [arXiv](https://arxiv.org/pdf/2410.13857). Title/abstract only.

### Inferences
- All families agree on the sign of the D-effect (more tokens → more quantization damage at fixed N, P). Kumar's and Ouyang's exponents on D (~0.5) imply degradation grows like sqrt(D), i.e., the per-weight information content grows and clips against the bit budget. None derives this from a mechanism; it is phenomenological.
- Kumar's law and Quartet's law are structurally different (N_eff = N·(1−e^{−P/γ}) vs N·effN with effN a per-recipe constant) and would make different predictions about the P→∞ limit and about mixed-precision schedules; they have not been fitted on the same data.
- Spectra 1.1's statement that tokens beat parameters for ternary models is in tension with Kumar/Ouyang's "more tokens hurt low-bit"; the resolution is likely that Spectra trains natively ternary (no PTQ cliff), while Kumar/Ouyang's D-penalty is a PTQ-sensitivity effect — but Chen et al.'s QAT law also finds D hurts, so native training is not immune.

### Gaps
- No law covers block-scaled FP formats (NVFP4/MXFP4) with their scale-format dependence; Kumar fits integer type and warns FP diverges below 4 bits.
- No law covers two-phase training (ParetoQ's ~90% FP + QAT) except 2509.22935, whose results were not retrievable.
- No law has been validated beyond ~1.7B parameters / ~26B tokens (Kumar) or ~1T tokens (Spectra 1.1); the 7–8 bit optimum and the 100T-token prediction are unverified extrapolations.
- The exact functional forms (Quartet Eq. 1, Chen et al. error term, Sun et al. law) could not be confirmed against primary text in this session.

---

## KQ4. Fully quantized / native low-bit training (BitNet, ternary, ParetoQ): bits-vs-parameters and true training cost

### Takeaway
Ternary-weight LLMs (BitNet b1.58, Spectra TriLM) match FP16 models of equal parameter count on benchmarks at 2–4B parameters and 100B–4T tokens, but only on downstream accuracy; perplexity on web corpora lags, and all of them train with full-precision latent weights, high-precision gradients and optimizer states plus STE, so training FLOPs/memory are not reduced (some reports say slower). ParetoQ (Meta) contradicts the from-scratch school: spending ~90% of the budget in full precision and then QAT beats native low-bit training, and 2–3 bits (not 1.58 or 4) sit on the accuracy-vs-size Pareto frontier. 2026 ternary work is mostly storage/kernels, not training.

### Cited Findings

**BitNet b1.58 (Ma et al., arXiv 2402.17764, Feb 2024) [ID PK] and JMLR version**
- Ternary weights {−1,0,1}; claim: matches the half-precision Transformer LLM with the same model size and training tokens in perplexity and end-task performance (from 3B); the gap between FP16 and BitNet shrinks as model size grows — [JMLR vol. 26 "BitNet: 1-bit Pre-training for LLMs"](http://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf); [emergentmind](https://www.emergentmind.com/topics/bitnet-style-1-58-bit-transformer).
- (PK) Training uses absmean weight quantization and 8-bit per-token absmax activation quantization in the forward pass, with FP16/BF16 latent weights and STE; optimizer states and gradients are high precision; no claim of training-FLOP reduction — [arXiv 2402.17764](https://arxiv.org/abs/2402.17764) **[PK]**.

**BitNet b1.58 2B4T (Microsoft, arXiv 2504.12285, Apr 2025) [ID PK]**
- 2B-parameter fully ternary model trained on 4T tokens; reported comparable to full-precision open models of similar size — [GitHub microsoft/BitNet](https://github.com/microsoft/BitNet); [medium summary](https://medium.com/data-science-in-your-pocket/bitnet-b1-58-2b4t-pushing-the-boundaries-of-efficient-on-device-llms-fe4c084bd4c0).

**BitNet a4.8 (Wang et al., arXiv 2411.04965, Nov 2024)**
- 4-bit activations for attention and FFN inputs, sparsification of intermediate states followed by 8-bit quantization; performance comparable to BitNet b1.58 "with equivalent training costs", faster inference — [arXiv](https://arxiv.org/pdf/2411.04965); [MSR page](https://www.microsoft.com/en-us/research/publication/bitnet-a4-8-4-bit-activations-for-1-bit-llms/). (PK: ~55% active parameters, 3-bit KV cache.)

**BitNet v2 (arXiv 2504.18415, Apr 2025)**
- H-BitLinear applies a Hadamard transform before activation quantization in output projections (Wo, down-proj); trained W1.58A8 from scratch, then continue-trained with INT4 activations for all linear layers except embeddings; 4-bit variant comparable to BitNet a4.8 with better batched-inference efficiency — [arXiv](https://arxiv.org/html/2504.18415). Note: native-4-bit-activation is achieved by a two-stage schedule, not from scratch.

**Training cost of BitNet-style training**
- A binary-NN survey reports BitNet training is ~60% slower than baseline and relatively more costly, attributing it to RMSNorm (and, PK, the extra quantization/dequantization ops) — [arXiv 2502.19008](https://arxiv.org/pdf/2502.19008). Latent-weight precision question not confirmed from search.

**Spectra (Kaushal et al., arXiv 2407.12327, Jul 2024) and Spectra 1.1 (arXiv 2506.23025, Jun 2025)**
- Suite of FloatLMs, QuantLMs (PTQ) and TriLMs (ternary from scratch), 99M–3.9B params, 300B tokens; TriLM 3.9B matches FloatLM 3.9B across benchmarks with fewer bits than FloatLM 830M; surpasses FloatLM on LAMBADA — [arXiv](https://arxiv.org/abs/2407.12327); [Nolano blog](https://blog.nolano.ai/Spectra-suite/).
- Stated limitations: TriLM 3.9B lags FloatLM in perplexity on validation/web corpora (better on cleaner Lambada/PTB); TriLM 3.9B is as toxic/stereotyping as FloatLM 3.9B — [arXiv](https://arxiv.org/pdf/2407.12327).
- Spectra 1.1: scaling to 1T tokens; tokens more valuable than parameters for TriLMs; law omits bit-width — [arXiv](https://arxiv.org/html/2506.23025).

**ParetoQ (Liu et al., Meta, arXiv 2502.02631, Feb 2025)**
- Unified comparison of 1, 1.58, 2, 3, 4 bits under one protocol; finds a learning transition between 2 and 3 bits (≥3 bits stay near pretrained distribution; ≤2 bits change representations drastically); ternary, 2-bit and 3-bit comparable and generally better than 4-bit and binary on the accuracy-vs-effective-size frontier; binary "significantly compromises accuracy" — [arXiv](https://arxiv.org/abs/2502.02631); [PyTorch blog](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/).
- Ternary 600M ParetoQ model outperforms the prior SoTA ternary 3B model with 1/5 the parameters — [arXiv](https://arxiv.org/pdf/2502.02631).
- Argues initializing from a pretrained full-precision model is more effective than from-scratch low-bit training, including for binary and ternary — [alphaxiv](https://www.alphaxiv.org/overview/2502.02631v2); a 2026 paper describes ParetoQ as finding that ~90% of the budget in full precision before switching to QAT beats low-bit training from scratch — [arXiv 2609.29397](https://arxiv.org/pdf/2609.29397).
- Fine-tuning token requirement: ≤2-bit needs ~30B tokens to saturate vs ~10B for 3–4 bit — [PyTorch blog](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/).
- CONTRADICTION: ParetoQ (FP-then-QAT) vs BitNet/Spectra (from scratch). Both are evaluated on downstream accuracy at ≤3B; neither isolates the effect of total token budget.

**2026 ternary items**
- "Breaking the 1.58-bit Barrier for Ternary LLMs" (Georganas, Heinecke, Dubey; Intel; arXiv 2609.16338, Sep 2026): a storage format (BITCOS: presence bitmap + compacted signs), not a training method; across 29 ternary models zeros are up to 51.5% of weights; 1.485 bits/weight on the sparsest; 1.11–2.30× kernel speedups on Intel Arc — [arXiv](https://arxiv.org/abs/2609.16338). Relevance to pretraining: the ~50% zero fraction is an emergent property of ternary training that no training paper models or controls.
- "Baseline Shape Decides the Verdict: A Controlled Re-Examination of Ternary LMs at 60K Parameters" (arXiv 2609.29397, Sep 2026) — toy-scale controlled study — [arXiv](https://arxiv.org/pdf/2609.29397).
- "TernaryLM: ... Native 1.5-bit Quantization with Adaptive Layer-wise Scaling" (arXiv 2602.07374) and "Ternary Mamba: Grouped QAT of W1.58A16 SSMs" (arXiv 2606.18114) — titles only — [TernaryLM](https://arxiv.org/pdf/2602.07374); [Ternary Mamba](https://arxiv.org/pdf/2606.18114).
- PT-BitNet (post-training route to 1-bit) — out of scope but shows the from-scratch vs post-hoc debate continues — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X).

### Inferences
- "Native low-bit" is a misnomer for training cost: every ternary/BitNet recipe is QAT with full-precision shadow weights; the reduction is in inference memory/compute only. No paper reports ternary training that is cheaper in FLOPs than BF16 training.
- The BitNet claim "gap shrinks with model size" and ParetoQ's "600M ternary beats 3B ternary" together imply that the ternary frontier is moving fast enough that comparisons across papers (different token counts, data) are unreliable; the only controlled suite (Spectra) stops at 3.9B/300B.
- The emergent ~50% zero density (Intel 2026) suggests ternary training implicitly performs 2:4-like sparsity; this is unexploited in training-time compute.

### Gaps
- No ternary/binary model >4B parameters trained from scratch with a public BF16 control; no ternary run >4T tokens.
- No study measures training FLOPs/memory of BitNet-style training vs BF16 at equal quality (the "60% slower" figure is from a survey, not a controlled benchmark).
- ParetoQ vs BitNet/Spectra disagreement on from-scratch vs FP-then-QAT is untested at equal total compute and equal data.
- No training paper models the ternary zero fraction or whether it should be targeted.

---

## KQ5. Gradient, optimizer-state and communication quantization: what theory exists and what is missing at LLM scale

### Takeaway
Optimizer-state quantization has moved from 8-bit (Dettmers) to 4-bit (Li et al. 2023), FP8 (FP8-LM, COAT, Fishman et al.), NVFP4 second moments (Full-Stack FP4, 2026) and 4-bit Muon (MuonQ, 2026); the 2026 Topollai–Choromanska paper gives the first dynamical model of why quantized EMA states "freeze" (staleness) and why periodic resets fix it. Gradient-communication compression has a mature convex/nonconvex theory (QSGD unbiasedness/variance bounds; error-feedback and EF21 convergence for contractive compressors) but LLM pretraining practice uses FP8 all-reduce/dispatch with ad-hoc scaling and no error feedback, and the theory assumes smoothness/bounded variance that adaptive optimizers and block-scaled FP formats violate.

### Cited Findings

**Optimizer states**
- 8-bit Adam (Dettmers et al., arXiv 2110.02861, ICLR 2022) — block-wise dynamic quantization + stable embedding layer; matches 32-bit optimizers across language modeling benchmarks **[PK: not re-verified]** — [arXiv](https://arxiv.org/abs/2110.02861).
- 4-bit optimizer states (Li, Chen, Zhu; NeurIPS 2023; arXiv 2309.01507 [ID PK]): adaptive block sizing for first-moment outliers and a linear quantizer that excludes the zero point for second moments — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/3122aaa22b2fe83f9cead1a696f65ceb-Paper-Conference.pdf). (PK: evaluated mostly on fine-tuning and small-scale pretraining, not LLM-scale pretraining.)
- 4-bit Shampoo (arXiv 2405.18144): quantizes preconditioner eigenvectors; comparable to 32-bit; evaluated on image classification, not LLMs — [arXiv 2412.10663 (4-bit preconditioned stochastic optimization)](https://arxiv.org/pdf/2412.10663).
- MicroAdam (arXiv 2405.15593 [ID PK]): compresses gradients before they enter optimizer states, with compressed error feedback, preserving convergence guarantees — [search result context, arXiv 2505.01043 survey](https://arxiv.org/html/2505.01043v1). Fine-tuning scale.
- FP8 optimizer states: FP8-LM (FP8 first moment, FP16 second) — [arXiv 2310.18313](https://arxiv.org/pdf/2310.18313); Fishman et al. (both moments FP8) — [arXiv 2409.12517](https://arxiv.org/abs/2409.12517); COAT (DRE power-function expansion) — [COAT GitHub](https://github.com/NVlabs/COAT); DeepSeek-V3 (BF16 moments) — [arXiv 2507.09955](https://arxiv.org/pdf/2507.09955).
- NVFP4 optimizer states: Full-Stack FP4 stores reshaped AdamW second moments in NVFP4 and runs Muon Newton–Schulz in NVFP4 (3B/64B tokens) — [arXiv 2607.04422](https://arxiv.org/html/2607.04422v1).
- MuonQ (arXiv 2605.11396, May 2026): 4-bit Muon states; recovers most of full-precision Muon's loss/accuracy on GPT- and LLaMA-style pretraining; up to 7.3× optimizer-state memory reduction; rationale: Muon's polar decomposition keeps only directional information, so magnitude errors are discarded — [arXiv](https://arxiv.org/html/2605.11396). "Effective Quantization of Muon Optimizer States" (2025): 8-bit Muon nearly matches FP Muon, −86% state memory vs AdamW — [ResearchGate](https://www.researchgate.net/publication/395969646_Effective_Quantization_of_Muon_Optimizer_States).
- Topollai & Choromanska, "Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets" (arXiv 2603.16731, Mar 2026; EMNLP 2026 main): low-precision EMA states freeze because small updates round away; a simple model predicts when and how often this happens and why well-timed resets restore the state — [awesomepapers](https://awesomepapers.io/llm-papers/papers/2603.16731); [author page](https://kristitopollai.github.io/). Theory details/assumptions not retrieved. A 2026 paper lists quantized-state staleness as a distinct failure mode — [arXiv 2607.09800](https://arxiv.org/html/2607.09800v3).
- Earlier: "Exploring Quantization for Efficient Pre-Training of Transformer LMs" (EMNLP Findings 2024) systematically quantizes weights/activations/gradients/optimizer states in small-scale pretraining — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.787.pdf).

**Gradient communication**
- QSGD (Alistarh et al., NeurIPS 2017, arXiv 1610.02132) **[PK]**: unbiased stochastic quantization with second-moment bound E||Q(v)||² ≤ (1 + min(n/s², √n/s))||v||²; convergence inherited from SGD under standard smooth (convex/nonconvex) assumptions — [arXiv](https://arxiv.org/abs/1610.02132).
- Error feedback: Karimireddy et al., "Error Feedback Fixes SignSGD" (ICML 2019, arXiv 1901.09847) **[PK]** prove nonconvex convergence for any δ-contractive (biased) compressor with EF; EF21 (Richtárik, Sokolov, Fatkhullin; NeurIPS 2021, arXiv 2106.05203) **[PK]** gives O(1/T) nonconvex rates without bounded-gradient assumptions — [1901.09847](https://arxiv.org/abs/1901.09847); [2106.05203](https://arxiv.org/abs/2106.05203). Both assume L-smoothness and (for stochastic versions) bounded variance; neither covers Adam-type preconditioning or block-scaled FP quantizers.
- PowerSGD (Vogels et al., NeurIPS 2019, arXiv 1905.13727) **[PK]**: low-rank compression with error feedback, all-reduce compatible — [arXiv](https://arxiv.org/abs/1905.13727).
- 1-bit Adam (Tang et al., ICML 2021, arXiv 2102.02888) **[PK]**: observes Adam's second-moment stabilizes after warmup, freezes it, then 1-bit compresses momentum with EF; convergence argument reduces to momentum SGD with EF — [arXiv](https://arxiv.org/abs/2102.02888).
- FP8 communication in LLM pretraining: FP8-LM adds FP8 gradient all-reduce with automatic scaling — [arXiv 2310.18313](https://arxiv.org/pdf/2310.18313); DeepSeek-V3 uses FP8 for MoE dispatch (PK) — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) **[PK]**.
- Low-communication pretraining: "Overcoming the Communication-Performance Tradeoff in LLM Pretraining" (arXiv 2508.15706, Aug 2025) finds 2-bit quantization of (DiLoCo-style) pseudo-gradients does not degrade when error feedback is used, and top-k with EF improves convergence on LLaMA-2 experiments — [arXiv](https://arxiv.org/pdf/2508.15706); SparseLoCo: top-k + EF on DiLoCo outer gradients, EF acts as a local approximation of outer momentum — [OPT-ML 2025](https://opt-ml.org/papers/2025/paper118.pdf).
- AGoQ: "Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs" (arXiv 2605.00539, May 2026) — title only — [arXiv](https://arxiv.org/pdf/2605.00539).

### Inferences
- Every production FP8 recipe quantizes optimizer states by hand-tuned range tricks (DRE, precision decoupling); Topollai–Choromanska's staleness model is the first principled account and suggests that SR or periodic resets, not more mantissa bits, are the right fix — connecting to the SR-unbiasedness theory in KQ6.
- Communication compression theory (QSGD/EF/EF21) is built for data-parallel gradient all-reduce; LLM pretraining today is dominated by FSDP reduce-scatter/all-gather of weights and MoE all-to-all of activations, which these theorems do not model. DiLoCo-style outer-gradient compression is where EF theory is actually being reused.
- MuonQ's argument (orthogonalization discards magnitude error) is a format-independent reason why Muon-family optimizers may tolerate lower state precision than Adam; Full-Stack FP4's NVFP4 Newton–Schulz is consistent.

### Gaps
- No convergence theorem exists for Adam/AdamW with quantized (stale) second moments; the staleness model's assumptions were not retrievable.
- No theory for FP8 block-scaled all-reduce (non-unbiased, saturating, scale-dependent) — FP8-LM's "auto scaling" is heuristic.
- No LLM-scale (≥7B, ≥1T tokens) study of gradient-communication quantization with error feedback in standard FSDP/TP training; evidence is DiLoCo-style or fine-tuning scale.
- 4-bit Adam states (Li et al.) have not been validated in >1B-parameter pretraining in any source found.

---

## KQ6. Theory of training under quantization noise: STE, stochastic rounding, regularization, flatness

### Takeaway
Rigorous results exist only for toy settings: STE descent/convergence (two-layer binarized ReLU nets, Gaussian data, population loss — Yin et al. 2019), finite-sample STE (O(n²)/O(n⁴) samples, Gaussian only — 2505.18113), high-dimensional ODE limits (2510.10693), and 2026 algorithmic-stability generalization bounds (2609.06430), all for two-layer networks. Stochastic rounding is unbiased by construction and its variance can be bounded after a Hadamard rotation (Tseng et al.; Quartet II), but RTN-based formats carry a provable negative "shrinkage" bias (UFP4, 2026). No theorem covers transformer-scale, adaptive optimizers, or block-scaled floats; the "quantization noise as implicit regularization / flat minima" story has no sourced pretraining-scale evidence in this session.

### Cited Findings

**STE theory**
- Yin et al., "Understanding STE in Training Activation Quantized Neural Nets" (ICLR 2019, arXiv 1903.05662): for a two-layer network with binarized ReLU activations and Gaussian inputs, a properly chosen STE (ReLU/clipped-ReLU surrogate) yields a "coarse gradient" positively correlated with the population gradient, so coarse-gradient descent converges to a critical point of the population loss; the identity STE can be unstable near some local minima — [alphaxiv](https://www.alphaxiv.org/abs/1903.05662); [emergentmind STE](https://www.emergentmind.com/topics/straight-through-estimator-ste). Assumptions: population loss (infinite data), Gaussian inputs, two layers, activation (not weight) quantization.
- "Beyond Discreteness: Sample Complexity Analysis of STE for 1-bit Quantization" (arXiv 2505.18113, May 2025): two-layer binarized net, Gaussian inputs; O(n²) samples suffice for ergodic-average convergence and O(n⁴) for last-iterate; guarantees fail for general sub-Gaussian inputs — [arXiv](https://arxiv.org/html/2505.18113).
- Ichikawa et al., "High-Dimensional Learning Dynamics of Quantized Models with STE" (arXiv 2510.10693, Oct 2025): in the high-dimensional limit STE dynamics converge to a deterministic ODE — [arXiv](https://arxiv.org/abs/2510.10693).
- "Stability and Generalization of STE for Training Two-Layer Quantized NNs" (arXiv 2609.06430, Sep 2026): finite-sample algorithmic-stability generalization bounds; explicitly notes prior population analyses do not study estimators trained on finite data — [arXiv](https://arxiv.org/html/2609.06430).

**Stochastic rounding and unbiased gradient estimation**
- Gupta et al., "Deep Learning with Limited Numerical Precision" (ICML 2015, arXiv 1502.02551) **[PK]**: SR is unbiased, E[Q(x)] = x; 16-bit fixed point with SR trains CNNs — [arXiv](https://arxiv.org/abs/1502.02551).
- Li, De, Xu, Studer, Samet, Goldstein, "Training Quantized Nets: A Deeper Understanding" (NeurIPS 2017, arXiv 1706.02379) **[PK]**: in the convex setting, SR-only training converges to an O(Δ) neighborhood of the optimum (cannot anneal), while BinaryConnect-style high-precision latent weights converge fully under (strong) convexity; nonconvex case: SR explores but cannot settle — [arXiv](https://arxiv.org/abs/1706.02379). This is the theoretical basis for why all low-bit training keeps high-precision master weights.
- Connolly, Higham, Mary, "Stochastic Rounding and Its Probabilistic Backward Error Analysis" (SIAM J. Sci. Comput. 2021, DOI 10.1137/20M1334796) **[PK]**: SR gives probabilistic error bounds O(√n·u) for inner products vs the deterministic nu worst case — [SIAM](https://doi.org/10.1137/20M1334796).
- Tseng et al. (MXFP4): RHT theoretically bounds the variance of SR; without it block outliers inflate variance — [arXiv 2502.20586](https://arxiv.org/pdf/2502.20586).
- Quartet II: MS-EDEN is provably unbiased in expectation with randomness placed on micro-scale factors; >2× lower error than element-wise SR — [arXiv 2601.22813](https://arxiv.org/html/2601.22813v1).
- Quartet: effN governed by forward MSE (bias in forward), effD by gradient-estimator bias; forward uses MSE-optimal clipping (biased RTN) because unbiased SR in the forward hurts — [liner](https://liner.com/review/quartet-native-fp4-training-can-be-optimal-for-large-language).
- UFP4/Shrinkage Bias: E2M1 under RTNE has a negative expected rounding error (geometric asymmetry of bins); compounds multiplicatively with depth and is amplified by RHT — [arXiv 2606.20381](https://arxiv.org/html/2606.20381v1). Assumption: input distribution roughly symmetric/continuous within bins; no large-scale isolation of the effect (Pith critique) — [Pith](https://pith.science/paper/2606.20381).
- "The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training" (arXiv 2603.10444, Mar 2026) — title only, likely a related bias analysis — [awesomepapers](https://awesomepapers.io/llm-papers/papers/2603.10444).
- Microsoft FP4's DGE replaces STE with a differentiable surrogate for the quantizer's derivative to reduce bias in weight updates — [ICML](https://proceedings.mlr.press/v267/wang25ae.html).

**Outlier / instability mechanisms (semi-theoretical)**
- Fishman et al.: SwiGLU w1–w2 alignment under ℓ2 regularization causes outlier growth — [Graphcore blog](https://graphcore-research.github.io/fp8_smooth_swiglu/).
- Dong et al.: outliers arise from Softmax, gating, SwiGLU; transient→persistent hot channels — [arXiv 2602.02047](https://arxiv.org/abs/2602.02047).
- TetraJet/TetraJet-v2: weight oscillation at bin boundaries as a dynamical failure mode — [arXiv 2510.27527](https://arxiv.org/abs/2510.27527).
- Depth Laws (2609.32060): noise-aware training's benefit decays with depth; amplification through residual stack — [arXiv](https://arxiv.org/abs/2609.32060).
- Topollai–Choromanska: quantized EMA state freezing model — [awesomepapers](https://awesomepapers.io/llm-papers/papers/2603.16731).

**Regularization / flatness / Hessian**
- Kumar et al. model low-precision effects as independent, accumulating noise (a modeling assumption) — [ICLR PDF](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf).
- (PK) Hessian-trace sensitivity arguments originate in PTQ work (HAWQ-V2, Dong et al., NeurIPS 2020, arXiv 1911.03852) and have not been shown to govern pretraining-time quantization noise — [arXiv](https://arxiv.org/abs/1911.03852) **[PK]**.
- No source retrieved in this session supports "quantization noise acts as implicit regularization improving generalization" at LLM pretraining scale; Spectra's result that ternary models beat FP16 on LAMBADA but lose on web perplexity is the closest empirical hint — [arXiv 2407.12327](https://arxiv.org/pdf/2407.12327).

### Inferences
- The theory stack splits cleanly: (i) STE theorems apply to the WEIGHT/ACTIVATION forward quantization with high-precision master weights (the BitNet/QAT regime) and require Gaussian data + 2 layers; (ii) SR theorems apply to GRADIENT quantization (the FP4 backward regime) and are distribution-free but only control bias/variance per step, not convergence of Adam. Nothing bridges them to transformer depth, attention, or adaptive optimizers.
- The 2026 "shrinkage bias" and "mean bias" papers imply that the standard E2M1 forward RTN introduces a systematic shrink of activations/weights that previous recipes compensated for implicitly (e.g., via LayerNorm/RMSNorm) — a potential explanation for why gaps widen at low LR when compensation dynamics slow.
- Li et al. 2017 (SR cannot anneal below O(Δ)) + Topollai–Choromanska (EMA states freeze) together predict that any attempt to remove high-precision master/state copies will need either SR with decreasing effective step or periodic resets — an open design space.

### Gaps
- No convergence or generalization theorem for STE/QAT or SR-based training on transformers (depth >2, attention, non-Gaussian tokens, Adam).
- Theorem statements/assumptions for Tseng's variance bound, Quartet II's unbiasedness proof, Topollai–Choromanska's staleness model and the shrinkage-bias derivation could not be read in this session.
- No sourced evidence either way for the "quantization noise = implicit regularization / flat minima" hypothesis at pretraining scale; no Hessian-trace measurement of FP4-trained LLMs was found.
- No analysis of the interaction between RHT, SR and RTN that reconciles the bias-amplification (UFP4) and variance-reduction (Tseng/Quartet) views.

---

## KQ7. Hardware trends and constraints shaping what is feasible

### Takeaway
Blackwell tensor cores natively support block-scaled MXFP8/MXFP6/MXFP4 (32-element blocks, UE8M0 scale) and NVFP4 (16-element blocks, UE4M3 scale + per-tensor FP32), roughly doubling FP4 vs FP8 throughput; the constraints that drive algorithm design are: scales are shared along the GEMM reduction (K) dimension, so each of the three training GEMMs needs a different quantization layout (motivating 2D weight scaling and "double quantization"); scale rounding direction (round-up vs floor) is decisive; SR hardware support exists for MXFP4 conversions but not uniformly; and FP8 tensor-core accumulation precision is limited, forcing FP32 promotion.

### Cited Findings
- OCP Microscaling (MX) spec: MXFP4 = E2M1 elements, 32-element blocks, one E8M0 (power-of-two, exponent-only) scale; effective 4.25 bits/element — [modal almanac](https://modal.com/llm-almanac/block-quants); [zeroentropy](https://zeroentropy.dev/concepts/nvfp4/). (PK: spec v1.0 Sept 2023, also defines MXFP8 E4M3/E5M2, MXFP6 E2M3/E3M2, MXINT8; companion paper Rouhani et al. arXiv 2310.10537.)
- UE8M0 values are 2^x for −127 ≤ x ≤ 127 — [Colfax CUTLASS tutorial](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/).
- NVFP4: 16-element blocks with an E4M3 (strictly UE4M3, non-negative) FP8 scale plus an optional outer FP32 per-tensor scale; UE4M3 offers more accuracy than UE8M0 at greatly reduced range; the OCP consortium did not adopt NVFP4 (single secondary claim) — [Colfax](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/); [zeroentropy](https://zeroentropy.dev/concepts/nvfp4/).
- Blackwell tensor cores read narrow-precision inputs with 8-bit scale factors per block of 16 or 32; PTX block-scaled MMA kinds include mxf8f6f4 (atom_K=32) and mxf4 / mxf4nvf4 (atom_K=64) — [Colfax](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/); [Blackwell low-level notes](https://brendanjameslynskey.github.io/NVIDIA_GPU_32_Blackwell_Low_Level/).
- NVIDIA reports NVFP4 training support in JAX/MaxText on Blackwell — [NVIDIA dev blog](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/). (PK: GB200 FP4 ≈2× FP8 dense throughput; GB300/Blackwell Ultra ≈3× — claimed in the NVFP4 pretraining paper intro **[PK: verify]**.)
- Scale-rounding constraint: OCP MX conversion effectively floors the scale exponent, which saturates/clips and can diverge LLM pretraining; NVIDIA's MXFP8 recipe rounds up — [Pith](https://pith.science/paper/2506.08027); the NVFP4 pretraining paper is cited for "rounding scales up to the next UE8M0 value avoids saturation" — [zeroentropy](https://zeroentropy.dev/concepts/nvfp4/).
- Transposition constraint: because scales are shared along K, the weight GEMM in backward contracts over a different axis than forward; NVIDIA uses 2D 16×16 weight blocks so the same quantized weights serve forward and backward — [arXiv 2509.25149](https://arxiv.org/html/2509.25149v1); TetraJet uses "double quantization" to handle MXFP4's non-square group — [arXiv 2502.20853](https://arxiv.org/html/2502.20853v1); TetraJet-v2 "double-block quantization" for NVFP4 scale numerics — [arXiv 2510.27527](https://arxiv.org/abs/2510.27527).
- Accumulation constraint: H800 FP8 tensor cores accumulate at limited precision, so DeepSeek promotes partial sums to FP32 CUDA cores every 4 WGMMA; DeepSeek asks hardware vendors for higher accumulation precision and native fine-grained scaling — [arXiv 2505.09343](https://arxiv.org/pdf/2505.09343).
- SR hardware support: Quartet notes SR relies on specialized operations supported for MXFP4 but possibly lacking for other formats — [themoonlight](https://www.themoonlight.io/en/review/quartet-native-fp4-training-can-be-optimal-for-large-language-models).
- Scale-format proposals from the algorithm side: Four-over-Six adaptive block scaling for NVFP4 (arXiv 2512.02010) — [arXiv](https://arxiv.org/pdf/2512.02010); UE5M3 scale (arXiv 2609.02846) — [arXiv](https://arxiv.org/pdf/2609.02846); HiFloat4 for Ascend NPUs (arXiv 2604.08826) — [arXiv](https://arxiv.org/html/2604.08826v1); UFP4 argues for uniform E1M2/INT4 elements instead of E2M1 — [Pith](https://pith.science/paper/2606.20381).
- Pre-Blackwell FP4 studies were simulated on FP8 hardware (Microsoft FP4) — [arXiv 2501.17116](https://arxiv.org/html/2501.17116v2); Fishman et al. ran FP8 on Gaudi2 — [arXiv 2409.12517](https://arxiv.org/abs/2409.12517).

### Inferences
- Hardware fixed the element format (E2M1) and block sizes (16/32) before the training community had evidence; the 2026 UFP4, UE5M3 and HiFloat4 papers are all arguments that these choices are suboptimal for training — a sign the format question is being reopened for the next generation.
- The three-GEMM layout problem (forward, dgrad, wgrad each want scales along a different axis) is the single most binding constraint: it forces either 2D blocks (coarser → more clipping) or re-quantization (→ forward/backward mismatch, biased gradients). Any training-specific hardware proposal should target this.
- Because FP4 throughput gains are only ~2× over FP8 while FP4 recipes add RHT, SR, 2D scaling, BF16 tails and late-phase switching, realized end-to-end speedups are far below 2×; no paper reports a measured end-to-end FP4 pretraining speedup at ≥7B with a BF16-quality match.

### Gaps
- Primary OCP MX v1.0 spec and NVIDIA PTX ISA text were not retrievable; bit-level details above come from secondary sources.
- No published measurement of Blackwell FP4 tensor-core accumulation precision or of SR conversion throughput.
- No end-to-end wall-clock FP4 pretraining speedup with quality parity at ≥7B found (Quartet II's 4.2× and Quartet's speedups are kernel/layer-level or third-party).
- Which GEMMs (forward vs dgrad vs wgrad) each 2026 recipe quantizes could only be partially determined.

---

## KQ8. Cross-cutting contradictions and open problems (consolidated for idea generation)

### Takeaway
The literature contains at least six unresolved contradictions and a dozen untested extrapolations; most stem from the absence of controlled ≥7B / ≥1T-token studies with BF16 controls and from theory that stops at two-layer networks.

### Cited Findings (contradictions, each with both sides sourced)
- SR + randomized Hadamard for gradient GEMMs: works (Tseng [2502.20586](https://arxiv.org/pdf/2502.20586); Quartet [2505.14669](https://arxiv.org/pdf/2505.14669); NVIDIA [2509.25149](https://arxiv.org/html/2509.25149v1)) vs fails when Wgrad is quantized at 8B, deterministic Hadamard needed ([2605.09825](https://arxiv.org/html/2605.09825v3)).
- RHT reduces SR variance (Tseng) vs RHT amplifies E2M1 shrinkage bias (UFP4 [2606.20381](https://arxiv.org/html/2606.20381v1)).
- Late BF16 switch closes the FP4 gap (NVIDIA [2509.25149](https://arxiv.org/html/2509.25149v1)) vs switching all tensors to BF16 did not fix divergence (Nemotron 3 Ultra [2606.15007](https://arxiv.org/pdf/2606.15007), per search summary).
- From-scratch ternary (BitNet [JMLR](http://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf); Spectra [2407.12327](https://arxiv.org/abs/2407.12327)) vs FP-then-QAT superior (ParetoQ [2502.02631](https://arxiv.org/abs/2502.02631)).
- More tokens hurt low-bit (Kumar [2411.04330](https://arxiv.org/html/2411.04330); Ouyang [2411.17691](https://arxiv.org/abs/2411.17691); Chen [2505.14302](https://arxiv.org/abs/2505.14302)) vs tokens are the better investment for ternary models (Spectra 1.1 [2506.23025](https://arxiv.org/html/2506.23025)).
- Primary FP8 failure is underflow of small values (FP8-LM scaling design [2310.18313](https://arxiv.org/pdf/2310.18313)) vs primary failure is saturation from scale rounding-down (Mishra et al. [Pith](https://pith.science/paper/2506.08027)) vs SwiGLU outlier overflow (Fishman [2409.12517](https://arxiv.org/abs/2409.12517)).
- Quartet scaling-law functional form reported inconsistently across secondary sources ([themoonlight](https://www.themoonlight.io/en/review/quartet-native-fp4-training-can-be-optimal-for-large-language-models) vs [alphaxiv](https://www.alphaxiv.org/abs/2505.14669)).
- Compute-optimal precision 7–8 bits (Kumar, integer) vs "FP4 can be optimal" (Quartet) vs 4–8 bits cost-optimal (Sun et al. [2501.02423](https://arxiv.org/pdf/2501.02423), PK).

### Inferences (candidate open problems)
- A unified bias–variance accounting of the FP4 gradient estimator (RTN shrinkage bias + SR variance + RHT interaction + scale-format quantization) as a function of block size and scale format is missing and would arbitrate three of the contradictions above.
- The LR-decay-phase gap growth suggests a "quantization-noise floor" that becomes visible when LR × gradient falls below the rounding scale; a scaling law with an LR-dependent term (none exists) could predict when to switch precision, replacing the empirical late-BF16 switch.
- Optimizer-state staleness (Topollai–Choromanska) and SR's O(Δ) floor (Li et al. 2017) are the same phenomenon in different tensors; a shared theory/fix (reset schedules, SR on states, Muon-style magnitude-free updates) is unexplored.
- Precision scaling laws fit only one of {integer PTQ, QAT, FP block-scaled training, ternary}; a law with format-aware terms (E/M bits, block size, scale format) validated at ≥7B is absent.
- The emergent ~50% zero fraction in ternary models and the "hot channel" persistence in FP4 both point at training-time structure (sparsity, channel outliers) that could be exploited or regularized during pretraining rather than patched at quantization time.

### Gaps (research-process)
- Everything above is from search summaries and tagged prior knowledge; theorem statements, exact equations and limitations sections of ~15 key papers (Quartet, Quartet II, Tseng, UFP4, Topollai–Choromanska, Chen QAT law, Sun FP law, Compute-Optimal QAT, TetraJet-v2, UE5M3, 2605.09825, Full-Stack FP4, COAT, FP8-LM, Kumar) need a direct read before being quoted.
- Communication-quantization theory for LLM pretraining (EF21/QSGD at FSDP scale, FP8 all-reduce theory) and the "implicit regularization / flat minima" literature were not reachable before the search budget ran out; treat those sections as incomplete.
- Venue confirmations (Quartet NeurIPS 2025, Topollai–Choromanska EMNLP 2026, Chen ICML 2026) rest on single secondary sources.
