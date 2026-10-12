# New idea generation — TRAINING-SIDE lens (2026-10-11)

Scope: FP8/FP4/MX pretraining dynamics, optimizer-state / master-weight quantization, gradient & communication quantization, QAT, quantized RL rollouts / RL fine-tuning, distillation into low-bit students.
Inputs read: report §0 / §0.4 (preempted list), `research_notes/arxiv_sweep/theorems/*` (training papers: 2411.04330, 2411.17691, 2502.02631, 2505.14302, 2505.14669, 2506.08027, 2509.22935, 2509.25149, 2510.06213, 2510.11696, 2601.22813, 2602.02047, 2603.16731, 2606.20381, 2607.26515), `recent/*`, `hypotheses/h00.md`, `judge/idea4.md`, `judge/idea6.md`.
New full texts grepped this run (untrusted data, quoted only): 2605.20402, 2601.22813, 2607.24953, 2606.15682, 2410.19313, 2607.15810, 2511.00874, 2602.02555, 2608.09805, 2601.18150, 2603.02170, 2609.39813, 2609.20807, 2610.12444, 2608.22322, 2607.04422, 2607.10611, 2510.21314, 2610.07767, 2603.02731, 2610.00053, 2609.39223.
Search tool: `scripts/arxiv_search.py` through a retry wrapper (the arXiv API intermittently returns empty feeds; "0 hits after 3 tries" below means three empty responses). The API is literal, so many conjunctive queries return 0; I compensated with broad date-sorted sweeps (`abs:"FP4" AND abs:pretraining`, `abs:"optimizer states" AND abs:quantization`, `abs:"on-policy" AND abs:quantization AND abs:distillation`, `abs:"weight synchronization" AND abs:reinforcement`) and title scanning.

## Ideas generated and DROPPED by the self prior-art screen (for the record)

| Draft idea | Killed by | Why |
|---|---|---|
| Persistent fwd/bwd transpose-mismatch bias + "re-quantize the forward-quantized W" fix | 2601.22813 l.710 ("saved quantized weights and activations are then de-quantized, transposed and then re-quantized with MS-EDEN"); 2609.39813 (backward-state policy) | Quartet II already re-quantizes the forward tensor; 2609.39813 makes "which rounded copy the backward reads" its whole contribution |
| SR backward noise shifts the critical batch size | 2511.00874 ("increased batch sizes can compensate for reduced precision during back-propagation", Lemma 3) | same contribution for SGD |
| RTN weight-sync deadband lag in quantized-rollout RL | 2609.21834 (cell theorem, preempted list), 2609.20807 (drift = persistent bias accumulating per step; score centering) | thin variant |
| On-policy (student-sampled) QAT/distillation, DAgger bound | 2610.09346 OnlineQAT, 2609.26708 "Train Where the Quantized Model Goes" | same contribution (Sept–Oct 2026) |
| FP8 second moment = max-tracker; companding reconciles COAT vs stalling | 2603.16731, 2610.12444 (ZIP-SR, preconditioner-space rounding), 2608.22322 (adaptive log-space), 2510.21314 | area saturated; increment too thin |
| FP8/INT8 attention backward: dS = P∘(dP−δ) cancellation | 2603.02170 SageBwd ("quantization errors primarily arise from the backward-pass score gradient dS") | preempted |
| Low-load MoE experts suffer more FP4 Wgrad noise | own analysis | premise wrong: SR noise and data noise both scale 1/n_e, ratio independent of load |
| Error-feedback low-bit delta weight sync for RL | 2610.08430 NeMo-DCR (bit-exact delta refit), 2605.07330 SparseRL-Sync (lossless, ~100x), 2608.00455 | lossless sparse sync dominates lossy EF |
| Logit-scale growth explains growing quantized-rollout mismatch | 2609.20807 (drift), judge/idea6 H1 | thin |
| LM-head token-axis push-down underflow | own analysis | every surveyed recipe keeps LM head BF16 (2509.25149, 2610.00053 "bfloat16 output projections"); effect only under RTN |

---

## 1. Block-scale and shrinkage bias do NOT compound exponentially through depth: a residual-linearization theorem and a refutation of the "multiplicative backward accumulation" story

- **Stage:** FP4/MX pretraining and FP4 RL training (backward pass).
- **Hypothesis.** In pre-norm residual LLMs, a per-layer quantization gain error ε_l (from power-of-two scale rounding or E2M1 shrinkage) changes the backpropagated gradient norm by a factor 1 + Σ_l c_l ε_l + O(ε²) with Σ_l c_l ≤ 1, not by Π_l (1+ε_l). Measured log gradient-norm ratios of MXFP4 (E8M0 ceil) vs BF16 will have std ≪ 1 and will not grow like √L. The RL gain of Macro-Block Scaling is explained by lower grid/deadzone MSE, not by removing exponential gradient-scale drift.
- **Gap evidence (concrete inconsistency).**
  - 2605.20402 §4.1 Eq.(4) (txt l.512–551): "Scale bias is multiplicative: each layer's output is scaled by γ_b = s_b/s*_b … In the backward pass … the STE chain rule multiplies scale factors across all L layers without any normalization", hence log(‖∇̂‖/‖∇‖) ≈ Σ_l δ_b(l), std √(L/12) = 1.73 for L=36, "ratio ranges from ∼0.18× to ∼5.6×". Fig. 3b ("emp. σ = 1.97, theory σ = 2.00", l.479–495) is a histogram of the **sum of scale residuals Σδ**, not of a measured gradient ratio; the gradient claim is never measured.
  - The same paper (l.~380) reports that the "scale" and "grid" components have a cross term ≈ −1.46‖e‖², i.e. they largely cancel; with identical encode/decode scale the dequantized value carries no multiplicative γ factor at all.
  - It ignores the residual stream (it says LayerNorm "resets" the forward, l.515, but treats the backward as a plain chain).
  - UFP4 2606.20381 also asserts multiplicative backward accumulation of E2M1 shrinkage (l.14, l.159, l.1035 per hypotheses/h00.md) with only MSE-type evidence (theorems note §"Line 162").
  - Contradicting theory: 2609.32060 (recent note, abstract level) gives a depth law G² = G₀² + κ s² D, i.e. **linear** growth of the precision-floor amplification with depth.
- **Math core.**
  - Prop. 1 (decode-scale identity). For x̂ = s·q(x/s) with the same s at encode and decode, E[x̂ − x | x] = s·b(x/s), with b the grid's rounding-bias function. A scale error only changes the argument of b; there is no (γ−1)·x term. Proof: one line.
  - Prop. 2 (residual linearization). Take h_{l+1} = h_l + α_l F_l(N(h_l)) with α_l = 1 + ε_l. Then ∂L/∂h_0 = Π_l (I + α_l J_l), and log‖∇̂‖ − log‖∇‖ = Σ_l ε_l c_l + O(‖ε‖²), with c_l the branch-l share of ⟨g, ·⟩. A product Πγ_l arises only for nested branch paths, which are higher order. Assumptions: pre-norm, ‖J_l‖ bounded, ε small. Proof: expand the product and keep first order.
  - Prop. 3 (block averaging). If each of n_b blocks in a layer has an independent γ_b, the layer-level gain error has variance Var(γ)/n_b times a participation factor. So even the per-layer ε_l concentrates.
  - Together these predict std(log ratio) ≈ sqrt(Σ c_l² Var(γ)/n_b,eff) ≪ 1.73.
- **Decisive experiment (≤1 GPU-week).**
  1. Qwen2.5-0.5B and 3B (L=24/36) with MXFP4 QDQ (E8M0 ceil, the setting of 2605.20402) in forward+backward, against BF16 on 64 fixed batches. Log per-tensor gradient-norm ratio and cosine at every layer.
  2. Repeat on a no-residual ablation (residual scaled ×0 in a 12-layer toy) to show that the exponential regime appears only there.
  3. Run GRPO on GSM8K for 200 steps on Qwen2.5-1.5B with four arms: MBS; a "scalar per-tensor debias" (free); 4/6-type MSE reduction without a mantissa; plain.
  - **Failure criterion:** measured std of the log ratio ≥ 1.0, or growth ∝ √L, falsifies Prop. 2's relevance. If the scalar debias recovers ≥ 80% of MBS's gain, the "gradient compounding" mechanism is refuted in favour of a gain-only effect.
- **Nearest papers.** 2605.20402 asserts the opposite and gives no measurement. 2606.20381 has the same assertion and only a bin-level bias. 2609.32060 is a linear depth law for the precision floor; this idea uses it as corroboration and adds the decode-scale identity, the block-averaging argument and the direct measurement. 2601.22813 studies unbiasedness, not depth.
- **Self prior-art queries:** `abs:"shrinkage" AND abs:FP4` (--sort date --from 20250601) → only 2606.20381; `abs:E2M1 AND abs:bias AND abs:training` (date) → only 2606.20381; `abs:"scale invariance" AND abs:normalization AND abs:quantization AND abs:bias` (adjacent vocabulary) → 0; `abs:"multiplicative bias" AND abs:quantiz AND abs:normalization` → 0.
- **Verdict.** Novelty medium. It is a refutation plus a corrected mechanism; main-track only if the measurement is clean and the debias arm matches MBS.

## 2. Pretraining precision sets the QAT budget: forward quantization noise as an implicit curvature regularizer

- **Stage:** pretraining → QAT for deployment.
- **Hypothesis.** The compute-optimal QAT fraction f* (Dremov et al.) shrinks when pretraining already runs the forward pass in a quantized format. The shrinkage is predicted by the alignment between the pretraining noise covariance Σ_pre and the deployment noise covariance Σ_tgt: NVFP4-forward pretraining → W4 target needs f* close to 0; FP8 pretraining → about the BF16 value.
- **Gap evidence.** 2509.22935 §6 (theorems note §4): "the relationship between the optimal QAT fraction and pretraining precision remains unknown". App. R.1 only speculates that f* is smaller for fp4/fp8 pretraining. Its law f̂ = exp(−6.7297/ln S_total) is weight-only and fitted on BF16 pretraining from 86M to 759M.
- **Math core.**
  - Training on E_ε L(w+ε) adds ½tr(HΣ_pre).
  - On a manifold of minima, noise-injected SGD drifts along −∇_θ tr(H(θ)Σ_pre). This is the known implicit-regularization result for label/parameter-noise SGD; cite it, do not reprove it.
  - Prop.: the PTQ penalty at the target is ½tr(HΣ_tgt). After pretraining with Σ_pre it is reduced by at least the projection of the drift onto ∇tr(HΣ_tgt). So the residual QAT work scales with 1 − cos_H(Σ_pre, Σ_tgt) and with the ratio of magnitudes.
  - Provable in a diagonal-linear-network toy. For LLMs the claim is empirical.
- **Decisive experiment (~2 GPU-weeks).**
  - 150M (full grid) and 400M (3 fractions). Pretraining forward ∈ {BF16, FP8 E4M3, NVFP4 forward-only}, QAT fraction ∈ {0, 5, 10, 20, 40}%, target ∈ {INT4 g128, NVFP4}, 20 tokens/param.
  - Also measure tr(HΣ_tgt) by Hutchinson at the end of pretraining.
  - **Failure criterion:** f* is identical across pretraining precisions, within Dremov's MAE of 0.091; or NVFP4-pretrained checkpoints show no lower INT4/NVFP4 PTQ penalty than BF16 ones at matched loss.
- **Nearest papers.** 2509.22935 poses the question but has no precision axis. 2411.04330 fits a joint law but uses integer forward-only QAT and gives no f*. 2502.02631 (10% rule, only at 125M/100B). 2505.14302 (QAT law, no pretraining-precision axis). 2606.09012 (QAT basin theorem, abstract level).
- **Self prior-art queries:** `abs:"quantization-aware training" AND abs:"pretraining precision"` → 0; `abs:"QAT" AND abs:"FP8" AND abs:pretrain` (date) → only an unrelated CV paper; `abs:"optimal QAT fraction"` → 0; `abs:"precision switch" OR abs:"switching precision"` (date, adjacent: control/numerics) → nothing relevant.
- **Verdict.** Novelty medium (an explicitly stated open problem). Main-track as a scaling-law-with-mechanism paper if the alignment predictor beats Dremov's law on held-out runs.

## 3. Why fusing QAT with the LR cooldown hurts at 1–2 bits: a displacement-budget law for QAT schedules

- **Stage:** QAT.
- **Hypothesis.** Low-bit QAT must change the codes of an O(1) fraction of weights ("reconstruction"). That requires latent displacements of order Δ_b. 4–6-bit QAT only needs sub-cell compensation. Cooldown fusion halves the LR integral available during QAT, so it hurts exactly when the required code-flip fraction exceeds about 2I/Δ_b. The crossover bit width moves predictably with peak LR × QAT length.
- **Gap evidence (unexplained numbers).**
  - 2509.22935 App. Table 10 (theorems note §5.3): cooldown fusion helps at 4 and 6 bits but is worse at 1 bit in 5 of 7 settings and at 2 bits in 4 of 11 settings. The authors' explanation ("updates lost during cooldown") is labelled speculation.
  - 2502.02631: the 2↔3-bit "compensation vs reconstruction" transition is only a hypothesis. It is confounded by 3× more steps at 2× the LR and a quantizer switch (theorems note).
- **Math core.**
  - Lemma: under sign-like Adam updates, the per-coordinate displacement in a phase is at most I = ∫η dt.
  - With cell offsets uniform, the fraction of codes that can flip is at most min(1, 2I/Δ_b).
  - Prop.: if the target requires a flip fraction r_b, then fused QAT is infeasible whenever r_b > 2I_fused/Δ_b, and the excess loss is lower-bounded by the curvature-weighted mass of unflipped weights.
  - Provable in a separable quadratic toy. The LLM transfer is empirical.
- **Decisive experiment (~1 GPU-week).**
  - 125M and 350M models; QAT at {1.58, 2, 3, 4} bits; fused vs separate cooldown; QAT-phase LR integral ×{0.5, 1, 2}; also a "fused + re-warm" arm.
  - Log the code-flip fraction per bit width.
  - **Failure criterion:** the flip fraction at ≤2 bits is not larger than at 4 bits, or the fusion penalty is not monotone in Δ_b/I.
- **Nearest papers.** 2509.22935 (the observation itself, no mechanism), 2502.02631 (transition hypothesis), 2606.09012 (QAT basin), 2510.06213 (LR and PTQ robustness, not QAT).
- **Self prior-art queries:** `abs:"cooldown" AND abs:"quantization-aware"` (date) → only 2509.22935; `abs:"quantization-aware training" AND abs:"learning rate schedule"` (date) → 0; `abs:ternary AND abs:"learning rate" AND abs:"cell"` → 2609.29397 (60K-param ternary re-examination, different question); `abs:binary AND abs:"weight flips" AND abs:"learning rate"` (adjacent BNN vocabulary) → 0.
- **Verdict.** Novelty medium-high. Main-track potential medium: a schedule rule plus a mechanism.

## 4. Token-axis block scaling in weight-gradient GEMMs: magnitude-sorted token blocking for low-precision RL fine-tuning

- **Stage:** FP4/FP8 RL trainer (and pretraining as a control).
- **Hypothesis.** In Wgrad = δᵀX the reduction axis is tokens, and NVFP4/MXFP4 share one scale per 16/32 consecutive tokens. In RL, per-token gradient magnitudes are extremely heterogeneous: ∝ |A_i|·(1−p_t), and they are dominated by a minority of forking tokens. Block maxima are therefore set by a few tokens, and the SR variance of the others is inflated (under RTN they are zeroed). Permuting tokens so that blocks group similar magnitudes cuts the Wgrad error at zero GEMM cost and closes part of the FP4-RL gap. For RL the sort key |A_i|(1−p_t) is available from the loss for free.
- **Gap evidence.**
  - Recipes treat the token axis only with RHT: 2509.25149 uses 16-point RHT on Wgrad inputs; 2610.00053 uses "row-gradient stochastic rounding and fixed-sign 32-value Hadamard weight-gradient preconditioning" (abstract).
  - 2601.22813 Table 1 measures SR 1×16 at 23.5e-3 vs RTN 9.0e-3, but only on Gaussian inputs, i.e. homogeneous magnitudes.
  - RL heterogeneity: 2506.01939 (high-entropy minority tokens drive learning) and 2606.15682 ReQAT (critical low-entropy tokens).
  - 2605.20402 treats deadzone only in the forward/rollout.
  - No paper found exploits the permutation invariance of the reduction axis.
- **Math core.**
  - Theorem (separable model |δ_{t,i}| = a_t c_i): under block-max scaling with SR, Var[Ŵgrad_{ik}] ≤ (Δ²/4) c_i² Σ_b n_b max_{t∈b} a_t² · max_t x_{tk}².
  - Among partitions into equal-size blocks, the sorted contiguous partition minimizes Σ_b max_{t∈b} a_t². Proof by exchange argument; the result is exact for the bound.
  - Comparison with RHT: after a Hadamard over n tokens, the block max ≈ sqrt(2 ln n)·rms. So sorting beats RHT when the a_t distribution has a heavy tail (Pareto α < 2 regime) and loses for near-uniform a_t. Both statements are provable under the separable model.
  - Exactness: the GEMM is invariant to a shared row permutation of δ and X.
- **Decisive experiment (~1 GPU-week).**
  - GRPO on Qwen2.5-1.5B (MATH/GSM8K) with an NVFP4-emulated trainer. Arms: SR+RHT (NVIDIA-style), SR+sorted, SR+sorted+RHT, BF16.
  - Measure (a) per-layer relative Wgrad error vs BF16 on 32 fixed batches and (b) reward/accuracy over 300 steps, 3 seeds. Control: 125M pretraining, where heterogeneity is low and the predicted gain is small.
  - **Failure criterion:** Wgrad error reduction < 10% vs RHT in RL, or no accuracy difference beyond seed noise.
- **Nearest papers.** RPTQ-style channel reordering for inference PTQ (reorders the K/channel axis of activations, not the training token axis); 2509.25149 and 2610.00053 (RHT on the token axis); 2601.22813 (Gaussian-only error tables); 2603.02731 (row-wise to column-wise FP4 conversion for Wgrad, no reordering).
- **Self prior-art queries:** `abs:"weight gradient" AND abs:FP4 AND abs:token` (date) → 2610.00053, 2605.09825 (no reordering); `abs:permutation AND abs:"block quantization" AND abs:gradient` (date) → 0; `abs:reorder AND abs:tokens AND abs:quantization AND abs:training` (adjacent: reordering) → MQuant, LightMamba, VC-Attention (inference); `abs:"sparse reduction" AND abs:"stochastic rounding"` → 0.
- **Verdict.** Novelty medium-high. A cheap method with an exact optimality lemma; main-track if the RL gains are real.

## 5. Shared-seed stochastic rounding across data-parallel replicas: positive error correlation that caps 1/P averaging, and a correlated-quantization fix

- **Stage:** FP4/FP8 backward + data-parallel gradient reduction (gradient/communication quantization).
- **Hypothesis.** If SR random numbers are generated from the same RNG state on every data-parallel rank (Megatron-style seeding is identical across DP ranks unless explicitly offset — **to be verified in step 0**), the SR errors of different ranks are positively correlated. The same holds across gradient-accumulation micro-batches if the RNG offset is not advanced. The all-reduced gradient's quantization noise then stops decaying as 1/P. Rank-stratified dither from the distributed-mean-estimation literature makes the errors anti-correlated and beats independent SR.
- **Gap evidence.**
  - Every "unbiased FP4 training" recipe argues per-step unbiasedness only: 2509.25149 (SR on gradients), 2505.19115 (SR backward), 2601.22813 Corollary 3.1. None states the independence assumption across replicas.
  - Quartet II App. A (l.1535–1548) concedes that EDEN needs independently sampled rotations, yet applies the "same rotation seed" (l.546) to both tensors and "identical rotations for every rotation group within a tensor for every micro-batch" (l.1544).
  - The adjacent field shows that correlation structure is a first-order design variable: correlated quantization for DME (2203.04925 Suresh et al.; 2401.05518 Panferov et al.).
- **Math core.**
  - Theorem: let residues f_r ~ iid U(0,1) and SR errors e_r = 1[u_r < f_r] − f_r.
    - With a shared u: Cov(e_r, e_s) = E[min(f_r, f_s)] − E[f_r f_s] = 1/3 − 1/4 = 1/12, against Var = 1/6, so ρ = 1/2. Hence Var(mean_P e) = 1/(6P) + (1−1/P)/12 → 1/12: at most a 2× reduction for any P.
    - With independent u it is 1/(6P).
    - With stratified u_r = (π(r) + U)/P it is O(1/P²).
  - Transfer to Wgrad: for Ŵ_ik = Σ_t ê_{t,i} x_{t,k} with independent data across ranks, the cross-rank covariance equals Cov(e)·E[x_k]². So the effect concentrates in channels with nonzero activation means, e.g. post-SwiGLU FC2 inputs and massive-activation channels. This is a falsifiable channel-level prediction.
  - Proofs are elementary and exact.
- **Decisive experiment (~1 GPU-week).**
  - Step 0: check whether the RNG for SR in Megatron/Transformer Engine NVFP4 is identical across DP ranks; measure the empirical cross-rank error correlation per tensor at DP=8.
  - Then train 350M with NVFP4 SR backward at DP ∈ {8, 32} under shared, independent and stratified seeds, 3 seeds each; report loss gaps and per-channel noise vs E[x_k]².
  - **Failure criterion:** measured cross-rank correlation < 0.05 in all tensors (practice already decorrelates, or the mean-channel transfer is negligible), or no loss difference between shared and independent seeds.
- **Nearest papers.** 2203.04925 and 2401.05518 (correlated quantization for distributed mean estimation, no FP4 training, no SR-seed analysis); 2511.00874 (SR × batch size, assumes independence); 2601.22813 (rotation reuse, empirical 1/B check only on a single GPU, Fig. 9).
- **Self prior-art queries:** `abs:"stochastic rounding" AND abs:correlated` → only 2607.18745 (unrelated identity); `abs:"stochastic rounding" AND abs:"data parallel"` (date) → 0; `abs:"shared randomness" AND abs:quantization AND abs:gradient` (date) → 0; `abs:dither AND abs:"common randomness" AND abs:averaging` (adjacent DME/IT vocabulary) → 0; plus `abs:"correlated rounding" OR abs:"correlated quantization"` → 2203.04925, 2401.05518 (adjacent, cited).
- **Verdict.** Novelty high, if the step-0 premise holds in common frameworks. If it does not, the stratified-dither method (a strict improvement over independent SR) still stands as the contribution.

## 6. Rotation-randomness reuse in unbiased FP4 backward estimators: a persistent bias floor from fixed or shared Hadamard signs

- **Stage:** FP4 pretraining backward (estimator theory).
- **Hypothesis.** MS-EDEN and RHT-based backward estimators are unbiased only in expectation over fresh rotations. Practical recipes reuse rotations: identical across groups and micro-batches (Quartet II), or a single fixed sign vector for all of training (2610.00053: "fixed-sign 32-value Hadamard"). For a deterministic rotation the EDEN correction S and the RTN error become deterministic functions of the operand. For the **weight** operand, which changes slowly, this is a bias that persists across steps and does not average out. The resulting loss floor is measurable and is removed by per-step resampling at negligible cost.
- **Gap evidence.**
  - 2601.22813 App. A l.1535–1548: "EDEN … only guarantees unbiasedness in the d → ∞ limit and requires rotations to be sampled independently, in practice we make a number of compromises … identical rotations for every rotation group within a tensor for every micro-batch". Unbiasedness of the implemented scheme is only checked empirically (Fig. 9, single attention block, ~1/B decay). Corollary 3.1 is stated without proof (theorems note §1).
  - 2610.00053 abstract: fixed-sign H32 Wgrad preconditioning.
- **Math core.**
  - Prop. (bias of a frozen rotation): for a fixed orthogonal R and RTN q, b(W) = Rᵀq(RW) − W is a deterministic function. Across steps, W_t − W_{t−1} = O(η), so Corr(b(W_t), b(W_{t+k})) ≈ 1 for k ≪ Δ/η. The accumulated parameter drift is then O(Tη‖b‖) instead of O(√T η‖b‖) under per-step resampling.
  - With shared rotations across G groups, the group errors are exchangeable but not independent, so the variance of the group sum does not fall as 1/G.
  - Bounds follow from standard SGD-with-bias arguments: a bias floor of order ½ bᵀH⁻¹b. Exact under a quadratic model.
- **Decisive experiment (~1 GPU-week).**
  - 125M–350M NVFP4 training with Quartet II's kernels (open source) under three rotation policies: fixed for all training, per-step shared across groups, per-step per-group independent. Measure the step-to-step autocorrelation of the backward error for weight operands vs activation operands, and the final loss gap over 3 seeds.
  - **Failure criterion:** weight-operand error autocorrelation across steps < 0.2, or loss differences < seed std.
- **Nearest papers.** 2601.22813 (makes the compromise, does not quantify it); 2609.39813 (backward-state policy: reuse vs fresh **rounding** of stored tensors, at 390M; not rotation reuse or bias persistence); 2609.08759 (EDEN scaling note, no reuse); 2604.18555 (DRIVE/EDEN lineage note).
- **Self prior-art queries:** `abs:"randomized Hadamard" AND abs:training AND abs:unbiased` (date) → 2606.23406 (PTQ), 2509.25149; `abs:EDEN AND abs:rotation AND abs:quantization` (date) → 2609.08759, 2605.19972, 2604.18555 (none on reuse); `abs:"random rotation" AND abs:"independent" AND abs:"distributed mean estimation"` (adjacent) → 0; `abs:"correlated rounding" OR abs:"correlated quantization"` → DME papers only.
- **Verdict.** Novelty medium. Main-track only combined with idea 5 as a "randomness budget of unbiased low-precision training" paper.

## 7. Quantized rollouts with truncated IS induce a predictable length drift: termination-hazard asymmetry

- **Stage:** quantized-rollout RL (W4/NVFP4 sampler, BF16 learner, token-level TIS).
- **Hypothesis.** A 4-bit sampler over-emits hesitation markers and under-emits commit/termination tokens. That gives per-token ratios ρ_t = π/μ < 1 on the former (never truncated) and ρ_t > 1 on the latter (truncated at C). TIS therefore under-reinforces termination systematically, and the learned policy's response length drifts upward relative to a BF16-sampler run at matched reward. The size of the drift is predicted before training from the sampler's hazard ratio κ = h_μ/h_π.
- **Gap evidence.**
  - 2606.00206 (theorems note): quantized reasoning models produce longer CoT; length/accuracy Spearman ρ = −0.73 over 28 model-quantization pairs; penalizing overthinking markers cuts length 12–23%. No RL.
  - 2609.20807 shows that TIS fails under heavy sampler quantization because a persistent bias ("drift") accumulates, but does not predict its direction or an observable.
  - 2601.18150 reports overlapping length curves for **FP8** rollouts (l.266, l.302). The prediction is therefore specific to ≤4-bit samplers, which makes it falsifiable.
- **Math core.**
  - Prop.: under a geometric hazard model with learner hazard h and sampler hazard κh (κ < 1), the TIS gradient on the termination logit is scaled by E_μ[min(ρ, C)] = κ·min(1/κ, C) relative to the on-policy gradient. If 1/κ > C, termination is under-reinforced by the factor κC, and the expected length after K steps drifts by Δ log E[len] ≈ −η K A_term (1 − κC) h (first order).
  - This is an application of the known TIS bias formula E_μ[(ρ−C)_+ A ∇log π] with a quantization-specific sign; it is not a new estimator.
- **Decisive experiment (~1–2 GPU-weeks).**
  - GRPO on Qwen2.5-Math-1.5B/7B, sampler ∈ {BF16, FP8, W4A16 AWQ, NVFP4 W4A4}, TIS C ∈ {2, 5}, plus a score-centering arm (2609.20807) and a "hazard-corrected" arm that leaves termination tokens untruncated.
  - Measure κ offline on 1k prompts before RL, then track length and the frequency of "wait"-type markers over 300 steps.
  - **Failure criterion:** the length difference between 4-bit and BF16 samplers is < 5%, or its sign or size is uncorrelated with κ across arms.
- **Nearest papers.** 2609.20807 (drift, score centering), 2609.32444 (CIS confidence-aware truncation; bf16 mismatch), 2607.22186, 2605.13907 (AIS), 2606.00206 (length under PTQ, no RL).
- **Self prior-art queries:** `abs:quantized AND abs:rollout AND abs:length` (date) → 2601.18150, 2606.18967 (no length-bias analysis); `abs:"truncated importance sampling" AND abs:length` (date) → 2509.23866 (GUI agents, unrelated); `abs:quantization AND abs:"response length" AND abs:reinforcement` (date) → 0; `abs:"overthinking" AND abs:quantized AND abs:reinforcement` (date) → 0; plus `abs:"training-inference mismatch" AND abs:grows` (date) → 2609.20807, 2606.29526, 2603.19470.
- **Verdict.** Novelty medium-low: the mechanism is generic TIS bias, and the novelty is the signed, pre-registered length prediction. Workshop-to-main depending on the size of the effect.

## 8. Deterministic quantization error is a group-shared perturbation and cannot be GRPO's exploration mechanism: a Jensen bound on degenerate groups

- **Stage:** quantized-rollout RL (QeRL-type claims).
- **Hypothesis.** Fixed quantization error, or AQN noise resampled once per step, perturbs all G rollouts of a group identically. For equal marginal success probability, a group-shared perturbation raises the probability that the group is degenerate (all-correct or all-wrong, zero advantage) relative to per-sample perturbations. Hence the "quantization improves exploration" gains come from entropy/temperature effects, not from parameter-space diversity. Per-sample perturbation at matched KL strictly reduces wasted groups.
- **Gap evidence.**
  - 2510.11696 §3.2 ("a key limitation of quantization errors is their deterministic nature") motivates AQN. The OpenReview rebuttal shows BF16 LoRA + AQN at 89.7 vs NVFP4 LoRA + AQN at 90.8 (theorems note §4), i.e. generic noise explains most of the gain.
  - 2605.20402 abstract: "grid noise raises the policy's entropy".
  - Neither paper distinguishes group-shared from per-sample perturbations.
- **Math core.**
  - Theorem: let the per-rollout success be Bernoulli(p'). Under a shared perturbation p' is random per group with mean p̄, so P(degenerate) = E[p'^G + (1−p')^G]. Under independent per-sample perturbations with the same marginal, rollouts are iid Bernoulli(p̄), so P(degenerate) = p̄^G + (1−p̄)^G.
  - By Jensen (x^G + (1−x)^G is convex), shared ≥ independent, with a gap of ≈ ½G(G−1)·Var(p')·(p̄^{G−2} + (1−p̄)^{G−2}).
  - Exact, under the assumption that per-sample perturbations leave the marginal unchanged.
- **Decisive experiment (~1 GPU-week).**
  - GRPO with G=8 on Qwen2.5-1.5B. Arms: BF16; NVFP4 deterministic; NVFP4 + AQN (per-step shared); per-sequence perturbation (multi-LoRA per-request random rank-1 adapters, or per-sequence activation dither), matched on KL(μ‖π).
  - Measure the zero-advantage fraction against the Jensen prediction from the measured Var(p').
  - **Failure criterion:** the per-sample arm does not reduce the degenerate fraction within the CI of the prediction, or there is no downstream difference.
- **Nearest papers / risk.** 2608.09805 (3PO: rollout grouping over sampled parameter perturbations; C3PO "cycles through N independently sampled policies" per group, l.1034) is close in method. 2602.02555 (PSN-RLVR) does per-rollout parameter noise with TIS. The new parts are the quantization-specific refutation and the degenerate-group theorem. **High overlap risk.**
- **Self prior-art queries:** `abs:GRPO AND abs:"parameter noise"` → 0; `abs:GRPO AND abs:noise AND abs:exploration` (date) → 2602.02555 and others; `abs:"zero advantage" OR abs:"zero-advantage"` (date) → 2609.33126, 2608.09805, etc.; `abs:"parameter space noise" AND abs:exploration AND abs:"language model"` (adjacent RL vocabulary) → 0.
- **Verdict.** Novelty low-medium. Keep only as a short decisive-finding paper or as an ablation inside idea 7.

## 9. Resolving the NVIDIA vs Chmiel backward-switch conflict: transposition mismatch vs re-warm confound

- **Stage:** FP4 pretraining, late precision switch.
- **Hypothesis.** Chmiel et al.'s gain from switching the backward pass to high precision (and NVIDIA's null result) is explained by (a) the persistent fwd/bwd weight mismatch of 1D-block recipes, Q_row(W) vs Q_col(Wᵀ), which a backward switch halves, against NVIDIA's 2D recipe, where a backward switch **creates** a mismatch; and (b) Chmiel's +40B-token LR re-warm protocol. A 2×3×2 design separates the two.
- **Gap evidence.**
  - 2509.25149 App. D: "In contrast to Chmiel et al. (2025), which reports loss recovery from switching precision in the backward pass, we observe no such improvement" (theorems note Q2).
  - 2505.19115 l.371–377 (forward Q_RtN(W), backward Q_RtN(Wᵀ)Q_SR(δ)) and l.675–684 (QAF re-warm, +40B tokens, compared against the 1T BF16 point) (h00).
  - 2607.24953 calls the mismatch "biased" qualitatively, with no quantification (grep: no theorem, no persistence analysis).
- **Math core.**
  - Lemma (h00 Lemma 2): the mismatch bias (W_b − W_f)ᵀδ_y is strongly autocorrelated across steps because W moves ≪ Δ per step.
  - Predicted switch gain: ΔL_bwd ≈ ½[b_beforeᵀH⁻¹b_before − b_afterᵀH⁻¹b_after] + (η/4)tr(Σ_q). Positive for 1D recipes, ≤ 0 for 2D. Standard quadratic analysis.
- **Decisive experiment (1.5–2 H100-weeks).**
  - 350M–1B models. Factors: recipe {1D Chmiel, 2D NVIDIA, Quartet II (re-quantizes the forward tensor)} × switch {fwd, bwd, both} × protocol {token-matched branch, +re-warm}.
  - Log ‖W_b − W_f‖ and the gradient-bias autocorrelation.
  - **Failure criterion:** the backward-switch gain does not differ in sign between the 1D and 2D recipes under the token-matched protocol.
- **Nearest papers.** 2505.19115, 2509.25149, 2607.24953, 2601.22813 (Quartet II requantizes the forward tensor, l.710, so it is a "no-mismatch" third arm), 2609.39813 (backward-state policy, closest conceptual neighbour).
- **Self prior-art queries:** `abs:transposition AND abs:quantization AND abs:training` (date) → 2607.24953 only; `abs:FP4 AND abs:training AND abs:backward AND abs:bias` (date) → 0; `abs:"chain rule" AND abs:quantization AND abs:gradient` (adjacent) → STE-theory papers only; `abs:"backward pass" AND abs:"higher precision" AND abs:switch AND abs:FP4` (date) → 0; `abs:"Chmiel" OR abs:"quantization-aware finetuning" AND abs:FP4` (date) → 0.
- **Verdict.** Novelty low-medium (h00 scored 2/5). A diagnostic paper; it resolves a published contradiction.

## 10. Quantization-variance-corrected Adam preconditioner for SR backward passes (DP-AdamBC transplanted, with element-wise known variance)

- **Stage:** FP4 training optimizer (gradient-quantization × optimizer interaction).
- **Hypothesis.** SR in the backward pass adds element-wise variance σ_q² that can be computed exactly from the pre-rounding residuals, Var = Σ_t s_b² r_t(1−r_t)x_t². That variance inflates Adam's v and silently lowers the effective step size. Subtracting an EMA of the computed σ_q² from v recovers part of the FP4 gap at small batch, and shifts the optimal LR of FP4 runs toward BF16's.
- **Gap evidence.**
  - 2505.19115 identifies a √3 noise threshold below which quantized training becomes ineffective, but does not touch the preconditioner.
  - 2511.00874 analyses SR × batch for SGD only.
  - 2510.21314 analyses Adam under quantized second-moment **storage**, not gradient-noise inflation.
  - The DP literature solved the analogous problem for a global, known noise level (2312.14334 DP-AdamBC, 2304.11208).
- **Math core.**
  - Prop.: at Adam stationarity, v ≈ μ² + σ_d²/B + σ_q², so the update magnitude is scaled by (1 + σ_q²/(μ² + σ_d²/B))^{-1/2}. The corrected v̂ = max(v − σ̂_q², ε') is unbiased for the noise-free v when σ̂_q² is unbiased.
  - σ̂_q² is computable with one extra GEMM of squared operands ((s²r(1−r))ᵀ x²) in low precision, or with a Hutchinson sketch.
- **Decisive experiment (~1 GPU-week).**
  - Step 0 (kill switch): measure σ_q²/(σ_d²/B) per tensor at 125M–350M NVFP4 with SR. A rough estimate from Quartet II Table 1 (SR relative MSE ≈ 2.4%) suggests it is small at standard batch sizes.
  - If the ratio is ≥ 5% for a material set of tensors, run the corrected vs plain Adam LR sweep at batch {64k, 256k} tokens.
  - **Failure criterion:** ratio < 5% (effect negligible, drop the idea) or no LR-optimum shift.
- **Nearest papers.** 2312.14334 / 2304.11208 (DP bias correction, global noise), 2511.00874, 2510.21314, 2505.19115.
- **Self prior-art queries:** `abs:"critical batch size" AND abs:quantization` → 2505.23725 (MuLoCo, unrelated); `abs:"gradient variance" AND abs:FP4` (date) → 0; `abs:"bias correction" AND abs:"second moment" AND abs:noise` (date) → FIBER (DP), unrelated; `abs:"DP-Adam" AND abs:"second moment"` (adjacent DP vocabulary) → 2304.11208, 2312.14334, 2511.07843; `abs:"quantization noise" AND abs:"stochastic rounding" AND abs:training` (date) → 2511.00874 (preempts the batch part).
- **Verdict.** Novelty medium, prior probability of a meaningful effect low. The weakest candidate; run step 0 before anything else.

---

## Ranking (cold)

| Rank | Idea | Novelty | Provable core | Main-track odds |
|---|---|---|---|---|
| 1 | 5 Shared-seed SR across DP + stratified dither | high (premise to verify) | exact covariance theorem | medium |
| 2 | 4 Sorted token blocking for Wgrad in RL | medium-high | exact partition optimality under the separable model | medium |
| 3 | 1 Residual linearization vs "multiplicative compounding" | medium | Props 1–3 (exact/first-order) | medium (refutation + mechanism) |
| 4 | 3 QAT displacement budget / cooldown fusion | medium-high | toy lemma | medium |
| 5 | 2 Pretraining precision → QAT fraction | medium | toy + known implicit-reg. | medium |
| 6 | 6 Rotation reuse bias floor | medium | quadratic bias floor | low-medium (pair with 5) |
| 7 | 9 NVIDIA–Chmiel conflict | low-medium | known quadratic | low-medium |
| 8 | 7 TIS length drift | medium-low | applied TIS bias | low-medium |
| 9 | 8 Group-shared noise Jensen | low-medium | exact Jensen | low |
| 10 | 10 Variance-corrected Adam | medium (likely null) | exact | low |
