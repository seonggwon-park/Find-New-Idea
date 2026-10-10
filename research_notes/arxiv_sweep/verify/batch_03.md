# Citation verification — batch 03

IDs: 2605.08114, 2402.02750, 2510.05373, 2608.20988
Report: `reports/LLM 양자화 연구 아이디어 10선.md` (Idea 2 / "softmax가 분산을 증폭" 절, line 52 and the novelty paragraph).
Source texts: `scratchpad/papers/<id>/<id>.txt` (arxiv_fetch.py). All four papers exist on arXiv. Paper text treated as data; quotes are verbatim from the extracted text.

---

## 2605.08114 — D'Alberto, "Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant"

- **Exists:** yes. Single author (Paolo D'Alberto, AMD). Published 2026-04-27 (v1 only). 23 pages.
- **Venue:** preprint / unknown. arXiv comment: "23 pages, 7 Figures, multiple tables, the process is highly assisted by AI". No venue found by web search.
- **Report tag:** [S] (plus "?" in the JSON). Cited twice (line 52 and the novelty paragraph "D'Alberto ... 인접하므로").

### Claims checked

| Report claim | Verdict | Where in paper |
|---|---|---|
| "QJL이 K의 내적 분산을 π/2배 키우고" | confirmed, with a factor nuance | Sec 4.2 Eq. (13): `Var[δ]QJL-K / Var[δ]scalar = (π/2)/ϵB`; Observation 1: "inflates the inner product variance by a factor (π/2)/ϵB ≥ π/2, independent of dimension d". Under the *fair-budget* comparison ((B−1)-bit MSE + 1-bit QJL vs B-bit MSE) the paper's factor is **2π**, Eq. (23): "σ²QJL/σ²MSE = 2π ≈ 6.28 ... QJL always loses by a constant factor of 2π under fair budget". Sec 3.3 also says "the cost of 2π times more variance". So "π/2" is the lower bound of one argument; the paper's headline fair-budget number is 2π. |
| "softmax가 Jensen으로 증폭한다" | confirmed | Sec 3.1 Eq. (4): `E[e^{⟨q,ki⟩+ϵi}] = e^{⟨q,ki⟩}·e^{σi²/2}`; Sec 3.4: "The convexity of e^x (Jensen's inequality) amplifies K direction errors superlinearly into KL." |
| "정성적 주장이며" | partially confirmed | The π/2 and 2π ratios are explicit derivations (Eqs. 10–13, 21–23) under stated assumptions ("This argument assumes WHT achieves its normalisation goal"), and the paper runs a large simulation study with Mann–Whitney / permutation tests (e.g. "KQV achieves 2.25× lower KL than QKQV (MW r = −0.983, p < 10⁻²³)", Table 3). The author himself calls the KL bridge "a qualitative mechanism" (Conclusions, Open problem). So: quantitative variance ratios + qualitative softmax mechanism; "정성적" is fair for the softmax part only. |
| "토큰 간 분산 이질성이 만드는 체계적 편향 ... 은 없다" | **incorrect** | The paper states exactly this mechanism. Sec 3.1: "If σi² is uniform across keys the factor e^{σ²/2} cancels in the softmax ratio. The dangerous regime is when σi² is correlated with the attention score: high-attention keys with large quantisation variance produce asymmetric distortion that cannot cancel." Sec 3.4: "When σi² is non-uniform across keys — the generic case under any quantisation scheme — the inflation is asymmetric and KL grows superlinearly". What is missing is a closed-form expression for the resulting bias and any correction. |
| "보정식 ... 은 없다" | confirmed | No debiasing / correction term anywhere (grep for debias, subtract, σ²/2 correction: none beyond Eq. 4). |
| "K-vs-V 할당 규칙은 없다" | partially confirmed | There is a *qualitative* K–V rule — QJL on V only: Sec 4 title "The K–V Asymmetry: Why QJL on Keys Harms Attention Quality"; Sec 3.3: "making QJL's unbiased correction genuinely useful here [V] without the 2π variance penalty that makes it harmful for K"; Conclusions item 5 "K–V asymmetry holds at the softmax level". There is **no quantitative bit-allocation rule** between K and V. |

### Correction for the report (line 52)
Replace "...는 '...' 는 정성적 주장이며, 토큰 간 분산 이질성이 만드는 체계적 편향, 보정식, K-vs-V 할당 규칙은 없다" with something like:
"D'Alberto는 QJL-on-K가 내적 분산을 (π/2)/ε_B ≥ π/2배(동일 비트 예산 비교에서는 2π배, Eq. 23) 키우고, softmax의 Jensen 항 e^{σ_i²/2}가 σ_i²가 어텐션 점수와 상관될 때 상쇄되지 않는 비대칭 왜곡을 만든다는 점(Sec 3.1 Eq. 4, Sec 3.4)과 'QJL은 V에만'이라는 K–V 비대칭 규칙(Sec 4)을 제시했다. 그러나 이 편향의 닫힌 식, 비용 0의 보정, 정량적 K/V 비트 할당 규칙은 없고, 실험은 합성 키 분포(gaussian/heavy-tail/low-rank/focused) 시뮬레이션에 한정된다."
Also soften "정성적 주장": the variance ratios are derived; only the softmax-amplification step is qualitative.

### New relevant facts
- Simulation only: "Three key distributions are tested: gaussian (K ∼ N(0, I)), heavy tail ..., low-rank ..." plus "focused"; no real LLM KV caches. "The current experiments are implemented in ROCm HIP on GPU with simulation-side Python post-processing."
- Internal factor inconsistency worth knowing when citing: π/2 (Sec 4.2) vs 2π (Sec 3.3, Sec 4.2 "Unified 2π Argument", Eq. 23).
- Reports a budget-dependent crossover the π/2 argument does not predict: "QKQV achieves better geometric K reconstruction at n ∈ {2,3,5}, KQV at n ∈ {4,6}" — "an open rate-distortion problem".
- **Novelty risk for Idea 2:** the paper's Eq. (4) is already a second-moment (variance) argument about heteroscedastic σ_i² across keys. The report's differentiation must rest on the closed-form bias expression, the correction, and the allocation rule — not on "identifying the heteroscedastic Jensen bias", which D'Alberto already names.
- Single-author, AI-assisted, non-peer-reviewed; cite as a preprint.

---

## 2402.02750 — Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"

- **Exists:** yes. First author Zirui Liu. Published 2024-02-05, updated 2024-07-25 (v2).
- **Venue:** ICML 2024 (arXiv comment "ICML2024").
- **Report tag:** [V].

### Claims checked
| Report claim | Verdict | Where in paper |
|---|---|---|
| "per-channel K ... 'outlier 채널'로 정당화" | confirmed | Sec 3.2: "in key cache, some fixed channels exhibit very large magnitudes, whereas in value cache, there is no significant pattern for outliers." ... "per-channel quantization can confine the quantization error to each individual channel without impacting the other normal channels. Thus, Figure 2 explains why key cache should be quantized per-channel." Table 2 (Llama-2-13B, avg over layers/heads): relative attention-score error ‖A−A′‖/‖A‖ = 47.00 (K per-token) vs 9.60 (K per-channel) — "almost 5× larger"; K reconstruction error 13.67 vs 4.55. |
| "per-token V ... '스트리밍 편의'로 정당화" | partially confirmed | Streaming is given as the *implementation* reason (Sec 3.3: "per-token quantization aligns well with streaming settings"). The paper's stated *accuracy* justification for V per-token is **attention sparsity**, not streaming: Sec 3.2 "the per-token quantization error is almost 15× smaller than per-channel quantization ... The intuition behind this observation stems from the attention sparsity" (Eq. 2); Table 2: Avg Δ = 3.55 (V per-token) vs 49.89 (V per-channel), attention sparsity 84.3%. Note V *reconstruction* error is actually slightly worse per-token (4.57 vs 3.73) — the win is only in the attention-output error Δ. |
| "최적성 증명이 없다" | confirmed | No theorem/lemma/proof anywhere (grep: none). Justification is empirical (Tables 1–3, Figure 2) and heuristic ("confine the error"). |

### Correction for the report (line 52 / 135)
Change "'outlier 채널'과 '스트리밍 편의'로 정당화됐고" to "'K의 고정 outlier 채널'(Table 2: 어텐션 점수 오차 47.00→9.60, ≈5×)과 'V는 어텐션 희소성(84.3%) 때문에 per-token이 출력 오차 Δ를 49.89→3.55로 ≈15× 줄인다'는 경험적 관찰로 정당화됐고(스트리밍 정합성은 구현상의 이유), 최적성 증명이 없다".

### New relevant facts
- KIVI's own V analysis (Eq. 2, attention sparsity → per-token confines error to unimportant tokens) is a mechanism the report's Idea 2 should engage with: it is an argument about *which tokens' errors matter*, i.e. about weighting by attention, and is adjacent to the heteroscedastic-variance framing.
- Headline systems numbers (abstract): 2.6× less peak memory, up to 4× larger batch, 2.35×–3.47× throughput.

---

## 2510.05373 — Saxena & Roy, "KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction"

- **Exists:** yes. Purdue. Published 2025-10-06 (v1 only). arXiv comment "14 pages, 7 figures, 6 tables". Running header "Preprint. Under Review."
- **Venue:** preprint / under review; no OpenReview or proceedings page found by web search as of 2026-10-10.
- **Report tag:** [S].

### Claims checked
| Report claim | Verdict | Where in paper |
|---|---|---|
| "'raw key per-channel + Hadamard value per-token'이 최선이라 보고" | confirmed | Observation 3: "Quantizing raw keys channel-wise together with Hadamard rotated values token-wise (Kq = QC(K), Vq = QT(VH)) emerges as the optimal configuration across all model families. This even outperforms the Kq = QC(KH), Vq = QT(VH) quantization scheme." Fig. 3: Wikitext PPL, 2-bit, group size 128, Llama-3.2-3B / Qwen-2.5-3B / Qwen-3-4B. |
| "KVLinC는 반대로 ..." (framed as the opposite of KIVI) | **incorrect framing** | KVLinC uses the *same* axes as KIVI (K per-channel, V per-token) and adds Hadamard to V only. Observation 2: "At the low-precision regime under consideration, KIVI's quantization configuration consistently outperforms QuaRot's." It confirms KIVI's axis choice rather than contradicting it; the finding that contrasts with intuition is that Hadamard on K *hurts* under channel-wise quantization. |
| "설명이 없다" | **incorrect** (a heuristic explanation is given; no theory) | Observation 3: "This is because the application of Hadamard rotation to keys redistributes each outlier dimension, thereby increasing the scaling factor of quantization leading to higher error (Figure 4)." Observation 2: "although, Hadamard rotation of keys reduces scaling factor and hence the quantization error with token-wise quantization, it still is much higher than channel-wise quantization of keys." Observation 1 (pre- vs post-multiplication): "pre-multiplication mixes tokens prior to quantization, which amplifies quantization noise". These are scaling-factor / outlier-redistribution heuristics backed by Fig. 4; there is no derivation or optimality argument. |

### Correction for the report (line 52 / 144)
Replace "KVLinC는 반대로 'raw key per-channel + Hadamard value per-token'이 최선이라 보고했는데 설명이 없다" with: "KVLinC는 KIVI와 같은 축(K per-channel, V per-token)을 유지하고 Hadamard를 V에만 적용한 구성이 3개 모델군(Llama-3.2-3B, Qwen-2.5-3B, Qwen-3-4B; 2-bit, g=128)에서 최선이며, K에 Hadamard를 걸면 outlier 차원이 재분배되어 그룹 스케일이 커져 오히려 오차가 늘어난다는 스케일 팩터 기반의 경험적 설명(Fig. 4)만 제시한다 ([2510.05373] [S]). 이론적 근거는 없다."

### New relevant facts
- Table 3 (Llama family, Wikitext PPL ↓ / 5-shot GSM8K ↑): Llama-3.1-8B — KIVI 7.8/34.1, KIVI+LinC 7.3/38.4, QC(K),QT(VH) 7.2/36.9, KVLinC 7.1/40.9; Llama-3.2-3B — KIVI 11.0/11.8, KIVI+LinC 9.8/14.5, QC(K),QT(VH) 9.7/13.9, KVLinC 9.4/16.4. I.e. the Hadamard-on-V axis change alone is worth ~0.6–1.3 PPL over KIVI.
- "For the Qwen-2.5 and Qwen-3 family of models, QuaRot fails to produce meaningful results, showcasing that per token quantization strategy for both keys and values is sub-optimal."
- Linear correction adapters are *trained* (Alpaca 500 steps / RedPajama 1500 steps, 4×H200) — not tuning-free like KIVI. Kernel: "up to 2.55× faster inference and up to 3.5× larger batch size".
- Its explanation is in terms of the per-group scaling factor (max-abs range), which is a first-moment/range argument, not a variance or softmax-sensitivity argument — a gap the report's idea can legitimately claim.

---

## 2608.20988 — Pandey et al., "Jacobian-guided Noise Injection for Quantization Robustness in Large Language Models"

- **Exists:** yes. Amazon. Published 2026-08-21 (v1 only).
- **Venue:** arXiv comment "Accepted at AdaptFM: Resource-Adaptive Foundation Model Inference, ICML 2026" — an **ICML 2026 workshop**, not the main track (the PDF carries the PMLR 306 proceedings header template; icml.cc/virtual/2026/75117 lists it). Also listed on amazon.science.
- **Report tag:** [S].

### Claims checked
| Report claim | Verdict | Where in paper |
|---|---|---|
| "1차 Jacobian 상한" (first-order Jacobian upper bound on attention sensitivity) | confirmed | Sec 2.1 Eq. (2)–(3): "Using first-order Taylor expansion, we approximate this error via the jacobian JS(z) ∈ R^{N×N}: Δa ≈ JS(z)δ ⟹ ‖Δa‖₂ ≤ ‖JS(z)‖₂‖δ‖₂". It is a first-order *approximation* presented as a bound, not a theorem; the Hessian link Eq. (8) "‖JS(z)‖₂ ≤ sqrt(Tr(∇²z L)/λmin)" relies on a Gauss-Newton approximation and λmin > 0 near a local optimum. |
| Implicit: the Jacobian bound says sharp (peaked) attention is *less* sensitive | confirmed, and the paper flags the tension itself | Eq. (12): "‖JS‖²F = ‖p‖²₂ − 2‖p‖³₃ + ‖p‖⁴₄" → 0 for one-hot p. Sec 3.3: "while zero Jacobian norm is achievable when probability mass concentrates on a single token, this regime is somewhat problematic since: (1) sparse attention degenerates to near-constant outputs ... and (2) large logit magnitudes amplify quantization errors exponentially through the softmax gate". |
| "어텐션 민감도 이론은 ... 뿐이고" (only these two theories exist) | not strictly checkable; overclaim | At minimum D'Alberto 2605.08114 Eq. (4) (Jensen e^{σ²/2}, a second-moment argument) is a third account the report itself cites. |

### Correction for the report (line 153 / 162)
Keep "1차 Jacobian 상한([2608.20988] [S])" but (i) mark the venue as ICML 2026 *workshop* (AdaptFM), (ii) note that 2608.20988 itself acknowledges the first-order bound's blind spot at peaked attention ("large logit magnitudes amplify quantization errors exponentially through the softmax gate"), so the "두 상한의 모순" is partly admitted in the source rather than discovered by the report, and (iii) drop "뿐이고" or qualify it, since D'Alberto's Jensen term is already a second-moment mechanism.

### New relevant facts
- Method: per-position Gaussian noise on pre-softmax logits with σ_i = α·sqrt(E_{b,h}‖JS‖²_{F,i}) (Eq. 13); training-time only, "At inference time, noise injection is disabled (σ = 0)". Claims λ = σ²/2 correspondence between noise variance and Hessian-trace regularisation (Sec 2.3) — derived via a second-order Taylor expansion of the *loss* (Eq. 10), so the paper is "first-order" only for the attention-error bound, not for the regulariser.
- Headline numbers: "up to +37% relative gains on Top-1 accuracy on ImageNet-1K for SigLIP" (RepQViT W4A4; +9% with ERQ) and "improves relative perplexity by upto 40% on WikiText". LLM experiments: Llama-3.2-3B and Qwen2.5-3B fine-tuned on Alpaca, PTQ with AWQ/GPTQ/SpinQuant (Table 1, Tables 4–5). α sweep (SigLIP W4A4 QAT): α=0.5 → 31.37% vs 30.31% baseline (+1.06 pt); α=50 → 27.69%.
- Empirically the trained models get *more diffuse* attention ("Noise-trained models exhibit more diffuse attention patterns"), i.e. the method pushes toward the high-Jacobian-norm-but-small-logit regime, which is itself in tension with the Eq. (3) reading — useful ammunition for the report's second-order reconciliation argument.
- Small models only (3B); workshop paper; no KV-cache-quantization experiments despite the report citing it in a KV-cache context.
